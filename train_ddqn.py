#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_functions

import airsim
drone = airsim.MultirotorClient()

import tensorflow as tf
import shutil
import os
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.policies import policy_saver

import tensorflow as tf
import numpy as np
import zipfile
import io
from tf_agents import specs
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec

#from tf_agents.trajectories import time_steps

import keyboard


dot = tf.matmul

env_name = 'SwarmDrone-v0'
env = suite_gym.load(env_name)

env.reset()

print('Observation Spec:')
print(env.time_step_spec().observation)

print('Reward Spec:')
print(env.time_step_spec().reward)

print('Action Spec:')
print(env.action_spec())

# Defining
M = 14

fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))


class SparseDrop(tf.keras.layers.Layer):
    """
    Sparse dropout layer.
    """

    def __init__(self, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False, **kwargs):
        super(SparseDrop, self).__init__(**kwargs)

        self.dropout = dropout
        self.is_sparse_inputs = is_sparse_inputs
        self.num_features_nonzero = num_features_nonzero

    def call(self, inputs, training=None):
        x = inputs

        # dropout
        if training is not False and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif training is not False:
            x = tf.nn.dropout(x, self.dropout)

        return x

class GraphConv(tf.keras.layers.Layer):
    """
    Graph convolution layer.
    """
    def __init__(self, input_dim, output_dim,
                 is_sparse_inputs=False,
                 bias=False,
                 featureless=False, **kwargs):
        super(GraphConv, self).__init__(**kwargs)

        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.bias = bias

        self.weights_ = []
        for i in range(1):
            w = self.add_variable('weight' + str(i), [input_dim, output_dim])
            self.weights_.append(w)
        if self.bias:
            self.bias = self.add_variable('bias', [output_dim])
        self.output_dim = output_dim
        #print(f"self.weights_:{len(self.weights_)}")
        

    def call(self, inputs, training=None):
        #print(inputs)
        #print(f"self.weights_:{self.weights_}")
        # n = int(inputs.shape[1]/2)
        # x = inputs[:,n:]
        # support_ = tf.reshape(inputs[:,n+1:], (-1,1,1))
        #m = 10 # number of features for each drone
        #print("inputs:", inputs[:,:,0:3])
        x = tf.reshape(inputs[:, :, :inputs.shape[2]-inputs.shape[1]], (-1, inputs.shape[1], inputs.shape[2]-inputs.shape[1]))#tf.reshape(inputs, (-1, 4, 100))
        x = tf.dtypes.cast(x,tf.float32)
        # print(inputs.shape, x.shape)
        support_ = tf.reshape(inputs[:, :, inputs.shape[2]-inputs.shape[1]: ], (-1, inputs.shape[1], inputs.shape[1]))#tf.reshape(inputs, (-1, 4, 100))
        #print(x.shape)
        #print(self.weights_[0].shape)
        output = dot(support_, dot(x, self.weights_[0])) #(n,4,100)*(100,2) = (n,4,2)
        #output = dot(x, self.weights_[0]) #(n,4,100)*(100,2) = (n,4,2)s
        # bias
        if self.bias:
            output += self.bias
        # output = tf.nn.relu(output)
        # a = tf.transpose(output)[1:,:]
        # print(output)
        #print(tf.reshape(output,(-1,3, self.test)))
        output = tf.nn.relu(tf.reshape(output,(-1,3, self.output_dim)))
        output = tf.concat([output, inputs[:, :, inputs.shape[2]-inputs.shape[1]: ]], 2)
        return  output


train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units): # why 2?
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
fc_layer_params = (100, 50)



c1 = GraphConv(input_dim=13,  # 1433
                            output_dim=2000,  # 16
                            is_sparse_inputs=True)
c2 = GraphConv(input_dim=2000,  # 1433
                            output_dim=1000,  # 16
                            is_sparse_inputs=True)
c3 = GraphConv(input_dim=1000,  # 1433
                            output_dim=M*M*M,  # 16
                            is_sparse_inputs=True)
reshape = [tf.keras.layers.Reshape((M*M*M*3+3*3,), input_shape=(-1, 3, M*M*M+3))]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
reshape2 = [tf.keras.layers.Reshape((M*M*M,), input_shape=(-1, 1, M*M*M))]
q_net = sequential.Sequential([c1]+[c2]+[c3]+reshape+[q_values_layer]+reshape2)

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-5  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#q_net.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
train_step_counter = tf.Variable(0)
#spec = tf_agents.specs.BoundedTensorSpec(shape=(), dtype=tf.int64, name='action', minimum=0, maximum=1)
global_step = tf.compat.v1.train.get_or_create_global_step()
agent = dqn_agent.DdqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_step)

#print("LOAD CHECK POINT")


agent.initialize()
q_net.summary()

# tf_env = tf_py_environment.TFPyEnvironment(env)

replay_buffer_capacity = 1000

'''
Trajectory(
{'action': BoundedTensorSpec(shape=(), dtype=tf.int64, name='action', minimum=array(0, dtype=int64), maximum=array(2196, dtype=int64)),
 'discount': BoundedTensorSpec(shape=(), dtype=tf.float32, name='discount', minimum=array(0., dtype=float32), maximum=array(1., dtype=float32)),
 'next_step_type': TensorSpec(shape=(), dtype=tf.int32, name='step_type'),
 'observation': BoundedTensorSpec(shape=(3, 13), dtype=tf.int32, name='observation', minimum=array(0), maximum=array(0)),
 'policy_info': (),
 'reward': TensorSpec(shape=(), dtype=tf.float32, name='reward'),
 'step_type': TensorSpec(shape=(), dtype=tf.int32, name='step_type')})

'''

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

# Add an observer that adds to the replay buffer:
replay_observer = [replay_buffer.add_batch]


collect_steps_per_iteration =  50
collect_op = dynamic_step_driver.DynamicStepDriver(
  train_env,
  agent.collect_policy,
  observers=replay_observer,
  num_steps=collect_steps_per_iteration)
# Initial data collection
#collect_op.run()

# Read the replay buffer as a Dataset,
# read batches of 4 elements, each with 2 timesteps:
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2)

iterator = iter(dataset)

num_train_steps = 25

time_step = None

EVAL_EPISODES = 10
EVAL_INTERVAL = 1000

def get_average_reward(environment, policy, episodes=2):

    total_reward = 0.0

    for _ in range(episodes):
        time_step = environment.reset()
        episode_reward = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_reward += time_step.reward
    
        total_reward += episode_reward
    avg_reward = total_reward / episodes
    
    return avg_reward.numpy()[0]

#important!!! Load Old variants of RL model. all of the vlaue weight and bias will be reset with the old model

tempdir = './'
checkpoint_dirload = os.path.join(tempdir, 'new_checkpoint_09111537')
print("checkpoint dir",checkpoint_dirload)
train_checkpointerload = common.Checkpointer(
    ckpt_dir=checkpoint_dirload,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)
    
train_checkpointerload.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()

policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)

print("LOADING GLOBAL STEP", global_step.numpy())

#Create the new checkpoint to save the updated variables. please use the new file name, should not be same as the existed file name.

tempdir = './'
checkpoint_dir = os.path.join(tempdir, 'new_checkpoint_11112100')
print("checkpoint dir",checkpoint_dir)
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)



#tempdir = './'
#policy_dir = os.path.join(tempdir, 'policy')
#tf_policy_saver = policy_saver.PolicySaver(agent.policy)

'''
#--------FOR SAVING POINT WITH 10 EVERY EPOCH------------#
checkpoint_dir = os.path.join(tempdir, 'checkpoint')
train_checkpointerA = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)
#--------FOR SAVING POINT WITH KEY INTERRUPT------------#
checkpoint_dir = os.path.join(tempdir, 'good_checkpoint')
train_checkpointer_GOOD = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)
#--------FOR SAVING POINT WITH BREAK POINT INTERRUPT------------#
checkpoint_dir = os.path.join(tempdir, 'breakpoint_checkpoint')
train_checkpointer_BREAKPOINT = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)
'''
print("----------------------------------------start training---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

for i in range(10000):
    try:
        #env.reset()
        
        time_step, policy_state = collect_op.run(
            time_step=time_step,
            policy_state=policy_state,
        )
    
        for a in range(num_train_steps):

            trajectories, _ = next(iterator)

            train_loss = agent.train(experience=trajectories).loss
            '''
            if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                print('----------------------------saving the model because you press the key-------------------------------------------------------------------------------------------------------------------')
                
                print("=-----------before save. global step,",global_step.numpy())            
                train_checkpointer.save(global_step)
                break # finishing the loop
            '''
        print(f"epoch {i}, loss: ", train_loss)
     

        if i%20==0:

            print("=-----------before save. global step,",global_step.numpy())   
            train_checkpointer.save(global_step)
          
        if i!=0 and i%100==0:
            reward = get_average_reward(eval_env, agent.policy, num_eval_episodes)
            print(f"epoch {i}, loss: {train_loss}, reward: {reward}")
    except:
        print("----------------Error-----------------------------------------------")
        print('----------------------------saving the model because you press the key-------------------------------------------------------------------------------------------------------------------')
        print("=-----------before save. global step,",global_step.numpy())            
        train_checkpointer.save(global_step)
        break # finishing the loop
