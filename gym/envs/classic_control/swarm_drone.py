#import setup_path
from ipaddress import ip_address
from itertools import count
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from gym.envs.classic_control.airsim_env import AirSimEnv
import time

class SwarmDroneEnv(AirSimEnv):
    #def __init__(self, ip_address, step_length, image_shape):
    def __init__(self):    
        ip_address = "127.0.0.1"
        image_shape = (84, 84, 1)
        step_length = 0.4
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        n = 3
        self.state = {
            "position": np.zeros(3*3),
            "velocity": np.zeros(3*3),
            "acceleration": np.zeros(3*3),
            "orientation": np.zeros(3*4),
            "collision": False,
            "prev_position": np.zeros(3*3),
            "prevelocity": np.zeros(3*3),
            "preacceleration": np.zeros(3*3),
            "preorientation": np.zeros(3*4),
            "motorthrust": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient()
      
        m = 3
        M = 14
        #self.action_space = spaces.MultiDiscrete(np.ones((n, m)))
        self.action_space = spaces.Discrete(M*M*M)
        #spaces.Box(0, 255, shape=(n,m), dtype=np.float32)
        self._setup_flight()

        numoffeature = 13

        
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(n, numoffeature+m), dtype=np.float32)
        #self.observation_space = spaces.MultiDiscrete()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        poss = [[1, 1, -2.0, 1],[0,0, -1, 0.5],[-1,1,-2.0, 1]]
        #vel = [[0.5, 0, -0.2],[0.5, -0.2, -0.2],[-0.5, 0, -0.2]]
        #print("set up flight")
        self.drone.reset()
        l = len(self.drone.listVehicles())
        drones1 = []
        #drones2 = []
        for i in range(l):
            name = "Drone"+str(i+1)
            #print(name)
            self.drone.enableApiControl(True, name)     # 获取控制权
            self.drone.armDisarm(True, name)            # 解锁（螺旋桨开始转动）
            d = self.drone.moveToPositionAsync(poss[i][0], poss[i][1], poss[i][2], poss[i][3], vehicle_name=name)
            
            drones1.append(d)
        drones1[l-1].join()
        # time.sleep(5)

    def _get_obs(self):
        
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["prev_velocity"] = self.state["velocity"]
        self.state["prev_acceleration"] = self.state["acceleration"]
        self.state["pre_orientation"] = self.state["orientation"]

        l = len(self.drone.listVehicles())
       
        state = np.zeros((l, 13+3))#3*13

        for i in range(l):
            name = "Drone"+str(i+1)
             
            state_rotor = self.drone.getRotorStates(vehicle_name = name)
            kinematic_state_groundtruth = self.drone.simGetGroundTruthKinematics(vehicle_name = name)

            self.state["position"][3*i] = kinematic_state_groundtruth.position.x_val
            self.state["position"][3*i+1] = kinematic_state_groundtruth.position.y_val
            self.state["position"][3*i+2] = kinematic_state_groundtruth.position.z_val

            self.state["velocity"][3*i] = kinematic_state_groundtruth.linear_velocity.x_val
            self.state["velocity"][3*i+1] = kinematic_state_groundtruth.linear_velocity.y_val
            self.state["velocity"][3*i+2] = kinematic_state_groundtruth.linear_velocity.z_val

            self.state["acceleration"][3*i] = kinematic_state_groundtruth.linear_acceleration.x_val
            self.state["acceleration"][3*i+1] = kinematic_state_groundtruth.linear_acceleration.y_val
            self.state["acceleration"][3*i+2] = kinematic_state_groundtruth.linear_acceleration.z_val

            self.state["orientation"][4*i] = kinematic_state_groundtruth.orientation.x_val
            self.state["orientation"][4*i+1] = kinematic_state_groundtruth.orientation.y_val
            self.state["orientation"][4*i+2] = kinematic_state_groundtruth.orientation.z_val
            self.state["orientation"][4*i+3] = kinematic_state_groundtruth.orientation.w_val
            
            self.state["motorthrust"][i] = 0
            #print(state_rotor.rotors)
            for a in range(4):
                self.state["motorthrust"][i]+=state_rotor.rotors[a]["thrust"]

            state[i][0:3] = self.state["position"][3*i:3*(i+1)]
            state[i][3:6] = self.state["velocity"][3*i:3*(i+1)]
            state[i][6:9] = self.state["acceleration"][3*i:3*(i+1)]
            state[i][9:13] = self.state["orientation"][4*i:4*i+4]
            #print(f"{i},{kinematic_state_groundtruth.position.x_val},{kinematic_state_groundtruth.position.y_val},{kinematic_state_groundtruth.position.z_val}")
            #print("state[i]:", state[i])
        #print(f"position:", state[:,0:3])    
        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision
        state[:,13:] = np.ones((3,3))#Adcency Matrix
        return state

    def _do_action(self, action):
        #print("action",action)
        l = len(self.drone.listVehicles())
        #print("l",l)
        tmp = self.interpret_action(action)
       # print("tmp")
        drones = []
        for i in range(l):
            name = "Drone"+str(i+1)
            self.drone.enableApiControl(True, name)   
            kinematic_state_groundtruth = self.drone.simGetGroundTruthKinematics(vehicle_name = name)
            quad_vel = kinematic_state_groundtruth.linear_velocity
            #控制权
            self.drone.armDisarm(True, name) 
            
            # if i != l-1:      
                # Set home position and velocity
                #print("action:", i,float(tmp[i][0]),float(tmp[i][1]),float(tmp[i][2]))
            d = self.drone.moveByVelocityAsync(quad_vel.x_val+float(tmp[i][0]), quad_vel.y_val+float(tmp[i][1]), float(tmp[i][2]), 2, vehicle_name=name)
            # d = self.drone.moveByVelocityAsync(float(tmp[i][0]), float(tmp[i][1]), float(tmp[i][2]), 2, vehicle_name=name)
            #self.drone.hoverAsync(vehicle_name = name)
            # print("weizhi")
            drones.append(d)
            # else:
                # Set home position and velocity
                #print("action:",i,float(tmp[i][0]),float(tmp[i][1]),float(tmp[i][2]))
                # self.drone.moveByVelocityAsync(quad_vel.x_val+float(tmp[i][0]), quad_vel.y_val+float(tmp[i][1]), float(tmp[i][2]), 5, vehicle_name=name).join()
                #self.drone.moveToZAsync(0, 1, vehicle_name=name).join()
               #self.drone.hoverAsync(vehicle_name = name).join()
        # for i in range(l):
        drones[l-1].join()
        #time.sleep(0.5)
    
    def _compute_reward(self):
        thresh_dist1 = 25
        thresh_dist2 = 12.5
        thresh_dist3 = 3
        beta = 0.1
        done = 0 
        z = -10
        pts = np.array([5,5])

        quad_pt = np.array(
            list(
                (
                    self.state["position"][0],
                    self.state["position"][1],
                    

                    self.state["position"][3],
                    self.state["position"][4],
                    

                    self.state["position"][6],
                    self.state["position"][7],
                    
                )
            )
        )

        l = len(self.drone.listVehicles())
        #print("in")
        reward = 0
        if self.state["collision"]:
            reward = -100
            done = 1
            return reward, done
        else:
            #dists =[]
            reward_dist = 0
            reward_dist_connection = 0
            RDist = 0
            #print("Start")
            for i in range(l):
                for j in range(l):
                    if i>=j:
                        continue
                    dist = np.linalg.norm(quad_pt[2*i:2*(i+1)] - quad_pt[2*j:2*(j+1)])
                    RDist = max(RDist, dist)
            #print(dist)
            if RDist > 5:
                reward_dist_connection += -20
            elif RDist>1:
                reward_dist_connection = -0.1*RDist
            
            RDist = 0 
            # pt = np.array([0.0, 0.0])
            # ptg = np.array([0.0, 0.0])
            for i in range(l):
                dist = np.linalg.norm(quad_pt[2*i:2*(i+1)] - pts)
                RDist = max(RDist, dist)

            if RDist > 10:
                reward_dist = -10
            elif RDist>3:
                reward_dist = -1*(RDist-3)
            else:
                reward_dist = -5*(RDist-3)
            reward = reward_dist + reward_dist_connection

            #calculate energy consumption Thrust
            reward_energy_power = 0
            for i in range(l):
                reward_energy_power += self.state["motorthrust"][i]
            #there should be positive and negative. Now the value is only negative
            reward_energy_power = -1*reward_energy_power
            reward += reward_energy_power
            
            if reward_dist <= -10 or reward_dist_connection <= -20:    
                done = 1
                return reward, done
            # print(f"reward:{reward}, reward_dist:{reward_dist}, reward_dist_connection:{reward_dist_connection}")
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        control_action = np.zeros((3, 3), np.float32)
        M = 14
        for i in range(3):

            d_action = action%M
            quad_offset = [0, 0, 0]
            '''
            [0,0]
            [1,0]
            [0,1]
            [-1,0]
            [0,-1]
            [1,1]
            [-1,-1]
            [1,2]
            [1,-2]
            [-1,2]
            [-1,-2]
            [2,1]
            [-2,1]
            [2,-1]
            [-2,-1]
            '''
            if d_action == 0:
                quad_offset = [self.step_length, 0, 0]
            elif d_action == 1:
                quad_offset = [0, self.step_length, 0]
            elif d_action == 2:
                quad_offset = [-1*self.step_length, 0, 0]
            elif d_action == 3:
                quad_offset = [0, -1*self.step_length, 0]
            elif d_action == 4:
                quad_offset = [1*self.step_length, 1*self.step_length, 0]
            elif d_action == 5:
                quad_offset = [-1*self.step_length, -1*self.step_length, 0]
            elif d_action == 6:
                quad_offset = [self.step_length, self.step_length, 0]
            elif d_action == 7:
                quad_offset = [self.step_length, -2*self.step_length, 0]
            elif d_action == 8:
                quad_offset = [-self.step_length, 2*self.step_length, 0]
            elif d_action == 9:
                quad_offset = [-1*self.step_length, -2*self.step_length, 0]
            elif d_action == 10:
                quad_offset = [2*self.step_length, 1*self.step_length, 0]
            elif d_action == 11:
                quad_offset = [-2*self.step_length, self.step_length, 0]
            elif d_action == 12:
                quad_offset = [2*self.step_length, -1*self.step_length, 0]
            elif d_action == 13:
                quad_offset = [-2*self.step_length, -1*self.step_length, 0]
            else:
                quad_offset = [0, 0, 0]



            #print(quad_offset, control_action)
            control_action[i][0] = quad_offset[0]
            control_action[i][1] = quad_offset[1]
            control_action[i][2] = quad_offset[2]

            action //= M
            
            #132-->1 for drone 1, 3 for drone2, 2 for drone 3
            '''
            ABC
            A*M^2+B*M+C

            M = 10
            132%M-->2
            132/M-->13
            13%M-->3
            13/M-->1
            1%M-->1
            ''' 
            # print(f"{i}, action:{np.round(quad_offset,2)}, action no.:{d_action}")
        return control_action