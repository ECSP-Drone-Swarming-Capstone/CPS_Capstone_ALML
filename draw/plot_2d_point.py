import airsim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()


# Initialize variables to store current positions
current_position1 = (0, 0)
current_position2 = (0, 0)
current_position3 = (0, 0)
current_center = (0,0)

# Update drone positions function
def update_drone_positions():
    global current_position1, current_position2, current_position3, current_center
    drone_position1 = client.simGetGroundTruthKinematics(vehicle_name = "Drone1").position
    #print(drone_position1.x_val)

    current_position1 = (drone_position1.x_val, drone_position1.y_val)

    drone_position2 = client.simGetGroundTruthKinematics(vehicle_name = "Drone2").position
    current_position2 = (drone_position2.x_val, drone_position2.y_val)

    drone_position3 = client.simGetGroundTruthKinematics(vehicle_name = "Drone3").position
    current_position3 = (drone_position3.x_val, drone_position3.y_val)

    current_center = ((current_position1[0]+current_position2[0]+current_position3[0])/3, (current_position1[1]+current_position2[1]+current_position3[1])/3)

    collision = client.simGetCollisionInfo().has_collided
    get_reward(collision)

def get_reward(collision):
    pts = np.array([5,5])
    quad_pt = np.array(
        list(
            (
                current_position1[0],
                current_position1[1],
                current_position2[0],
                current_position2[1],
                current_position3[0],
                current_position3[1],
            )
        )
    )
    reward = 0
    done = 0
    if collision:
        reward = -100
        done = 1
        print(f"done(collision):{reward}")
        return reward, done
    else:
        reward_dist = 0
        reward_dist_connection = 0
        RDist = 0
        #print("Start")
        l = 3
        for i in range(l):
            for j in range(l):
                if i>=j:
                    continue
                dist = np.linalg.norm(quad_pt[2*i:2*(i+1)] - quad_pt[2*j:2*(j+1)])
                RDist = max(RDist, dist)
        #print(dist)
        if RDist > 5:
            reward_dist_connection += -20
            print(f"done(Loss Conection):{RDist}")
        elif RDist>1:
            reward_dist_connection = -0.1*RDist

        RDist = 0 
        l = 3
        for i in range(l):
            #print(quad_pt[2*i:2*(i+1)])
            dist = np.linalg.norm(quad_pt[2*i:2*(i+1)] - pts)
            RDist = max(RDist, dist)
            
        if RDist > 10:
            reward_dist = -10
            print(f"done(Too Far):{RDist}")
        elif RDist>3:
            reward_dist = -1*(RDist-3)
        else:
            reward_dist = -5*(RDist-3)

        reward = reward_dist + reward_dist_connection

        if reward_dist <= -10 or reward_dist_connection <= -20:    
            done = 1
            return reward, done

        print(f"reward:{round(reward,2)}, reward_dist:{round(reward_dist,2)}, reward_dist_connection:{round(reward_dist_connection,2)}, dist:{round(RDist,2)}")

    return reward, done
# Update function for the animation
def update(num):
    update_drone_positions()
    ax.clear()  # Clear the previous frame
    ax.set_xlim(-20, 20)  # Set appropriate limits based on your environment
    ax.set_ylim(-20, 20)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    

    ax.plot(*current_position1, 'bo', markersize=2, label='Drone 1')
    ax.plot(*current_position2, 'go', markersize=2, label='Drone 2')
    ax.plot(*current_position3, 'ro', markersize=2, label='Drone 3')
    ax.plot(*current_center, 'yo', markersize=2, label='Center')
    ax.legend()

    ax.set_xticks(range(-20, 20, 2))  # X-axis ticks from -50 to 50 with a step of 10
    ax.set_yticks(range(-20, 20, 2))  # Y-axis ticks from -50 to 50 with a step of 10
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)  # Horizontal line at y=0
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)  # Vertical line at x=0
    
    ax.axhline(5, color='gray', linestyle='--', linewidth=1)  # Horizontal line at y=0
    ax.axvline(5, color='gray', linestyle='--', linewidth=1)  # Vertical line at x=0
    # ax.axhline(10, color='gray', linestyle='--', linewidth=1)  # Horizontal line at y=0
    # ax.axvline(10, color='gray', linestyle='--', linewidth=1)  # Vertical line at x=0
    # ax.axhline(8, color='gray', linestyle='--', linewidth=1)  # Horizontal line at y=0
    # ax.axvline(12, color='gray', linestyle='--', linewidth=1)  # Vertical line at x=0
  

# Set up the plot


# Set up the plot
fig, ax = plt.subplots()

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=None, blit=False, interval=100)

# Display the animation
plt.show()
