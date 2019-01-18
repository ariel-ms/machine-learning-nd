import numpy as np
import math
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.origin = np.array([0., 0., 0.])

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        x = self.sim.pose[0]
        y = self.sim.pose[1]
        z_target = self.target_pos[2]
        z = self.sim.pose[2]
        
        if z <= z_target:
            reward = np.linalg.norm(np.tanh(self.sim.pose[2]-0))
        elif z > z_target:
            reward = 1 - (np.linalg.norm(np.tanh(self.sim.pose[2]-self.target_pos[2]))*1.7)
            
        reward += 1 - (np.linalg.norm(np.tanh(self.target_pos[:2]-self.sim.pose[:2]*1))*1.5)
#         if (x >= 70 or x <= -70) or (y >= 70 or y <= -70):
#             reward += -0.3
        return -0.75 if z <= 0 or z >= 300 else reward
                
    def print_position(self):
        print(self.sim.pose[:3])
        
    def get_height(self):
        return self.sim.pose[2]

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state