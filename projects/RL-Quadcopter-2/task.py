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
            reward = ((-((z/10)-10)**2) + 100) / z_target
        elif z > z_target:
#             reward = (100 - (z - z_target))/100
            z = 200 if z > 200 else z
            x = (z - z_target) / 100
            reward = (x - 1)**2
        
#         print("x and y and z" + str(x) + " " + str(y) +" " + str(z))
#         print("before " + str(reward))
#         penalty = self.gaussian_func(1, x, y, 0, 0, 20, 20)
#         print("penalty" + str(penalty))
#         reward -=  (1 - penalty)
        reward += 1 - (np.linalg.norm(np.tanh(self.target_pos[:2]-self.sim.pose[:2]))*1.1)
#         print("after " + str(reward))
        return -0.8 if z <= 0 or z >= 300 else reward
    
    def gaussian_func(self, A, x, y, x0, y0, sigx, sigy):
        return A*np.exp(-1*(((x - x0)**2/(2*(sigx**2))) + ((y - y0)**2/(2*(sigy)**2))))
    
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