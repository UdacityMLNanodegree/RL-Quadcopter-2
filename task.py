import numpy as np
from physics_sim import PhysicsSim
import math


class MyTask(): 
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
        self.crash = False

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    # Original rewards function by Udacity
#     def get_reward(self):
#         """Uses current pose of sim to return reward."""
#         reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
#         return reward
    
    
    def __euclideanDistance(self, p, q):
        '''Calculates euclidean distance between two points in nth dimension'''
        n = len(p)
        sums = 0
        for i in range(n):
            sums += (p[i] - q[i]) ** 2
        dist = math.sqrt(sums)
        return dist

    def get_reward(self):
        """My own reward."""
        
#         current_position = self.sim.pose[:3]
#         print(type(current_position), current_position)
#         print(type(self.target_pos), self.target_pos)
        self.distanceFromTarget = self.__euclideanDistance(self.sim.pose[:3], self.target_pos)

#         reward = 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum() # Udacity
#         reward = np.tanh(1. - .3 * (abs(self.sim.pose[:3] - self.target_pos))).sum()
#         reward = np.tanh(1. - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
    
        reward_close = 0
        if self.distanceFromTarget < 2:
            reward_close = 50
        else:
            reward_close = -10
        
#         #detect crash
        reward_crash = 0
        if ((self.sim.pose[2] == 0) and (abs(self.sim.v[2]) > 2.)):
            self.crash = True
            self.counter = 0
            reward_crash = -1000
            
        reward = np.tanh(1. - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum() + np.tanh(reward_close) + np.tanh(reward_crash)
                  
        return reward

    
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
        self.crash = False
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state