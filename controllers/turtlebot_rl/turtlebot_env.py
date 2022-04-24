import gym
import numpy as np
import pandas as pd
import os
from random import randint
import math
import pickle

import matplotlib.pyplot as plt
import time

from gym import spaces
from sympy import false
from episode_history import EpisodeHistory

class TurtlebotEnv(gym.Env):
    def __init__(self, turtlebot, target, save_dir = 'logs'):
        super(TurtlebotEnv, self).__init__()

        # Store robot instance in memory
        self.T = turtlebot

        ####################################################################################
        ################################## ACTION SPACE ####################################
        ####################################################################################

        # Send a velocity to left [0] and right [1] motor
        self.action_space = spaces.Box(low = np.array([-1, -1]), high = np.array([1, 1]), dtype=np.float32)
        self.action_space.n = 2

        # Observe current motor speeds and 180 point LIDAR scan
        self.observation_space = spaces.Box
        self.observation_space.n = 182
        self.observation_space.low = np.ones(self.observation_space.n) * -1
        self.observation_space.high = np.ones(self.observation_space.n)

        # Observations: position of joints, mapped to the full range of a joint. Last three vals in array are position
        self.observation_space = spaces.Box(low = self.observation_space.low, high=self.observation_space.high, dtype=np.float32) 

        # self.action_count = 0
        self.hist = None
        self.hist_list = []
        self.ep_count = 0

        # Set target pose
        self.target = target
        self.save_dir = os.path.join(save_dir, str(int(time.time())))
        
        print(f'Logging at {self.save_dir}')
        os.makedirs(self.save_dir)

        self.init_pos = self.T.get_position()

    def _take_action(self, action):
        self.T.set_left_motor(action[0])
        self.T.set_right_motor(action[1])
    
    def step(self, action):
        target_tolerance = 0.1
        self.action_count += 1
        # Set the current robot position
        self._take_action(action)

        lidar_vals = self.T.get_lidar_vals()
        
        # reward = sum(-1/np.exp(10*lidar_vals-5) + 100)
        # Crashed into something, punish and end ep
        if lidar_vals.max() == 999:
            reward = -100000
            done = True
            print('Hit something!')
            
        reward = -(self.distance_from_target()**2)
        
        obs = self._next_observation()

        done = False
        
        if self.distance_from_target() < target_tolerance:
            reward = 10000
            done = True

        self.hist.record_position(self.T.get_position())
        self.hist.record_reward(reward)

        done = False
        # print(reward)
        if self.action_count > 100:
            # Done with episode
            done = True

        return obs, reward, done, {}

    def _next_observation(self):
        motors = np.array([self.T.get_left_motor(), self.T.get_right_motor()])
        lidar_obs = self.T.get_lidar_vals()

        return np.concatenate((motors, lidar_obs))

    def reset(self):
        self.T.set_right_motor(0)
        self.T.set_left_motor(0)

        # Reset the state of the environment to an initial state
        self.action_count = 0

        if self.hist is not None:
            self.hist_list.append(self.hist)
            print(f'Ep Reward: {self.hist.total_reward()}')
            # Save every 2 eps
            save_interval = 2
            if self.ep_count % save_interval == 0:
                # pickle.dump(self, open(os.path.join(self.save_dir, "env_autosave.p"), "wb" ))
                self.hist.save_episode(self.save_dir)


        self.ep_count +=1

        self.hist = EpisodeHistory(0, self.init_pos, self.target)      

        return self._next_observation()

    def distance_from_target(self):
        # Current position
        c = self.T.get_position()

        # Target position
        t = self.target

        distance = math.sqrt((c.x-t.x)**2 + (c.y-t.y)**2)
        
        return distance
