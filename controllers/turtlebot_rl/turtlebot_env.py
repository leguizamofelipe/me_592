import math
import gym
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use('agg')
import numpy as np
import pandas as pd
import os
from random import randint
import math
import pickle
# from controller import Supervisor
from point import *
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

        # Send thrust [0, 1] and turn command ([0, 1], where 0 is stopped and 1 is full) for each wheel
        self.action_space = spaces.Box(low = np.array([0, 0, 0]), high = np.array([1, 1, 1]), dtype=np.float32)
        self.action_space.n = 2

        # Observe current motor speeds and 180 point LIDAR scan
        self.observation_space = spaces.Box
        # self.observation_space.n = 185#6
        self.observation_space.n = 6
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

        # Read in box positions
        boxes_df = pd.read_csv('boxes.csv')
        
        self.boxes = []
        for line in boxes_df.index:
            self.boxes.append(Point(boxes_df['x'][line], boxes_df['y'][line]))

    def _take_action(self, action):
        thrust = action[0]
        direction = action[2]

        self.T.set_left_motor(thrust * direction)
        self.T.set_right_motor(thrust * direction)
        # print(self.T.get_turtlebot_heading())
        pass
    
    def step(self, action):
        target_tolerance = 0.1
        self.action_count += 1
        # Set the current robot position
        self._take_action(action)

        lidar_vals = self.T.get_lidar_vals()

        reward = 0

        self.error = self.T.get_angle_error_1(self.target)

        # gyro_angle = gyro_angle % 360
        # print(self.T.get_turtlebot_angle()*180/math.pi)
        # print(self.T.get_angle_error(self.target))
        # reward = sum(-1/np.exp(10*lidar_vals-5) + 100)
        # Crashed into something, punish and end ep

        done = False
        # print(lidar_vals.max())
        gamma = 0
        if lidar_vals.max() == 999:
            gamma = -100000
            done = True
        #    print('Hit something!')
            
        # reward += (-math.exp(self.distance_from_target()/0.5) + -3*self.error**2)
        # reward += (-math.exp(self.distance_from_target()/0.5) + -3*self.error**2 + 1000/self.distance_from_target)
        #reward += -3*self.error**2
        # reward += (1000/self.distance_from_target() + -3*self.error**2)
        alpha = -math.exp(self.distance_from_target()/0.5)
        beta = 0#-750*self.error**2 
        reward += alpha + beta + gamma
        
        obs = self._next_observation()
        
        if self.distance_from_target() < target_tolerance:
            reward = 1000000
            done = True
            self.T.set_right_motor(0)
            self.T.set_left_motor(0)

        self.hist.record_position(self.T.get_position())
        self.hist.record_reward(reward)
        self.hist.record_proportions(alpha, beta, gamma, reward)

        # measure_interval = 3
        # if self.action_count %  

        # print(f'Alpha: {alpha/reward}')
        # print(f'Beta: {beta/reward}')
        # print(f'Gamma: {gamma/reward}')
        # print(f'Reward: {reward}')
        if self.action_count > 10000:
            # Done with episode
            done = True

        return obs, reward, done, {}

    def _next_observation(self):
        motors = np.array([self.T.get_left_motor(), self.T.get_right_motor()])
        # lidar_obs = self.T.get_lidar_vals()
        position = self.T.get_position()
        distance_from_target = self.distance_from_target()
        angular_error = self.T.get_angle_error_1(self.target)

        #return np.concatenate((motors, lidar_obs, np.array([position.x, position.y, angular_error, distance_from_target])))
        # return np.concatenate((motors, lidar_obs, np.array([position.x, position.y, distance_from_target])))
        # return np.concatenate((motors, np.array([position.x, position.y, distance_from_target])))
        return np.concatenate((motors, np.array([position.x, position.y, angular_error, distance_from_target])))

    def reset(self):
        self.T.set_right_motor(0)
        self.T.set_left_motor(0)

        # Reset the state of the environment to an initial state
        self.action_count = 0

        if self.hist is not None:
            self.hist_list.append(self.hist)
            # print(f'Ep Reward: {self.hist.total_reward()}')
            # Save every 2 eps
            save_interval = 50
            if self.ep_count % save_interval == 0:
                save_dir = os.path.join(self.save_dir, str(int(time.time())))
                os.makedirs(save_dir)
                # pickle.dump(self, open(os.path.join(self.save_dir, "env_autosave.p"), "wb" ))
                target_vec = np.array([self.target.x-self.T.get_position().x, self.target.y-self.T.get_position().y], dtype=float)
                self.hist.save_episode(save_dir, self.boxes, addendum=f'\nErr: {self.T.get_angle_error_1(self.target)} \nT angle: {self.T.get_turtlebot_heading()} \nTarg angle: {target_vec}')
                fig, ax = plt.subplots()
                ax.plot([ep.total_reward() for ep in self.hist_list])
                plt.title('Total Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.savefig(os.path.join(self.save_dir, '_reward.png'))
                plt.close()

        # self.simulationResetPhysics()
        self.T.robot.simulationReset()
        self.T.initialize_devices()
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
