"""turtlebot_rl controller."""

from controller import Robot, GPS
from turtlebot import Turtlebot
from turtlebot_env import TurtlebotEnv
from point import Point
from stable_baselines3 import PPO

target = Point(0.889, -0.958)
time_steps = 1000000

T = Turtlebot()

env = TurtlebotEnv(T, target)

model = PPO('MlpPolicy', env, verbose = 1, device = 'cuda')

model.learn(total_timesteps=int(time_steps), n_eval_episodes = 100000)

