"""turtlebot_rl controller."""

from controller import Robot, GPS, Display
from turtlebot import Turtlebot
from turtlebot_env import TurtlebotEnv
from point import Point
from stable_baselines3 import PPO

target = Point(0.5, 0.5)
time_steps = 1000000

T = Turtlebot()

env = TurtlebotEnv(T, target)

# while(True):
#     print(T.get_turtlebot_heading())

model = PPO('MlpPolicy', env, verbose = 1, device = 'cpu')

model.learn(total_timesteps=int(time_steps), n_eval_episodes = 100000)

