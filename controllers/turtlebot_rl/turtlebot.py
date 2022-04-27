from cmath import inf
from controller import Robot, GPS, Supervisor
import numpy as np
from point import Point

class Turtlebot():
    def __init__(self):
        ####################### WEBOTS ENV SETUP #####################################
        # self.robot = Supervisor()
        self.robot = Supervisor()
        self.initialize_devices()


    def initialize_devices(self):
        self.TIME_STEP = 64
        self.MAX_SPEED = 6.67

        self.robot.step(self.TIME_STEP)

        # get a handler to the motors and set target position to infinity (speed control)
        self.lidar = self.robot.getDevice('LDS-01')
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.right_motor.setVelocity(0)
        self.left_motor.setVelocity(0)

        self.GPS = GPS('gps')

        self.GPS.enable(self.TIME_STEP)
        
        # lidar_width = self.lidar.getHorizontalResolution()
        # lidar_max_range = self.lidar.getMaxRange()
        self.lidar.enable(self.TIME_STEP)
        self.lidar.enablePointCloud()
        # lidar_values = self.lidar.getRangeImage()

    def get_lidar_vals(self):
        # For now, only care about front 180 vals
        # Set inf to 999 to induce high punishment for collisions
        while(self.robot.step(self.TIME_STEP) != -1):
            vals = np.array(self.lidar.getRangeImage())
            break
        vals[vals==np.inf] = 999
        # print(vals)
        vals = vals[90:270]
        return vals

    def set_right_motor(self, val):
        self.right_motor.setVelocity(abs(val * self.MAX_SPEED))

    def set_left_motor(self, val):
        self.left_motor.setVelocity(abs(val * self.MAX_SPEED))

    def get_right_motor(self):
        return self.right_motor.getVelocity() / self.MAX_SPEED

    def get_left_motor(self):
        return self.left_motor.getVelocity() / self.MAX_SPEED

    def get_position(self):
        pose = self.GPS.getValues()
        return Point(pose[0], pose[1])

    