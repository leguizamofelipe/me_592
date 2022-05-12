from cmath import inf
import math
from controller import Robot, GPS, Supervisor
import numpy as np
import time
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
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.TIME_STEP)
        self.gyro = self.robot.getDevice('gyro')
        self.gyro.enable(self.TIME_STEP)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.right_motor.setVelocity(0)
        self.left_motor.setVelocity(0)

        self.theta_dot = 0
        self.prev_time = self.robot.getTime()
        self.gyro_angle = 0

        self.GPS = GPS('gps')

        self.GPS.enable(self.TIME_STEP)
        
        # lidar_width = self.lidar.getHorizontalResolution()
        # lidar_max_range = self.lidar.getMaxRange()
        self.lidar.enable(self.TIME_STEP)
        self.lidar.enablePointCloud()
        
        # lidar_values = self.lidar.getRangeImage()
        # print(self.get_turtlebot_angle())

    def get_lidar_vals(self):
        # For now, only care about front 180 vals
        # Set inf to 999 to induce high punishment for collisions
        while(self.robot.step(self.TIME_STEP) != -1):
            vals = np.array(self.lidar.getRangeImage())
            break
        vals[vals==np.inf] = 0
        # print(vals)
        vals = vals[90:270]
        # print(vals)
        return vals

    def set_right_motor(self, val):
        self.right_motor.setVelocity(val * self.MAX_SPEED)

    def set_left_motor(self, val):
        self.left_motor.setVelocity(val * self.MAX_SPEED)

    def get_right_motor(self):
        return self.right_motor.getVelocity() / self.MAX_SPEED

    def get_left_motor(self):
        return self.left_motor.getVelocity() / self.MAX_SPEED

    def get_position(self):
        pose = self.GPS.getValues()
        return Point(pose[0], pose[1])

    def get_turtlebot_angle(self):
        compass_val = self.compass.getValues()[1]
        angle = compass_val * math.pi/2
        if np.sign(self.gyro_angle) == -1:
            return angle
        else:
            return math.pi - angle

    def get_angular_vel(self):
        return(self.gyro.getValues()[2])

    def update_gyro_angle(self):
        self.gyro_angle += self.get_angular_vel() * (self.robot.getTime()-self.prev_time)
        self.prev_time = self.robot.getTime()
        return self.gyro_angle
    
    '''
    def get_angle_error(self, target):
        dx= target.x - self.get_position().x
        dy= target.y - self.get_position().y
        angle_target =  math.pi - math.atan2(dy, dx)
        val = angle_target-self.get_turtlebot_angle()
        # print(f'Angle to target: {angle_target}')
        # print(f'Turtlebot angle: {self.get_turtlebot_angle()}')
        # print(f'Error: {val}')
        return val
    '''
    
    def get_turtlebot_heading(self):
        #print(math.atan2(self.compass.getValues()[0], self.compass.getValues()[1])*180/math.pi)    
        # print(self.compass.getValues())
        return math.atan2(self.compass.getValues()[0], self.compass.getValues()[1])

    def get_angle_error_1(self, target):
        '''
        heading = self.get_turtlebot_heading()
        heading_vector = np.array([math.sin(heading), math.cos(heading)], dtype=float)
        goal_vector = np.array([target.x-self.get_position().x, target.y-self.get_position().y], dtype=float)
        val = math.acos((np.inner(heading_vector, goal_vector))/(np.sqrt(goal_vector.dot(goal_vector))))
        '''
        heading = self.get_turtlebot_heading()
        goal_vector = np.array([target.x-self.get_position().x, target.y-self.get_position().y], dtype=float)
        goal_direction = math.atan2(goal_vector[1], goal_vector[0])
        val = goal_direction-heading
        return val