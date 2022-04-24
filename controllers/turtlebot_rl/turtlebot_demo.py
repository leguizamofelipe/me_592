from controller import Robot, Motor, Lidar

robot = Robot()

TIME_STEP = 64
MAX_SPEED = 6.67

# get a handler to the motors and set target position to infinity (speed control)
lidar = robot.getDevice('LDS-01')
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# lidar_width = self.lidar.getHorizontalResolution()
# lidar_max_range = self.lidar.getMaxRange()
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

while robot.step(TIME_STEP) != -1:
    print(lidar.getPointCloud()[1])
    pass
