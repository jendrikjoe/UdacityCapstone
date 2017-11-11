#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math
import numpy as np
from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        self.twiddleController = rospy.get_param('~twiddle', False)
        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)
        self.currentVelocity = 0
        self.currentAngularVelocity = 0
        self.cmdVelocity = 0
        self.cmdAngularVelocity = 0
        self.isDBMEnabled = True
        self.controller = None

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle

        
        if self.twiddleController:
            self.twiddleState = -1
            self.twiddleScale = .2
            self.meanThrottle = .5
            self.error = 1e9
            self.twiddleStorage = [0.025, 0.234, 0.080]
            self.twiddleParams = [ 0.025, 0.234, 0.080]
            self.twiddleMax = .3
            self.twiddle(0, 0)
        else:
            self.controller = Controller(vehicle_mass=self.vehicle_mass, 
                fuel_capacity=self.fuel_capacity, brake_deadband=self.brake_deadband,
                 decel_limit=self.decel_limit, accel_limit=self.accel_limit, 
                 wheel_radius=self.wheel_radius, wheel_base=self.wheel_base, 
                 steer_ratio=self.steer_ratio,
                 speed_kp=self.twiddleParams[0], accel_kp=self.twiddleParams[1], 
                 accel_ki=self.twiddleParams[2], max_lat_accel=self.max_lat_accel, 
                 max_steer_angle=self.max_steer_angle)

        rospy.Subscriber('/current_velocity', TwistStamped, self.velCallback)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twistCmdCallback)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbwEnabledCallback)

        self.loop()

    def twiddle(self, vel, cmdVel):
        if self.controller != None:
            if rospy.get_time() - self.startTime <600:
                self.currentErr += abs(vel - cmdVel)
                return
            else:
                if self.error > self.currentErr:
                    self.error = self.currentErr
                    self.twiddleStorage[:] = self.twiddleParams[:]
                    rospy.logerr("New params: %.3f, %.3f, %.3f. With error: %.3f",
                          self.twiddleParams[0], self.twiddleParams[1], 
                          self.twiddleParams[2], self.currentErr )
                    self.twiddleScale *= 1.1
                else:
                    rospy.logerr("No new params: %.3f, %.3f, %.3f. With error: %.3f",
                          self.twiddleParams[0], self.twiddleParams[1], 
                          self.twiddleParams[2], self.currentErr)
                self.twiddleParams[:] = self.twiddleStorage[:]
                self.twiddleState += 1
                if self.twiddleState > 5:
                    self.twiddleState = 0
                    self.twiddleScale *= .9
                if self.twiddleState == 0: 
                    self.twiddleParams[1] = (1+self.twiddleScale)*self.twiddleParams[1]
                elif self.twiddleState == 1: 
                    self.twiddleParams[1] = (1-self.twiddleScale)*self.twiddleParams[1]
                elif self.twiddleState == 2: 
                    self.twiddleParams[2] = (1+self.twiddleScale)*self.twiddleParams[2]
                elif self.twiddleState == 3: 
                    self.twiddleParams[2] = (1-self.twiddleScale)*self.twiddleParams[2]
                elif self.twiddleState == 4: 
                    self.twiddleParams[0] = (1+self.twiddleScale)*self.twiddleParams[0]
                elif self.twiddleState == 5: 
                    self.twiddleParams[0] = (1-self.twiddleScale)*self.twiddleParams[0]
        self.controller = Controller(vehicle_mass=self.vehicle_mass, 
                fuel_capacity=self.fuel_capacity, brake_deadband=self.brake_deadband,
                 decel_limit=self.decel_limit, accel_limit=self.accel_limit, 
                 wheel_radius=self.wheel_radius, wheel_base=self.wheel_base, 
                 steer_ratio=self.steer_ratio,
                 speed_kp=self.twiddleParams[0], accel_kp=self.twiddleParams[1], 
                 accel_ki=self.twiddleParams[2], max_lat_accel=self.max_lat_accel, 
                 max_steer_angle=self.max_steer_angle)
        self.startTime = rospy.get_time()
        self.currentErr = 0
    
    def dbwEnabledCallback(self, data):
        try:
            self.isDBMEnabled = data.dbw_status
        except: pass

    def velCallback(self, data):
        #rospy.loginfo('Got velocity data. ' + data.__str__())
        self.currentVelocity = data.twist.linear.x
        self.currentAngularVelocity = .5*self.currentAngularVelocity + .5*data.twist.angular.z
        if self.twiddleController:
            self.twiddle(self.currentVelocity, self.cmdVelocity)


    def twistCmdCallback(self, data):
        #rospy.loginfo('Got a twist Cmd. ' + data.__str__())
        self.cmdVelocity = data.twist.linear.x
        self.cmdAngularVelocity = data.twist.angular.z

    def loop(self):
        rate = rospy.Rate(50) # 20Hz
        while not rospy.is_shutdown():
            
            throttle, brake, steer = self.controller.control(
				self.cmdVelocity, 
				self.cmdAngularVelocity, 
				self.currentVelocity
			)
            #rospy.logerr('Control: %.3f is %.3f throttle %.3f brake %.3f', 
            #             self.cmdVelocity, self.currentVelocity, throttle, brake)
            if self.twiddleController:
                self.currentErr += 2*np.abs(self.meanThrottle-throttle)
                self.meanThrottle = 0.999*self.meanThrottle + 0.001*throttle
                self.currentErr += brake
            #rospy.logerr('Commanding. Throttle:%.3f Brake:%.3f Steer:%.3f' % (throttle, brake, steer))
            if self.isDBMEnabled:
                self.publish(throttle, brake, steer)
            else:
                # TODO: Come up with a better strategy. For now come to a complete hauld
                self.publish(0.0, 0.0, 0.0)
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            # throttle, brake, steering = self.controller.control(<proposed linear velocity>,
            #                                                     <proposed angular velocity>,
            #                                                     <current linear velocity>,
            #                                                     <dbw status>,
            #                                                     <any other argument you need>)
            # if <dbw is enabled>:
            #   self.publish(throttle, brake, steer)
            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake*1000
        self.brake_pub.publish(bcmd)




if __name__ == '__main__':
    DBWNode()
