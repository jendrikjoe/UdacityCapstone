import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import numpy as np


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
	def __init__(self, vehicle_mass, fuel_capacity, brake_deadband,
                 decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio,
                 speed_kp, accel_kp, accel_ki, max_lat_accel,
                 max_steer_angle):
		
		self.yawController = YawController(wheel_base, steer_ratio,
 								1., max_lat_accel, max_steer_angle)
		self.pid = PID(accel_kp, accel_ki, speed_kp,-1,1)
		self.lowPass = LowPassFilter(.1,.1)#Doesn't work anymore....(self.yawController)
		self.intValue = 0
		self.lastError = 0
		self.lastErr = 0
		self.lastTime = 0
				
	def control(self, linearVelocityCmd, angularVelocityCmd, currentVelocity):
		if(self.lastTime == 0):
			self.lastTime = rospy.Time.now()
			return 0,0,0
		else:
			#accCmd = self.calculateValue(linearVelocityCmd, currentVelocity)
			accFilter = self.lowPass.filt(linearVelocityCmd)
			step = (rospy.Time.now() - self.lastTime).to_sec()
			self.lastTime = rospy.Time.now()
			throttle = self.pid.step(linearVelocityCmd-currentVelocity, step)
			#if(linearVelocityCmd == 0 and currentVelocity < 0.9): throttle = -.2
			brake = -throttle if throttle < 0 else 0
			throttle = 0 if throttle < 0 else throttle
			steering = self.yawController.get_steering(linearVelocityCmd, 
						angularVelocityCmd, currentVelocity)
			return throttle, brake, steering
