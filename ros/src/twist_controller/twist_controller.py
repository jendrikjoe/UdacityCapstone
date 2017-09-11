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
		self.intValue = 0
		self.lastError = 0
		self.lastErr = 0
		self.speed_kp = speed_kp
		self.accel_kp = accel_kp
		self.accel_ki = accel_ki
	

	def calculateValue(self, cmd, value):
		err = cmd-value
		self.intValue += err
		der = err - self.lastErr
		self.lastError = err
		return self.speed_kp*err + self.accel_kp*der + self.accel_ki*self.intValue
				
	def control(self, linearVelocityCmd, angularVelocityCmd, currentVelocity):
		#accCmd = self.calculateValue(linearVelocityCmd, currentVelocity)
		throttle = self.calculateValue(linearVelocityCmd,currentVelocity)
		brake = 0#-1.*accCmd if accCmd < 0 else 0
		steering = self.yawController.get_steering(linearVelocityCmd, 
					angularVelocityCmd, currentVelocity)
		return throttle, brake, steering
