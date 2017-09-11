#!/usr/bin/python
from geometry_msgs.msg import PoseStamped, Quaternion
from styx_msgs.msg import Lane, Waypoint
import rospy
from matplotlib import pyplot as plt
import math
import tf
import numpy as np

class TrajectoryPlotter(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        self.baseWaypoints = None

        rospy.Subscriber('/current_pose', PoseStamped, self.position_cb)
        rospy.Subscriber('/final_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.position = [0,0,0]
        self.yaw = 0
        self.speed = 0
        self.targetLane = 1
        self.currentWPIndex = -1
        fig = plt.figure()
        self.ax = fig.gca()
        self.ax.set_title('Trajectory')
        self.ax.set_xlabel('x')
        self.ax.set_xlabel('y')
        plt.show(block=True)
        #rospy.spin()
        

    def position_cb(self, msg):
        self.position = [msg.pose.position.x,
            msg.pose.position.y, msg.pose.position.z]
        orientation=(msg.pose.orientation.x, msg.pose.orientation.y,
                        msg.pose.orientation.z, msg.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(orientation)
        self.yaw = euler[2]
        #rospy.logerr('yaw:%.3f' % self.yaw)
        self.yaw = self.yaw if self.yaw < np.pi else self.yaw - 2*np.pi

    @staticmethod
    def quaternion_from_yaw(yaw):
        return tf.transformations.quaternion_from_euler(0., 0., yaw)

    def waypoints_cb(self, lane):
        self.finalWaypoints = []
        localX, localY = self.getXY(lane.waypoints)
        self.ax.cla()
        self.ax.plot(localX, localY)
        self.ax.plot(self.position[0], self.position[1], marker = 'o', ms =5, color='r')
        plt.draw()
        
    def getXY(self, waypoints):
        xs = []
        ys = []
        for waypoint in waypoints:
            x = self.getX(waypoint)
            y = self.getY(waypoint)
            xs.append(x)
            ys.append(y)
        return xs, ys

    def getLocalXY(self, waypoints):
        localXs = []
        localYs = []
        for waypoint in waypoints:
            x = self.getX(waypoint)
            y = self.getY(waypoint)
            x = x - self.position[0]
            y = y - self.position[1]
            localX = x*math.cos(self.yaw) + y * math.sin(self.yaw) 
            localY = -x*math.sin(self.yaw) + y * math.cos(self.yaw) 
            localXs.append(localX)
            localYs.append(localY)
        return localXs, localYs
        


    @staticmethod
    def getX(waypoint):
        return waypoint.pose.pose.position.x

    @staticmethod
    def getYaw(waypoint):
        orientation=(waypoint.pose.pose.orientation.x, waypoint.pose.pose.orientation.y,
                        waypoint.pose.pose.orientation.z, waypoint.pose.pose.orientation.w)
        euler =  tf.transformations.euler_from_quaternion(orientation)
        return euler[2]

    @staticmethod
    def getY(waypoint):
        return waypoint.pose.pose.position.y

    @staticmethod
    def getVelocity(waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    """def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist"""
    
            


if __name__ == '__main__':
    try:
        TrajectoryPlotter()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
