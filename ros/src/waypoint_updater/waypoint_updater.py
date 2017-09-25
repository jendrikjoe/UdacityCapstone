#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        self.baseWaypoints = None

        rospy.Subscriber('/current_pose', PoseStamped, self.position_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.position = [0,0,0]
        self.yaw = 0
        self.speed = 0
        self.targetLane = 1
        self.currentWPIndex = -1

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        self.loop()



    @staticmethod
    def quaternion_from_yaw(yaw):
        return tf.transformations.quaternion_from_euler(0., 0., yaw)

    def waypoints_cb(self, lane):
        if np.any(self.baseWaypoints == None):
            self.baseWaypoints = []
            for waypoint in lane.waypoints:
                self.baseWaypoints.append([
                    self.getX(waypoint),
                    self.getY(waypoint),
                    self.get_waypoint_velocity(waypoint),
                    self.getYaw(waypoint)])


    def getNextWpIndex(self):
        closestWpIndex = self.closestWaypointIndex()
        x = self.baseWaypoints[closestWpIndex][0]
        y = self.baseWaypoints[closestWpIndex][1]
        heading = math.atan2( (y-self.position[1]),(x-self.position[0]))
        angle = abs(self.yaw-heading)
        if(angle > np.pi/4): closestWpIndex+=1
        return closestWpIndex

    def closestWaypointIndex(self):

        closestLen = 100000. #large number
        closestWaypoint = 0
        for i in range(self.currentWPIndex,len(self.baseWaypoints)):
            x = self.baseWaypoints[i][0]
            y = self.baseWaypoints[i][1]
            dist = WaypointUpdater.eucldian_distance(self.position[0], self.position[1], x, y)
            if(dist < closestLen):
                closestLen = dist
                closestWaypoint = i
        return closestWaypoint


    @staticmethod
    def eucldian_distance(x1, y1, x2, y2):
        return np.sqrt((x1-x2)**2+(y1-y2)**2)

    @staticmethod
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def position_cb(self, msg):
        self.position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        orientation=(msg.pose.orientation.x, msg.pose.orientation.y,
                        msg.pose.orientation.z, msg.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(orientation)
        self.yaw = euler[2]
        # rospy.logerr('yaw:%.3f' % self.yaw)
        self.yaw = self.yaw if self.yaw < np.pi else self.yaw - 2*np.pi

    def pose_cb(self, msg):
        # TODO: Implement
        pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

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
    def get_waypoint_velocity(waypoint):
        return waypoint.twist.twist.linear.x

    @staticmethod
    def set_waypoint_velocity(waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    """def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist"""
    
    def loop(self):
        rate = rospy.Rate(10) # 5Hz
        while not rospy.is_shutdown():
            if np.any(self.baseWaypoints == None): rate.sleep()
            else:
                self.currentWPIndex = self.getNextWpIndex()
                usedWps = []
                for i in np.arange(self.currentWPIndex, self.currentWPIndex+LOOKAHEAD_WPS):
                    if(i < len(self.baseWaypoints)-1): 
                        usedWps.append(self.baseWaypoints[i])
                        #rospy.logerr('heading:%.3f \nwpX:%.3f wpY:%.3f' %
                        #         (self.baseWaypoints[i][3], self.baseWaypoints[i][0], self.baseWaypoints[i][1]))
                lane = Lane()
                msgWps = []
                for waypoint in usedWps:
                    #if(len(msgWps) == 0):
                    #    rospy.logerr('heading:%.3f \nwpX:%.3f wpY:%.3f' %
                    #             (waypoint[3], waypoint[0], waypoint[1]))
                    msgWp = Waypoint()
                    msgWp.pose.pose.position.x = waypoint[0]
                    msgWp.pose.pose.position.y = waypoint[1] 
                    q = self.quaternion_from_yaw(waypoint[3])
                    msgWp.pose.pose.orientation = Quaternion(*q)
                    msgWp.twist.twist.linear.x = 10.
                    msgWps.append(msgWp)
                    
                lane.header.frame_id = '/world'
                lane.header.stamp = rospy.Time(0)
                lane.waypoints = msgWps
                self.final_waypoints_pub.publish(lane)
                rate.sleep()


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
