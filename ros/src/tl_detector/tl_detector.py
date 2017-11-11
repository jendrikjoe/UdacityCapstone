#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import cv2
import tf
import yaml
import numpy as np
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.baseWaypoints = None
        self.camera_image = None
        self.lights = []
        self.lightWPs = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        
        
        

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.annImagePub = rospy.Publisher('/annotated_image', Image, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg



    def waypoints_cb(self, lane):
        if np.any(self.baseWaypoints == None):
            self.baseWaypoints = []
            for waypoint in lane.waypoints:
                self.baseWaypoints.append([
                    self.getX(waypoint),
                    self.getY(waypoint),
                    self.get_waypoint_velocity(waypoint),
                    self.getYaw(waypoint)])
        if len(self.lightWPs) == 0:
            stopLines = self.config['stop_line_positions']
            for line in stopLines:
                stopWp = -1
                smallestDist = 1000 
                for i, wp in zip(xrange(len(self.baseWaypoints)), self.baseWaypoints):
                    stopX = line[0] - wp[0]
                    stopY = line[1] - wp[1]
                    if(smallestDist > np.sqrt(stopX**2+stopY**2)):
                        stopWp = i
                        smallestDist = np.sqrt(stopX**2+stopY**2)
                self.lightWPs.append(stopWp)
                rospy.loginfo("Waypoint: %d", stopWp)
            rospy.loginfo("Len light waypoints: %d", len(self.lightWPs))
                
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

    def traffic_cb(self, msg):
        self.lights = msg.lights
                
            

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state != TrafficLight.GREEN else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closestLen = 100000. #large number
        closestWaypoint = 0
        for i in range(0,len(self.waypoints)):
            x = self.waypoints[i][0]
            y = self.waypoints[i][1]
            dist = self.eucldian_distance(x, y, self.pose.position.x, self.pose.position.y)
            if(dist < closestLen):
                closestLen = dist
                closestWaypoint = i
        return closestWaypoint
    
    @staticmethod
    def eucldian_distance(x1, y1, x2, y2):
        return np.sqrt((x1-x2)**2+(y1-y2)**2)


    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """
        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (transT, rotT) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")
            return None, None
        pt = PointStamped()
        pt.header.stamp = point_in_world.header.stamp
        pt.header.frame_id = "world"
        pt.point.x = point_in_world.pose.pose.position.x
        pt.point.y = point_in_world.pose.pose.position.y
        pt.point.z = point_in_world.pose.pose.position.z
        #rospy.loginfo("Point: " + str(pt))
        target_pt = self.listener.transformPoint("base_link", pt)
        cx = image_width/2
        cy = image_height/2
        
        ##########################################################################################
        # DELETE THIS MAYBE - MANUAL TWEAKS TO GET THE PROJECTION TO COME OUT CORRECTLY IN SIMULATOR
        # just override the simulator parameters. probably need a more reliable way to determine if 
        # using simulator and not real car
        # See discussion in forum
        if fx < 10:
            fx = 2344
            fy = 2552 #303.8
            target_pt.point.y += 0.5
            target_pt.point.z -= 1.2
            cy = cy * 2 
        ##########################################################################################

        
        #rospy.loginfo("Target Point: " + str(target_pt))
        x = -target_pt.point.y * fx / target_pt.point.x; 
        y = -target_pt.point.z * fy / target_pt.point.x; 
        #rospy.loginfo("x: %.3f, y:%.3f"%(x,y))
        x = int(x + cx)
        y = int(y + cy) 

        return (x, y)
    

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        # Are these on the center of the traffic light?
        # Use RQT image
        x, y = self.project_to_image_plane(light)
        
        image = cv_image[:]

        try:
            rospy.loginfo("Position in image: %d, %d"%(x,y))
            cv2.circle(image,(int(x),int(y)),10,(0,0,255),3) # draw center
        except:
            pass
        try:
            image = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.annImagePub.publish(image)
        except CvBridgeError, e:
            print e
   
        

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        if(len(self.lights) == 0): return -1, 4
        if(len(self.lightWPs) == 0): return -1, 4
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            #car_position = self.get_closest_waypoint(self.pose.pose)
            orientation=(self.pose.pose.orientation.x, self.pose.pose.orientation.y,
                        self.pose.pose.orientation.z, self.pose.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(orientation)
            yaw = euler[2]
            light_wp = -1
            smallestDist = 10000.
            for i, stopLine in zip(xrange(len(stop_line_positions)), stop_line_positions):
                shiftStopX = stopLine[0] - self.pose.pose.position.x
                shiftStopY = stopLine[1] - self.pose.pose.position.y
                relStopX = math.cos(yaw)*shiftStopX + math.sin(yaw)*shiftStopY
                relStopY = -math.sin(yaw)*shiftStopX + math.cos(yaw)*shiftStopY
                #rospy.loginfo("RelX: %.3f, relY: %.3f, dist:%.1f"%(relStopX, relStopY, np.sqrt(relStopX**2+relStopY**2)))
                if(relStopX > -2 and relStopX < 100 and smallestDist > np.sqrt(relStopX**2+relStopY**2)):
                    light_wp = i
                    smallestDist = np.sqrt(relStopX**2+relStopY**2)
            
            #rospy.loginfo("Selected light: %i"%light_wp)
            if light_wp != -1:
                x, y = self.project_to_image_plane(self.lights[light_wp])
                if(x > 0 and x < self.config['camera_info']['image_width'] and
                   y > 0 and y < self.config['camera_info']['image_height']) :
                    light = self.lights[light_wp]
                    
        if light:
            state = light.state#self.get_light_state(light)
            return self.lightWPs[light_wp], state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
