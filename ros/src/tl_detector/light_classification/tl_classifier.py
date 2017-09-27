from styx_msgs.msg import TrafficLight
import numpy
import sys
import PIL
import message_filters
import rospy

from PIL import Image
from cv_bridge import CvBridge
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from road_wizard.msg import Signals
from runtime_manager.msg import traffic_light
from sensor_msgs.msg import Image

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
		self.loaded_model = self.get_model()
        

	def get_model(self):
	global model
	if not model:
		model = load_model('light_classifier_model.h5')
	return model

	def predict_light(self, image_array):
		# Load CNN Model
		prediction = self.loaded_model.predict(image_array[None, :])
		if prediction[0][0] == 1:
			return TrafficLight.GREEN
		elif prediction[0][1] == 1:
			return TrafficLight.RED
		else:
			return TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
		# UGH checkout https://github.com/lexfridman/deepcars/blob/master/5_tensorflow_traffic_light_classification.ipynb 
		# dewwww ittttt

		image_array = img_to_array(image.resize((64, 64), PIL.Image.ANTIALIAS))
        return self.predict_light(image_array)
