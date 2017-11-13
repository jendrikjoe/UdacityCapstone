import rospy
from styx_msgs.msg import TrafficLight
from PIL import Image
from cv_bridge import CvBridge

class TLClassifier(object):
	def __init__(self):
        #TODO load classifier
		PATH_TO_MODEL = '../../../resnet_rcnn/fine_tuned_model/frozen_inference_graph.pb'
		PATH_TO_LABELS = '../../../resnet_rcnn/data/bosch_label_map.pbtxt'
		IMAGE_TENSOR = 'image_tensor:0'
		BOXES_TENSOR = 'detection_boxes:0'
		SCORES_TENSOR = 'detection_scores:0'
		CLASSES_TENSOR = 'detection_classes:0'
		NUM_DETECTIONS_TENSOR = 'num_detections:0'
		NUM_CLASSES = 14

		TRAFFIC_LIGHT_THRESHOLD = 0.7

		self.category_index = {1: {'id': 1, 'name': 'Green'}, 2: {'id': 2, 'name': 'Red'}, 3: {'id': 3, 'name': 'GreenLeft'}, 4: {'id': 4, 'name': 'GreenRight'}, 5: {'id': 5, 'name': 'RedLeft'}, 6: {'id': 6, 'name': 'RedRight'}, 7: {'id': 7, 'name': 'Yellow'}, 8: {'id': 8, 'name': 'off'}, 9: {'id': 9, 'name': 'RedStraight'}, 10: {'id': 10, 'name': 'GreenStraight'}, 11: {'id': 11, 'name': 'GreenStraightLeft'}, 12: {'id': 12, 'name': 'GreenStraightRight'}, 13: {'id': 13, 'name': 'RedStraightLeft'}, 14: {'id': 14, 'name': 'RedStraightRight'}}
		# Initialize to we really don't know man!
		self.current_traffic_light = TrafficLight.UNKNOWN

        
	def get_model(self):
		"""
		global model
		if not model:
			model = load_model('light_classifier_model.h5')
		return model
		"""
		# Thanks for tensorflow and for Daniel Stang's tutorial for this part and training with the Object API
		# https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-4-training-the-model-68a9e5d5a333

		self.model = tf.Graph()
		with self.model.as_default():
			graph_def = tf.GraphDef()
			# Like in the notebook
			with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
				serialized_graph = fid.read()
				graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(graph_def, name='')

			self.image_tensor = self.model.get_tensor_by_name(IMAGE_TENSOR)
			self.d_boxes = self.model.get_tensor_by_name(BOXES_TENSOR)
			self.d_scores = self.model.get_tensor_by_name(SCORES_TENSOR)
			self.d_classes = self.model.get_tensor_by_name(CLASSES_TENSOR)
			self.d_num_detections = self.model.get_tensor_by_name(NUM_DETECTIONS_TENSOR)
		self.sess = tf.Session(graph=self.model)
		
	def predict_light(self, image):
		# Load CNN Model
		"""
		prediction = {}  #self.loaded_model.predict(image_array[None, :])
		if prediction[0][0] == 1:
			return TrafficLight.GREEN
		elif prediction[0][1] == 1:
			return TrafficLight.RED
		else:
		"""
		self.get_model()
		return self.get_classification(image)

	def get_classification(self, image):
		"""Determines the color of the traffic light in the image
		Args:
			image (cv::Mat): image containing the traffic light
		Returns:
			int: ID of traffic light color (specified in styx_msgs/TrafficLight)
		"""

		#TODO implement light color prediction
		# checkout https://github.com/lexfridman/deepcars/blob/master/5_tensorflow_traffic_light_classification.ipynb 
		# Similar to the work done on the notebook
		
		with self.model.as_default():
			numpy_image = np.expand_dims(image, axis=0)
			(boxes, scores, classes, num_d) = self.sess.run(
				[self.d_boxes, self.d_scores, self.d_classes, self.d_num_detections],
				feed_dict= {self.image_tensor: numpy_image}
			)
		
		boxes = np.squeeze(boxes)
		scores = np.squeeze(scores)
		classes = np.squeeze(classes).astype(np.int32)

		for i in range(boxes.shape[0]):
			if scores is not None or scores[i] > TRAFFIC_LIGHT_THRESHOLD:
				traffic_class = self.category_index[classes[i]['name']]
				if 'Red' in traffic_class:
					self.current_traffic_light = TrafficLight.RED
				elif 'Green' in traffic_class:
					self.current_traffic_light = TrafficLight.GREEN
				elif 'Yellow' in traffic_class:
					self.current_traffic_light = TrafficLight.YELLOW
		
				# Thanks for Anthony Sarkis's and Vatsal Srivastava's blog posts for helping with this part :)
				# https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62
				# https://medium.com/@anthony_sarkis/self-driving-cars-implementing-real-time-traffic-light-detection-and-classification-in-2017-7d9ae8df1c58

				fx = 2.943
				fy = 4.843

				p_width_x = (boxes[i][3] - boxes[i][1]) * 800
				p_width_y = (boxes[i][2] - boxes[i][0]) * 600

				p_depth_x = ((0.1 * fx) / p_width_x)
				p_depth_y = ((0.3 * fy) / p_width_y)

				e_distance = round((p_depth_x + p_depth_y) / 2.0)
				print("Traffic is at : ", e_distance, " Away!")
				print("Current Traffic Light Detected: ", self.current_traffic_light)
		return self.current_traffic_light
