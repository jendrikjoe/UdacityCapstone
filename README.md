
# Team Beta-Testers 
![Beta Testers](https://d3o1wlpkmt4nt9.cloudfront.net/wp-content/uploads/2016/04/20182258/find-beta-testers.jpg)
![Image credit to Startups.co since we got it from a google image search referencing their blog]

Jendrik Joerdening : jendrik.joerdening@outlook.de
Rana Khalil: rana.ae.khalil@gmail.com

## Waypoint Updater

The waypoint updater provides the next 200 waypoints for the car from its current position. Hereby, it is decreasing the waypoint speed down to zero if a red light is coming up, such that the car stops at the stopping line until the light turns green. The target speed is hereby changed from the waypoints speed down to 1.5m/s from 50 waypoints prior to the traffic light until the traffic light. Then the speed is kept at 1.5 m/s until the waypoint prior to the stopping line.


## Controller

### Throttle Controller

Throttle and brake commands are determined by a PID controller what values were determined using the twiddle algorithm. The error was hereby the sum of the error in velocity and the distance of the commanded values from the mean commanded value. It was then twiddled two times. A pre-twiddling with just a constant speed was used to determine the right region of controller values. For this every parameter combination was tested for 20s. Then the determined values were taken and the car in the simulator had to obey the traffic lights. Now each parameter set was tested for 10 mins. This procedure let to the following PID values: [0.025, 0.234, 0.080].

### Steering Controller

The provided model based steering controller was used as is.

## Transfer learning and deep learning

### Dataset used
We decided to use the bosch dataset [bosch data set](https://hci.iwr.uni-heidelberg.de/node/6132)
The bosch dataset had two sets for training and testing.

We have also created our own testing dataset where we took images from the simulator and the traffic light bag
to be able to test our trained model with.

### Training and Testing

We have utilized the tensorflow object detection api and its zoo of availble frozen models during this project.
We have focused on using the Fast Resnet RCNN due to its fast realtime detection.

If you would like to setup the API, you can find instructions here:
[Installation Instructions](https://github.com/tensorflow/models/blob/18a4e59fd7209422a9fa23c9c950876299ce534d/research/object_detection/g3doc/installation.md)

You can see all the models at the zoo here: [API Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

We have used specifically the Resnet RCNN 101 trained on the COCO dataset due to its rich featureset.

Thanks to the awesome post by Anthony Sarkis [Anthony's Medium Post](https://medium.com/@anthony_sarkis/self-driving-cars-implementing-real-time-traffic-light-detection-and-classification-in-2017-7d9ae8df1c58)

The object detection api also offers some samples of train python classes and configurations to easily load
model's layers and retrain using frozen models

You can find out train.py edited from tensorflow's here:[Train.py](https://github.com/jendrikjoe/UdacityCapstone/blob/master/resnet_rcnn/models/train/train.py)

There are also sample configs for each of the frozen models, where we could edit to include paths to the frozen model and edit other features such as batch size, and ofcourse classes to be able to further train. Here is our config: [coco resnet rcnn config](https://github.com/jendrikjoe/UdacityCapstone/blob/master/resnet_rcnn/faster_rcnn_resnet101_coco.config)

To start training as well, here is what your train command would look like:

```
python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=/mnt/d/SDCND/UdacityCapstone/res
net_rcnn/faster_rcnn_resnet101_coco.config
```

### Classes and Labels

Thanks to Anthony Sarkis's blog post, and the good documentation provided at the Bosch Dataset, we were able to see that we have 14 unique classes. 

The Tensorflow Detection API takes in labels in a pbtxt format where we list items with indecies.

Our labels pbtxt can be listed and found here:

[Labeled Classes ](https://github.com/jendrikjoe/UdacityCapstone/blob/master/resnet_rcnn/data/bosch_label_map.pbtxt)

### Saving and freezing model

After training, the most important thing to do is to freeze the checkpoint of the model which we would like to use in our project. Luckily tensorflow detection API does have an example python file to export the features and freeze the model

Here is our export features adapted python file:

[Export and Freeze Model ](https://github.com/jendrikjoe/UdacityCapstone/blob/master/resnet_rcnn/export_inference_graph.py)

When you run it, your command will look something like:

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix ./data/train/model.ckpt-2000 --output_directory ./fine_tuned_model
```

### Validation pre integration

It was really important to be able to test our model before plugging it into our ros architecture. The best way to do this was to create our own python notebook based on the Tensorflow Object Detection tutorial one.

To view our validation pipeline, here is the notebook with the frozen models loaded to test:
[Testing Notebook](https://github.com/jendrikjoe/UdacityCapstone/blob/master/resnet_rcnn/udacity-traffic-light.ipynb)

### Integration with ROS

After freezing the model, and storing the frozen model somewhere within the approach of the ROS architecture.

We start building out TrafficLight Classifier in ROS by loading the model's graph. We can load the model graph like exactly how we do at the python notebook:

``` 
detection_graph = tf.Graph()

with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
```

Once the model graph is loaded, you will need to define the input and output tensors for your model to be able to start carrying your predictions pipeline:

```
# Definite input and output Tensors for detection_graph
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
```
We could see from the above tensors we are expecting an image tensor, detection boxes tensor, scores tensors , classes and number of detections tensors as well.

Then we would read our image, convert it into a numpy array and expand the image dimensions. Right after that , we could then run the session to get a prediction:

```
# Actual detection.
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_np_expanded})
```

At this point we have our boxes where the traffic lights are, the scores and our classes. We could then in our tl_classifier decide on what is a minimum threshold where we would consider taking in the boxes, and based on the different classes take an action with our car.

## References
[Great Tutoria for Object Detection](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-2-converting-dataset-to-tfrecord-47f24be9248d)

[Nice Bosch Dataset ROS package](https://github.com/bosch-ros-pkg/bstld)

