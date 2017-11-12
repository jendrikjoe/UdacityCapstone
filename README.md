## Waypoint Updater

The waypoint updater provides the next 200 waypoints for the car from its current position. Hereby, it is decreasing the waypoint speed down to zero if a red light is coming up, such that the car stops at the stopping line until the light turns green. The target speed is hereby changed from the waypoints speed down to 1.5m/s from 50 waypoints prior to the traffic light until the traffic light. Then the speed is kept at 1.5 m/s until the waypoint prior to the stopping line.



## Controller

### Throttle Controller

Throttle and brake commands are determined by a PID controller what values were determined using the twiddle algorithm. The error was hereby the sum of the error in velocity and the distance of the commanded values from the mean commanded value. It was then twiddled two times. A pre-twiddling with just a constant speed was used to determine the right region of controller values. For this every parameter combination was tested for 20s. Then the determined values were taken and the car in the simulator had to obey the traffic lights. Now each parameter set was tested for 10 mins. This procedure let to the following PID values: [0.025, 0.234, 0.080].

### Steering Controller

The provided model based steering controller was used as is.

## Transfer learning and deep learning

## Dataset used
### Bosch dataset ROS tools :) To get labels etc:
https://github.com/bosch-ros-pkg/bstld

### Awesome tutorial for Bosch dataset:
https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-2-converting-dataset-to-tfrecord-47f24be9248d

## Setup object detection

Follow the following wiki for installation instructions, and make sure you have updated the python path as indicated in the instructions:
https://github.com/tensorflow/models/blob/18a4e59fd7209422a9fa23c9c950876299ce534d/research/object_detection/g3doc/installation.md

## Run training on resnet

To kick off training, cd into the following directory

```
cd resnet_rcnn/models/train
```

Run the following command to start training:

```
python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=/mnt/d/SDCND/UdacityCapstone/res
net_rcnn/faster_rcnn_resnet101_coco.config
```

After training is complete, we will need to extract the features to be able to freeze our trained model:

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./rcnn_resnet101_coco.config --trained_checkpoint_prefix ./models/train/model.ckpt-5000 --output_directory ./fine_tuned_model
```

