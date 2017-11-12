## Controller

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

