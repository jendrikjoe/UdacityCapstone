import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

NUM_CLASSES = 14
LABELS_PATH=os.getcwd()

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to CSV Input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def dataset_labels_dict(file_name="bosch_label_map.pbtxt"):
    label_map = label_map_util.load_labelmap(LABELS_PATH + '/' + file_name)
    '''
    TODO plot categories for visualization
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    '''
    return label_map


def create_dataset_tf_record(encoded_data):
    # Those values wil change [TODO]
    height = 800
    width = 600

    file_name = ''
    image_format = b'png'

    xmins = [600 / 200]
    xmaxs = [800 / 150]

    ymins = [600/300]
    ymaxs = [600/200]

    classes_text = dataset_labels_dict("bosch_label_map.pbtxt")
    classes = 14

    tf_data = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(file_name),
        'image_sourceid': dataset_util.bytes_feature(file_name),
        'image/encoded': dataset_util.bytes_feature(encoded_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }))

    return tf_data

def main(_):
    list_of_images = []  # place the dataset here
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    for image in list_of_images:
        tf_data = create_dataset_tf_record(image)
        writer.write(tf_data.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()