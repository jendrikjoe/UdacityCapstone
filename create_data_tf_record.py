import os
import yaml
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

IMAGE_HEIGHT = 720
IMAGE_WIDHT = 1280

NUM_CLASSES = 14
LABELS_PATH=os.getcwd()

X_MIN_KEY = 'x_min'
X_MAX_KEY = 'x_max'

Y_MIN_KEY = 'y_min'
Y_MAX_KEY = 'y_max'

flags = tf.app.flags
# flags.DEFINE_string('csv_input', '', 'Path to CSV Input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def dataset_labels_dict(file_name="bosch_label_map.pbtxt"):
    label_map = label_map_util.load_labelmap(LABELS_PATH + '/object_detection/data/' + file_name)
    dictionary = {}
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

    for item in categories:
        dictionary[item['name']] = item['id']

    return dictionary


def create_dataset_tf_record(data, dictionary):
    # Those values wil change [TODO]
    height = IMAGE_HEIGHT
    width = IMAGE_WIDHT

    encoded_image = None

    file_name = data['path']
    file_name = file_name.encode()

    with tf.gfile.GFile(data['path'], 'rb') as fid:
        encoded_image = fid.read()
    
    image_format = b'jpg'

    xmins = []
    xmaxs = []

    ymins = []
    ymaxs = []

    classes_text = []
    classes = []

    for box in data['boxes']:
        xmins.append(box[X_MIN_KEY])
        xmaxs.append(box[X_MAX_KEY])
        ymins.append(box[Y_MIN_KEY])
        ymaxs.append(box[Y_MAX_KEY])
        classes_text.append(box['label'].encode())
        classes.append(int(dictionary[box['label']]))
    

    tf_data = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(file_name),
        'image_sourceid': dataset_util.bytes_feature(file_name),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }))

    return tf_data

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    list_of_images = []  # place the dataset here
    dictionary = dataset_labels_dict("bosch_label_map.pbtxt")

    BASE_PATH = '/mnt/f/capstone data/dataset_train_rgb/'
    TEST_YAML = BASE_PATH + 'dataset_test_rgb_bosch/test.yaml'

    #TRAIN_YAML = BASE_PATH + 'dataset_train_rgb/train.yaml'
    #data = yaml.load(open(TRAIN_YAML, 'rb').read())
    data = yaml.load(open(TEST_YAML, 'rb').read())

    print("Currently converting test data : ", len(data))

    # print("Test Data before fucking loops : ", data[0])

    for i in range(len(data)):
        data[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(TEST_YAML), data[i]['path']))
        #print(data[i]['path'])


    # print("Test data after many fucking loops : ", data[0:4])
    for image in data:
        tf_data = create_dataset_tf_record(image, dictionary)
        writer.write(tf_data.SerializeToString())
        print("#")

    writer.close()

if __name__ == '__main__':
    tf.app.run()