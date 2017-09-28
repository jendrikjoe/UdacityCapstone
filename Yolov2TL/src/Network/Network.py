'''
Created on Jun 8, 2017

@author: jendrik
'''


from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import functools
from .Layer.Layer import ConvolutionLayer, DeconvolutionLayer, ConvolutionalBatchNormalization
from tensorflow.python.training import moving_averages
from Network.Layer.Layer import SplittedDeconvolutionLayer
#from tf.contrib.keras import MaxPooling2D

class Network(ABC):
    '''
    A class representation of a neural network
    '''


    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.avg = None
        self.groundTruths = []
    
    @abstractmethod
    def setWeights(self): pass
        
    def train(self, context, groundTruth, weights, sess):
        
        trainLoss, trainAcc, crossEntropy, imu, _ = sess.run([self.loss, self.accuracy, self.crossEntropy, self.imu, self.trainStep], 
                                                        feed_dict={self.inputs[0]: context,
                                                                self.labels: groundTruth, self.trainPh: True,
                                                                self.keepProbSpatial: .7, self.keepProb: .6, self.keepProbAE: .5, self.weights: weights})
        print(imu)
        return trainLoss, trainAcc, crossEntropy, imu
    
    def val(self, context, groundTruth, weights, sess):
        valLoss, valAcc, crossEntropy, imu = sess.run([self.loss, self.accuracy, self.crossEntropy, self.imu], feed_dict={self.inputs[0]: context, 
                                                                self.labels: groundTruth, self.trainPh: False,
                                                                self.keepProbSpatial: 1., self.keepProb: 1.,self.keepProbAE: 1., self.weights: weights})
        return valLoss, valAcc, crossEntropy, imu
    
    def eval(self,context, sess):
        return sess.run([self.outputs[0]], feed_dict={self.inputs[0]: context, self.trainPh: False,
                                                                 self.keepProbSpatial: 1., self.keepProb: 1.,self.keepProbAE: 1.})
    
    def evalWithAverage(self, context, sess):
        return sess.run([self.avg], feed_dict={self.inputs[0]: context, self.trainPh: False,
                                                                 self.keepProbSpatial: 1., self.keepProb: 1.,self.keepProbAE: 1.})

class YOLOv2(Network):
    
    def __init__(self, inputShape, learningRate, globalStep, numberOfBoxes, numberOfClasses):
        super(YOLOv2, self).__init__()
        self.numberOfBoxes = numberOfBoxes
        self.numberOfClasses = numberOfClasses
        self.inputs.append(tf.placeholder(tf.float32, shape=(None,inputShape[1],inputShape[2],inputShape[3]), name='inputImage'))
        
        self.groundTruths.append(tf.placeholder(tf.int32, shape=(None, int(inputShape[1])*int(inputShape[2])), name='gtImage'))
        self.weights = tf.placeholder(tf.float32, shape=(None, int(inputShape[1])*int(inputShape[2])), name='weights')
        self.trainPh = tf.placeholder(tf.bool, name='training_phase')
        self.keepProbSpatial = tf.placeholder(tf.float32, name='keepProbSpatial')
        self.keepProb = tf.placeholder(tf.float32, name='keepProb')
        self.keepProbAE = tf.placeholder(tf.float32, name='keepProbAE')
        self.layers = {}
        self.learningRate = learningRate
        self.globalStep = globalStep
        
        number = 0
        
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, 3, 32, stride=(1,1), padding='SAME', useBias=False, trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(self.inputs[0])
            bn = ConvolutionalBatchNormalization(number, 32,trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
            xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
        
        
        print(xC.get_shape())
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 64, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
            
        xC4 = xC
        
        xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
        
        number += 1
        
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 128, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 64, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 128, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        xC3 = xC
        
        xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
        
        number += 1
        
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 256, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 128, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 256, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        xC2 = xC
        
        xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')

        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 256, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 256, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        xC1 = xC
        
        xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 1024, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 1024, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
            
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 1024, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
            
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 125, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            out = conv(xC)
            self.outputs.append(out)
        print(number)
        
        self.addLossToCollection(out)
        
        self.labels = tf.placeholder(tf.int64, shape=(None, inputShape[1]* inputShape[2]))
        #weightsPh = tf.placeholder(tf.float32, shape=(weights.shape[1]))
        print(self.outputs[0].get_shape(), self.labels.get_shape())
        
        
            
        self.loss = tf.reduce_sum(tf.stack(tf.get_collection('my_losses')))
        #self.loss = tf.scan(lambda a, x: tf.scalar_mul(x[0], x[1]), (weightsPh,loss))
         
        #self.trainStep = tf.train.MomentumOptimizer(learning_rate=self.learningRate, momentum=0.9).minimize(self.loss, self.globalStep)
        self.trainStep = tf.train.AdamOptimizer(5e-4).minimize(self.loss)
        #with tf.control_dependencies(self.movingVars):
        #    self.trainStep = trainStep
        correctPrediction = tf.equal(self.outputs, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        
    def addLossToCollection(self, out):
        """
        Takes net.out and placeholders value
        returned in batch() func above,
        to build train_op and loss
        """
        # meta 
        m = self.meta
        sprob = 1#float(m['class_scale'])
        sconf = .5#float(m['object_scale'])
        snoob = .5#float(m['noobject_scale'])
        scoor = 1#float(m['coord_scale'])
        h = out.get_shape()[1]
        w = out.get_shape()[2]
        numberOfGridCells = h * w 
        
        anchors = m['anchors']
    
        print('{} loss hyper-parameters:'.format(m['model']))
        print('\tH       = {}'.format(h))
        print('\tW       = {}'.format(w))
        print('\tbox     = {}'.format(self.numberOfBoxes))
        print('\tclasses = {}'.format(self.numberOfClasses))
        print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))
    
        size1 = [None, numberOfGridCells, self.numberOfBoxes, self.numberOfClasses]
        size2 = [None, numberOfGridCells, self.numberOfBoxes]
    
        # return the below placeholders
        _probs = tf.placeholder(tf.float32, size1)
        _confs = tf.placeholder(tf.float32, size2)
        _coord = tf.placeholder(tf.float32, size2 + [4])
        # weights term for L2 loss
        _proid = tf.placeholder(tf.float32, size1)
        # material calculating IOU
        _areas = tf.placeholder(tf.float32, size2)
        _upleft = tf.placeholder(tf.float32, size2 + [2])
        _botright = tf.placeholder(tf.float32, size2 + [2])
    
        self.placeholders = {
            'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
            'areas':_areas, 'upleft':_upleft, 'botright':_botright
        }
    
        # Extract the coordinate prediction from net.out
        net_out_reshape = tf.reshape(out, [-1, h, w, self.numberOfBoxes, (4 + 1 + self.numberOfBoxes)])
        coords = net_out_reshape[:, :, :, :, :4]
        coords = tf.reshape(coords, [-1, numberOfGridCells, self.numberOfBoxes, 4])
        adjusted_coords_xy = self.expit_tensor(coords[:,:,:,0:2])
        adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, self.numberOfBoxes, 2]) / np.reshape([w, h], [1, 1, 1, 2]))
        coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)
    
        adjusted_c = self.expit_tensor(net_out_reshape[:, :, :, :, 4])
        adjusted_c = tf.reshape(adjusted_c, [-1, numberOfGridCells, self.numberOfBoxes, 1])
    
        adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
        adjusted_prob = tf.reshape(adjusted_prob, [-1, numberOfGridCells, self.numberOfBoxes, self.numberOfClasses])
    
        adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)
    
        wh = tf.pow(coords[:,:,:,2:4], 2) * np.reshape([w, h], [1, 1, 1, 2])
        area_pred = wh[:,:,:,0] * wh[:,:,:,1]
        centers = coords[:,:,:,0:2]
        floor = centers - (wh * .5)
        ceil  = centers + (wh * .5)
    
        # calculate the intersection areas
        intersect_upleft   = tf.maximum(floor, _upleft)
        intersect_botright = tf.minimum(ceil , _botright)
        intersect_wh = intersect_botright - intersect_upleft
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])
    
        # calculate the best IOU, set 0.0 confidence for worse boxes
        iou = tf.truediv(intersect, _areas + area_pred - intersect)
        best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
        best_box = tf.to_float(best_box)
        confs = tf.multiply(best_box, _confs)
    
        # take care of the weight terms
        conid = snoob * (1. - confs) + sconf * confs
        weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
        cooid = scoor * weight_coo
        weight_pro = tf.concat(self.numberOfClasses * [tf.expand_dims(confs, -1)], 3)
        proid = sprob * weight_pro
    
        self.fetch += [_probs, confs, conid, cooid, proid]
        true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3)
        wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3)
    
        print('Building {} loss'.format(m['model']))
        loss = tf.pow(adjusted_net_out - true, 2)
        loss = tf.multiply(loss, wght)
        loss = tf.reshape(loss, [-1, numberOfGridCells*self.numberOfBoxes*(4 + 1 + self.numberOfClasses)])
        loss = tf.reduce_sum(loss, 1)
        loss = .5 * tf.reduce_mean(loss)
        tf.add_to_collection('my_losses', loss)
        
        
    @staticmethod
    def expit_tensor(x):
        return 1. / (1. + tf.exp(-x))

    def setWeights(self, weightFile, sess):
        weightFile = open(weightFile, 'rb')  # ../darknet19_448.weights
        weights_header = np.ndarray(
                shape=(4,), dtype='int32', buffer=weightFile.read(16))
        print('Weights Header: ', weights_header)
        weightLoader = functools.partial(YOLOv2.load_weights, weightFile=weightFile)
        for i in range(18):
            conv = self.layers['conv%d'%i]
            weights = weightLoader(int(conv.dnshape[3]), int(conv.ksize), True, int(conv.dnshape[2]))
            conv.setWeights(weights)
            bn = self.layers['bn%d'%i]
            bn.setWeights(weights)
        print(tf.get_collection("assignOps"))
        sess.run(tf.get_collection("assignOps"))
        print('Unused Weights: ', len(weightFile.read()) / 4)
    
    @staticmethod
    def load_weights(filters, size, batchNormalisation, prevLayerFilter, weightFile):
        weights = {}
        weights_shape = (size, size, prevLayerFilter, filters)
        # Caffe weights have a different order:
        darknet_w_shape = (filters, weights_shape[2], size, size)
        weights_size = np.product(weights_shape)
        print(weights_shape)
        print(weights_size)
        
        conv_bias = np.ndarray(
                shape=(filters, ),
                    dtype='float32',
                    buffer=weightFile.read(filters * 4))
        weights.update({'bias' :conv_bias})
        
        if batchNormalisation:
            bn_weights = np.ndarray(
                shape=(3, filters),
                dtype='float32',
                buffer=weightFile.read(filters * 12))
            
            weights.update({'gamma' :bn_weights[0]})
            weights.update({'movingMean' :bn_weights[1]})
            weights.update({'movingVariance' :bn_weights[2]})
    
        conv_weights = np.ndarray(
            shape=darknet_w_shape,
            dtype='float32',
            buffer=weightFile.read(weights_size * 4))
    
        # DarkNet conv_weights are serialized Caffe-style:
        # (out_dim, in_dim, height, width)
        # We would like to set these to Tensorflow order:
        # (height, width, in_dim, out_dim)
        # TODO: Add check for Theano dim ordering.
        conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
        weights.update({'kernel' :conv_weights})
        return weights
    
    
    
    
    
