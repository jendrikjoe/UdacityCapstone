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
    
    def __init__(self, inputShape, learningRate, globalStep):
        super(YOLOv2, self).__init__()
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
            self.outputs.append(conv(xC))
        print(number)
        
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
    
    
    
    
    
