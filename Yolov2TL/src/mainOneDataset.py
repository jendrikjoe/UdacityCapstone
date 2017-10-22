'''
Created on Jun 8, 2017

@author: jendrik
'''
from Network.Network import YOLOv2
import numpy as np
from matplotlib import image as mpimg
import pickle
from keras.utils import to_categorical
import glob
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from Data.Datasets import BoschChallenge
import time
from datetime import datetime
import cv2
import yaml

if __name__ == '__main__':
    
    
    
    patience = 50
    samplesPerBatch = 1
    numEpochs = 500
    batchPerEpoch = 1024
    batchesPerVal = 128
    count = 0
    startEpoch = 0
    
    
    """shadowVars = {}
    for var in tf.get_collection('movingVars'):
        shadowVars.update({ema.average_name(var) : var})
    print(shadowVars)"""
    stream = open("../data/train.yaml", "r")
    files = yaml.load(stream)
    #print(files)
    dataset = BoschChallenge()
    path = files[0]['path']
    prefix = '../data'
    path = prefix + path[path.find('/'):]
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.show()
    summaryWriter = tf.summary.FileWriter('./log/'+datetime.now().isoformat())
    globalStep = tf.Variable(startEpoch*batchPerEpoch, name='globalStep')
    saveGraph = True
    config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True)
    jitLevel = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jitLevel
    learningRate = tf.train.exponential_decay(
                    1e-4,                # Base learning rate.
                    globalStep,  # Current index into the dataset.
                    1024,          # Decay step.
                    (1.-0.0005),                # Decay rate.
                    staircase=True)
    yolo = YOLOv2([1,img.shape[0],img.shape[1],img.shape[2]], learningRate, startEpoch*batchPerEpoch)
    init = tf.global_variables_initializer()
    localInit = tf.local_variables_initializer()
    tf.logging.set_verbosity(tf.logging.FATAL)
    sess = tf.Session(config=config) 
    sess.run([init, localInit])
    saver = tf.train.Saver()
    for i in np.arange(startEpoch, numEpochs,1):
        if i == 0:
            summaryWriter.add_graph(
                sess.graph
            )
            print(np.min(img))
            print(np.max(img))
                
                
            yolo.setWeights('../darknet19_448.weights', sess)
            res = yolo.eval(img,sess)
            tf.train.write_graph(sess.graph, '../ckpt', 'train.pb', as_text=False)
        else:
            checkpointFile = os.path.join('./ckpt', 'trainCkpt.ckpt')
            saver.restore(sess, checkpointFile)
            
        batchLoss = 0 
        batchAcc = 0
        batchCE = 0
        batchIMU = 0
        for j in range(batchPerEpoch):
            data = None
            while trainQueuer.is_running():
                if not trainQueuer.queue.empty():
                    data = trainQueuer.queue.get()
                    break
                else:
                    time.sleep(.05)
            tmpLoss, tmpAcc, tmpCE, tmpImu = yolo.train(np.array(data[0]['inputImg']), np.array(data[1]['outputImg']), data[2], sess)
            print("Step: %d of %d, Train Loss: %g" % (j, batchPerEpoch, tmpLoss))
            batchLoss += tmpLoss
            batchAcc += tmpAcc
            batchCE += tmpCE
            batchIMU += tmpImu
        summary = tf.Summary()
        batchAcc /= batchPerEpoch
        batchLoss /= batchPerEpoch
        batchCE /= batchPerEpoch
        batchIMU /= batchPerEpoch
        summary.value.add(tag="TrainAccuracy", simple_value=batchAcc)
        summary.value.add(tag="TrainLoss", simple_value=batchLoss)
        summary.value.add(tag="TrainCrossEntropy", simple_value=batchCE)
        summary.value.add(tag="TrainIMU", simple_value=batchIMU)
        # Add it to the Tensorboard summary writer
        # Make sure to specify a step parameter to get nice graphs over time
        testAcc=0
        testLoss=0
        testCE=0
        testIMU=0
        for j in range(batchesPerVal):
            data = None
            while valQueuer.is_running():
                if not valQueuer.queue.empty():
                    data = valQueuer.queue.get()
                    break
                else:
                    time.sleep(.01)
            tmpLoss, tmpAcc, tmpCE, tmpImu = yolo.val(np.array(data[0]['inputImg']), np.array(data[1]['outputImg']), data[2], sess)
            testLoss += tmpLoss
            testAcc += tmpAcc
            testCE += tmpCE
            testIMU += tmpImu
        testAcc/=batchesPerVal
        testLoss/=batchesPerVal
        testCE/=batchesPerVal
        testIMU /= batchPerEpoch
        summary.value.add(tag="ValidationAccuracy", simple_value=testAcc)
        summary.value.add(tag="ValidationLoss", simple_value=testLoss)
        summary.value.add(tag="ValidationCrossEntropy", simple_value=testCE)
        summary.value.add(tag="ValidationIMU", simple_value=testIMU)
        # Add it to the Tensorboard summary writer
        # Make sure to specify a step parameter to get nice graphs over time
        summaryWriter.add_summary(summary, i)
        res = dataset.convertTargetToImage(np.reshape(yolo.eval(image,sess),
                    (inputShape[1], inputShape[2])))
        imageTemp = (255.*image[0]).astype('uint8')
        mpimg.imsave('../images/Epoch%04d.png'%i, cv2.addWeighted(res, 0.5, imageTemp, 0.5, 0))
        checkpointFile = os.path.join('./ckpt', 'trainCkpt.ckpt')            
        saver.save(sess, checkpointFile)
        count += 1
            
        if testCE < (crossEntropyComp - .001):
            checkpointFile = os.path.join('./ckpt', 'segNet.ckpt')
                
            saver.save(sess, checkpointFile)
            count = 0
            crossEntropyComp = testCE
        if count >= patience:
            break
    print("Finished")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    