'''
Created on May 31, 2017

@author: jendrik
'''
from abc import ABC, abstractmethod
import glob
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from sklearn.model_selection._split import train_test_split
import pickle
from Preprocess.Preprocess import mirrorImage, augmentImageAndLabel
from keras.utils.np_utils import to_categorical
from keras.layers import Input
import threading
from multiprocessing import Queue
from keras.engine.training import GeneratorEnqueuer
from collections import namedtuple  
import multiprocessing
from multiprocessing import Pool
from matplotlib import pyplot as plt
import functools
import yaml
from src.Network.Network import YOLOv2

class Dataset(ABC):
    '''
    classdocs
    '''


    def __init__(self, path, **kwargs):
        '''
        Constructor
        '''
        self.path = path
        self.trainData = None
        self.valData = None
        
    @abstractmethod
    def generator(self, batchSize, train, **kwargs):
        pass
    
    def buildInputLayer(self):
        inputShape = mpimg.imread(self.trainData.iloc[0]['input']).shape
        return Input(shape=(inputShape[0], inputShape[1], inputShape[2]), name='inputImg')
    
    @abstractmethod
    def getTrainDistribution(self):
        pass
    
    @abstractmethod
    def setWeights(self):
        pass
    
class BoschChallenge(Dataset):
    
    def __init__(self, **kwargs):
        super(BoschChallenge, self).__init__(path=None)
        stream = open("../data/train.yaml", "r")
        files = yaml.load(stream)
        df = pd.DataFrame(files)
        df['path'] = df['path'].apply(lambda x: '../data/'+x[x.find('/'):])
        self.trainData, self.valData = train_test_split(df, test_size=.2)
        
    def generator(self, batchSize, train, **kwargs):
        start=0
        inputShape = mpimg.imread(self.trainData.iloc[0]['path'])[::2,::2]
        data = self.trainData if train else self.valData
        while(1):
            imageArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
            targets = []
            #weights = np.ones((batchSize, self.numberOfClasses))
            weights = np.zeros((batchSize, inputShape[0] * inputShape[1]))
            if train:
                indices = np.random.randint(0, len(data), batchSize)
            else:
                indices = np.zeros(batchSize).astype(int)
                for i in range(batchSize):
                    indices[i] = start%len(data)
                    start += 1
                start = start%len(data)
            for i, index in zip(range(len(indices)), indices):
                #if(not train): print(imgList[index])
                image = mpimg.imread(data.iloc[index]['path'])
                target = data.iloc[index]['boxes']
                targets.append(target)
                
                #flip = np.random.rand()
                #if flip>.5:
                #    image = mirrorImage(image)
                #    target = mirrorImage(target)
                #if train: image, target = augmentImageAndLabel(image, target)
                imageArr[i] = image
                
            yield({'inputImg': imageArr}, {'targets': targets})
            
    
    def getTrainDistribution(self):
        subDfs = []
        # Zähle die Prozessoren
        cores = multiprocessing.cpu_count() 
        # Teile den DataFrame in cores gleichgrosse Teile
        for i in range(cores):
            subDfs.append(self.trainData.iloc[int(i/cores*len(self.trainData)):int((i+1)/cores*len(self.trainData)), :])
        print("Build SubDfs")
        # Öffne einen Pool mit ensprechend vielen Prozessen 
        pool = Pool(processes=cores)
        # Wende die Funktion an
        func = functools.partial(BoschChallenge.getSubTrainDistribution, numberOfClasses=self.numberOfClasses, dictionary=self.dict)
        arrs = pool.map(func, subDfs)
        pool.close()
        arr = np.zeros(self.numberOfClasses)
        for i in range(len(arrs)):
            arr += arrs[i]
            
        return arr
    
    
    def setWeights(self, weights):
        self.weights = weights
    
