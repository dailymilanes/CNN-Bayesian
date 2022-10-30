# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:03:49 2021

@author: Daily Milanés Hermosilla
"""

from keras import backend as K
import keras.utils
import numpy as np
import math
import os
import scipy.io as sio

# Load .mat files corresponding to every trial located in a specific directory
# Return four lists, corresponding to left hand, right hand, feet and tongue trials

def load_eeg(dir, strLabels):
    data = []
    labels = []    
    contenido = sorted(os.listdir(dir))
    for fichero in contenido:
        if strLabels[1]=='Foot':
            nombre1='data'
        else:
            nombre1 = fichero[:3] + "trial"
        file_load = sio.loadmat(os.path.join(dir, fichero))
        draw = file_load[nombre1]     
        x = np.array(draw, dtype = np.float32)
        for i in range(0, len(strLabels)):
            strLabel = strLabels[i]
            if strLabel in fichero:
                data.append(np.copy(x))
                labels.append(i)
                break
    return data, labels


# This function create crops for each trial especified in data parameter. 
# cropDistance: is stride between 2 crops
# cropsize: size of each crop, in our model is considered 1000 samples.
# Return a list of numpy array with all crops 



# This function create the numpy arrays to train or to validation from the lists of trials of left hand (leftList),
      # right hand(rightList), feet (footList) and tongue(tongueList)
# indexs is a list that contains the number of trials to train or to validate 	

       
def Generator(trialList, trialLabels, classCount, indexs, channels, cropDistance, cropSize, cropsCount):
    numbers = []
    for i in indexs:           
        for k in range(0, cropsCount):
            numbers.append(np.array([i, k]))   
             
    while 1:
       i = 0
       random.shuffle(numbers)  
       while (i < len(numbers)):
           m = min(32, len(numbers) - i)
           data = []
           labels = []
           for j in range(i, i+m):
               index = numbers[j]  # index es el numbers correspondiente a j, dos parametros
               trial = trialList[index[0]]  # el trial correspodiente al 1er elemento de index
               n = index[1]*cropDistance
               if K.image_data_format() == 'channels_first':
                   crop = np.zeros((1,cropSize, channels), dtype=np.float32)
                   crop[0,:,:] = trial[n:cropSize+n, :]
               else:
                   crop = np.zeros((cropSize, channels,1), dtype=np.float32)    
                   crop[:,:,0] = trial[n:cropSize+n, :]
                   
               data.append(np.copy(crop))
               labels.append(keras.utils.to_categorical(trialLabels[index[0]], classCount).astype('float32'))
           x = np.array(np.copy(data))                
           y = np.array(np.copy(labels))
           i += m
           yield(x, y)
