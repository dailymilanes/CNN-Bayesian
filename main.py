# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 20:09:07 2021

@author: Daily Milanés Hermosilla
"""

# The authors recomend dropoutRate 0.8 to dataset 2a, dropoutRate 0.5 to dataset 2b and dropout 0.9 to dataset IVa


# To ensure repeatability of the experiment #2, #3 and #4, please use seed=1 up to 16

# To run any experiment, select appropietly subject, seed, dropoutRate, cropDistance=2
# cropSize=1000 to datatsets 2a and 2b, cropSize=750 to datset IVa

# nb_classes=4, channels=22, fraction=6 to dataset 2a
# nb_classes=2, channels=3, fraction=5 to dataset 2b
# nb_classes=2, channels=118 to dataset IVa

# Tu run experiment #4, please exclude parameter must be different of 0, and specify unknown subject



import eegTrain
import eegEvaluate

global dataDirectory
eegTrain.dataDirectory = '../Data/'    

global weightsDirectory 
eegTrain.weightsDirectory = '../Weights/'


# Experiment #1
#eegTrain.trainKFold('B01',dropoutRate=0.5, cropDistance = 2, cropSize = 1000)

#Nested KFold

#eegTrain.trainNestedKFold('aa', dropoutRate=0.9, optim='adam', cropDistance = 125, cropSize = 750)

# Experiment #2
# eegTrain.intraSubjectTrain('B01', dropoutRate=0.5, cropDistance = 2, cropSize = 1000)

# # Experiment #3
# eegTrain.interSubjectTrain(dropoutRate=0.5, cropDistance = 2, cropSize = 1000,
#                       nb_classes = 2,exclude = 0)

# # Experiment 4
#eegTrain.trainUnkownSubject(dropoutRate=0.5, cropDistance = 2, cropSize = 1000,
 #                      nb_classes = 2, exclude = 1)
# Evaluate function
# Specified the weightsFileName
#weightsFileName='../Weights/B01_Seed_19_R_1_d_0.50_c_2_x_0_weights.hdf5'
#eegEvaluate.eegEvaluate('B01', cropDistance=2, cropSize=1000, weightsFileName=weightsFileName,
     #                   dropoutRate = 0.5,channels = 3, nb_classes = 2)

