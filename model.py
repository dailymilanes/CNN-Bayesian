# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:03:49 2021

@author: Daily Milanés Hermosilla
"""

import keras.utils
from keras import backend as K
from keras.constraints import max_norm
from keras.layers import Conv2D, Input, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Model

def square(x): 
    return K.square(x) 

def log(x): 
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000)) 

# Deterministic model
def createModel(nb_classes = 4, Chans = 22, Samples = 1000, dropoutRate = 0.5, init='he_uniform'): 
    my_shape = (Samples, Chans, 1)
    if K.image_data_format() == 'channels_first':
        my_shape = (1, Samples, Chans)
    input_main   = Input(my_shape) 
    block = Conv2D(40, (45, 1), strides = (2, 1), use_bias = False,
                    input_shape = my_shape, 
                    kernel_constraint = max_norm(3.0, axis=(0,1,2)),name='TimeConv')(input_main) 
    block = Conv2D(40, (1, Chans), use_bias=False,  name='channelConv')(block)
    block = BatchNormalization(epsilon=1e-05, momentum=0.1)(block) 
    block = keras.layers.Activation(square)(block)
    block  = keras.layers.AveragePooling2D(pool_size=(45, 1), strides=(8, 1))(block) 
    block  = keras.layers.Activation(log)(block) 
    flatten = Flatten()(block) 
    block  = Dropout(dropoutRate)(flatten)
    dense   = Dense(nb_classes, kernel_constraint = max_norm(0.5),name='Dense')(flatten) 
    softmax = keras.layers.Activation('softmax')(dense)      
    return Model(inputs=input_main, outputs=softmax) 

# estocastic model Prior as a Standard Gaussian

def SCNBayesianTL(nb_classes, Chans, Samples, dropoutRate,cropDistance,count_trial):
    tfd = tfp.distributions
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p)/tf.cast(int(count_trial*math.ceil((1125-Samples)/cropDistance)), dtype=tf.float32))
   
    my_shape = (Samples, Chans, 1)
    if K.image_data_format() == 'channels_first':
        my_shape = (1, Samples, Chans)
    # start the model     
    input_main = Input(my_shape) 
    block1 = tfp.python.layers.Convolution2DFlipout(40, kernel_size=(45, 1), strides = (2, 1), input_shape = my_shape, kernel_prior_fn=default_multivariate_normal_fn1,
                                                   kernel_divergence_fn=kl_divergence_function)(input_main) 
    block1 = tfp.python.layers.Convolution2DFlipout(40, (1, Chans),kernel_prior_fn=default_multivariate_normal_fn2,kernel_divergence_fn=kl_divergence_function)(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1) 
    block1 = tf.keras.layers.Activation(square)(block1)
    block1 = tf.keras.layers.AveragePooling2D(pool_size=(45, 1), strides = (1,1))(block1)
    block1 = tf.keras.layers.MaxPooling2D(pool_size=(8, 1))(block1)
    block1  = tf.keras.layers.Activation(log)(block1) 
    flatten = Flatten()(block1) 
  #  block1 = tf.keras.layers.Dropout(dropoutRate)(flatten)
    dense   = tfp.python.layers.DenseFlipout(nb_classes,kernel_prior_fn=default_multivariate_normal_fn3, bias_prior_fn=default_multivariate_normal_fn4,
                                              kernel_divergence_fn=kl_divergence_function,activation=tf.nn.softmax)(flatten) 
    return Model(inputs=input_main, outputs=dense)


# estocastic model MOPED
def SCNBayesianTL(nb_classes, Chans, Samples, dropoutRate,cropDistance,count_trial):
    tfd = tfp.distributions
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p)/tf.cast(int(count_trial*math.ceil((1125-Samples)/cropDistance)), dtype=tf.float32))
   
    my_shape = (Samples, Chans, 1)
    if K.image_data_format() == 'channels_first':
        my_shape = (1, Samples, Chans)
    # start the model     
    input_main = Input(my_shape) 
    block1 = tfp.python.layers.Convolution2DFlipout(40, kernel_size=(45, 1), strides = (2, 1), input_shape = my_shape, kernel_prior_fn=default_multivariate_normal_fn1,
                                                   kernel_divergence_fn=kl_divergence_function)(input_main) 
    block1 = tfp.python.layers.Convolution2DFlipout(40, (1, Chans),kernel_prior_fn=default_multivariate_normal_fn2,kernel_divergence_fn=kl_divergence_function)(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1) 
    block1 = tf.keras.layers.Activation(square)(block1)
    block1 = tf.keras.layers.AveragePooling2D(pool_size=(45, 1), strides = (1,1))(block1)
    block1 = tf.keras.layers.MaxPooling2D(pool_size=(8, 1))(block1)
    block1  = tf.keras.layers.Activation(log)(block1) 
    flatten = Flatten()(block1) 
  #  block1 = tf.keras.layers.Dropout(dropoutRate)(flatten)
    dense   = tfp.python.layers.DenseFlipout(nb_classes,kernel_prior_fn=default_multivariate_normal_fn3, bias_prior_fn=default_multivariate_normal_fn4,
                                              kernel_divergence_fn=kl_divergence_function,activation=tf.nn.softmax)(flatten) 
    return Model(inputs=input_main, outputs=dense)
 
 
 
def default_multivariate_normal_fn1(dtype, shape, name, trainable,
                                   add_variable_fn):
    global weights_layer1
    global var_layer1
    
    if len(shape)==4:
       dist = normal_lib.Normal(
         loc=weights_layer1[0,:,:,:,:], scale=var_layer1[0,:,:,:,:])
    elif len(shape)==1: 
       dist = normal_lib.Normal(
         loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(0))
       
    batch_ndims = tf.size(dist.batch_shape_tensor())   
    return independent_lib.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)  
 
def default_multivariate_normal_fn2(dtype, shape, name, trainable,
                                   add_variable_fn):
    global weights_layer2
    global var_layer2
    
    if len(shape)==1:
      dist = normal_lib.Normal(
        loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(0))
    elif len(shape)==4: 
      dist = normal_lib.Normal(
        loc=weights_layer2[0,:,:,:,:], scale=var_layer2[0,:,:,:,:])
      
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return independent_lib.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)  
     
def default_multivariate_normal_fn3(dtype, shape, name, trainable,
                                   add_variable_fn):
    global weights_layer33
    print(shape)
      
    weights_layer3 = tf.convert_to_tensor(weights_layer33[0],tf.float32)
    z=np.abs(weights_layer33[0])
    var_layer3=np.sqrt(0.1*z)
       
    dist = normal_lib.Normal(loc=weights_layer3, scale=var_layer3)
    
      
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return independent_lib.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)  
        
def default_multivariate_normal_fn4(dtype, shape, name, trainable,
                                   add_variable_fn):
    global weights_layer33
    print(shape)
      
    bias_layer3=tf.convert_to_tensor(weights_layer33[1],tf.float32)
    z=np.abs(weights_layer33[1])
    bias_var3=np.sqrt(0.1*z)
    
   
    dist = normal_lib.Normal(
    loc=bias_layer3, scale=bias_var3)
      
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return independent_lib.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)


