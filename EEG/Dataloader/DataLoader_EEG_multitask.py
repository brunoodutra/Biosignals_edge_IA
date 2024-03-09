#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

class CustomDataloader(tf.keras.utils.Sequence):
    '''Class used to load Data for model training
    '''

    def __init__(self, h5_file_name, idx, processing=None, batch_size=32, shuffle=True, categorical=True, data_name='eeg', label_name='label', ep_flag_name='ep_flag'):
        '''Class Constructor
        
            Args: 
                h5_file_name (str): path to H5 data
                idx (list[int])   : List of indices to be considered
                data_processing (list[callable]): List of processing methods to be applied to data
                batch_size (int)  : Number of samples to load
        '''
        
      
        self.list_IDs = idx    
        self.indices= np.arange(len(self.list_IDs))
        self.shuffle = shuffle
        self.categorical = categorical
        
        if self.shuffle == True:
            np.random.shuffle(self.indices)

        self.data_processing = processing
        self.batch_size = batch_size

        self.h5_file_name = h5_file_name
        self.data_name = data_name
        self.label_name = label_name
        self.ep_flag_name = ep_flag_name

    def __len__(self):
        ''' Number of possible data generations

            Returns:
                int: Number of batches per epoch the DataLoader may generate
        '''

        return int(np.floor(len(self.list_IDs) / self.batch_size))    
    

    def on_epoch_end(self):
        """Shuffle the data at the end of each epoch"""

        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)           
            
    def __getitem__(self, index):
        '''Generate one batch of data
            Args 
                idx (list[int]): List of data indices to be used

            Returns:
                x (list[list[float]]): List of data samples
                y (list[int])        : List of data labels
        '''
        
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        '''Generate data containing batch_size samples
            Args 
                idx (list[int]): List of data indices to be used

            Returns:
                x (list[list[float]]): List of data samples
                y (list[int])        : List of data labels
        '''
        
        # Initialization
        list_IDs_temp = np.sort(list_IDs_temp)
    
        with h5py.File(self.h5_file_name, 'r') as hf:
            
            batch_input = (hf[self.data_name][list_IDs_temp,:,:,:])
            batch_output = hf[self.label_name][list_IDs_temp]
            batch_output2 = hf[self.ep_flag_name][list_IDs_temp]
                          
        if self.data_processing != None:
            x, (y, y2) = self.data_processing(batch_input, (batch_output, batch_output2))
            # print('dataloader x',x.shape)
            # print(y)
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            y2 = tf.convert_to_tensor(y2, dtype=tf.float32)
        else:
            x = tf.convert_to_tensor((batch_input), dtype=tf.float32)
            y = tf.convert_to_tensor(batch_output, dtype=tf.float32)
            y2 = tf.convert_to_tensor(batch_output2, dtype=tf.float32)

        # y -= np.min(y)
        if self.categorical == True:
            if len(y.shape) == 1:
                y = (to_categorical(y[:],num_classes=5))#[:,0]#.reshape(-1,4,1)
                y2 = (to_categorical(y2[:],num_classes=2))

            elif (y.shape[1]) == 1 :
                y = (to_categorical(y[:,0],num_classes=5))#[:,0]#.reshape(-1,4,1)
                y2 = (to_categorical(y2[:,0],num_classes=2))#[:,0]#.reshape(-1,4,1)
            
            return x, (y, y2)

        # print('x min max', tf.math.reduce_max(x),tf.math.reduce_min(x))
        # print('y min max', tf.math.reduce_max(y),tf.math.reduce_min(y))
        # print('y2 min max', tf.math.reduce_max(y2),tf.math.reduce_min(y2))
        
        return x, y # (y, y2)
