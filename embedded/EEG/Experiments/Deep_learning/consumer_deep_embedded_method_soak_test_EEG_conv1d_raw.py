#!/usr/bin/env python
# coding: utf-8

# ## Custom Packages

# In[1]:


import os, sys

processing_source_path = os.path.abspath('./../../../Processing/')
if(processing_source_path not in sys.path):
    sys.path.append(processing_source_path)
from processing import preprocessing
from processing import posprocessing
from processing import FeaturesExtract

Utils_source_path = os.path.abspath('./../../../Utils/')
if(Utils_source_path not in sys.path):
    sys.path.append(Utils_source_path)
from log_soak_test import soak_test

Embedded_source_path = os.path.abspath('./../../../Quantization/functions/')
if(Embedded_source_path not in sys.path):
    sys.path.append(Embedded_source_path)
from Embedded_Model import Embedded_Model


# ## Python Packages

# ##  EDGE AI Deep learning model

# In[2]:


#protocol communication liraries
import zmq
import logging
LOG_LEVEL=logging.DEBUG #change to logging.DEBUG to enable print logs
ZEROMQ_SOCKET="tcp://127.0.0.1:12345"

import numpy as np
import time
import pickle
#import matplotlib.pyplot as plt


# #### Signal Processing functions

# In[3]:


#  EDGE AI pre-definitions
pp = preprocessing(axis=1)
ppos=posprocessing()
features =FeaturesExtract(axis=2)


# #### Params and initial conditions

# In[4]:


params = {
    'dataset_path': '/root/data/data/OpenBMI/',
    'channels': ['O1', 'Oz', 'O2', 'PO10', 'PO9', 'POz', 'PO3', 'PO4'],
    'freqs': [0, 12.0, 8.57, 6.67, 5.45],
    'num_harmonics': 3,
    'extracted_frequencies': [5,55],
    'sfreq': 1000,
    'window': 512,
    'step': 64,
    'features_type': 'psd',
    'preprocessing': 'log10',
    'num_repetitions': 100,
    'num_channels': 8,
    'num_classes':  5,
    'num_samples': 16,
    'num_windows': 35,
    'window_size': 512,
    'batchsize': 32,
    'epochs': 100,
    'dropout_rate': 0.5,
    'print_model': True,
    'num_epochs':60,
    'num_voting': 5,
    'output_function': 'activations.sigmoid',
    'model_name': 'Conv2D_noBN'
}

window_EEG= np.random.uniform(0.00001, 0,size=(params['num_channels'],params['window_size']))
prep_time=[]
AI_time=[]
y_AI_array = np.zeros(params['num_voting'])

psd_2D = np.zeros((params['num_samples'],
                           params['extracted_frequencies'][1]-params['extracted_frequencies'][0],
                           params['num_channels']),dtype=float)


# #### Load the ML model

# In[5]:


#AI_tf_model=Embedded_Model('./../../keras_quantized_model/SSVEP_CNN_1D_minmax_normalized_by_channel_data_Raw_multidense_dataloader_irene')
AI_tf_model=Embedded_Model('./../../keras_quantized_model/SSVEP_CNN_1D_minmax_normalized_by_channel_data_Raw_multihead')
#AI_tf_model2=Embedded_Model('./../../keras_quantized_model/SSVEP_CNN_1D_minmax_normalized_by_channel_data_Raw_custom_dataloader')
AI_tf_model=Embedded_Model('./../../keras_quantized_model/SSVEP_CNN_1D_minmax_normalized_by_channel_data_Raw_multidense_dataloader_irene_weights')


# #### Soak test log settings 

# #### Communication protocol settings

# In[6]:


context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect(ZEROMQ_SOCKET)
subscriber.setsockopt(zmq.SUBSCRIBE, b"")


# ### Main-loop 

# #### Soakt test time settings
# 

# In[7]:


#Acc variables
N_correct=0
N_interactions=0

# log updates time
log_update_time= 60 #seconds

# soak test total time
duration_time= 1 # hour
end_time= time.time()+ 60 * 60 * duration_time

update_time_duration=time.time()


# In[8]:


log_name=f'logs/system_monitor_deep_conv1D_raw_embedded_model_during_{duration_time}_hours_with_noise'
ST=soak_test(log_name=log_name)


# In[9]:


# noise
def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def create_noise(minimum, maximum,data_len):
    x = band_limited_noise(10, 900, data_len, 2048)
    x -= np.min(x)
    x /= (np.max(x))
    x *= (maximum-minimum)
    x += minimum
    return x
max_var=15/10


# #### AI model in loop

# #### Loop

# In[10]:


#while True:
Data=[]
all_labels=[]
all_pred=[]
Labels=[]
while time.time() < end_time:
    data = subscriber.recv()
    # Convert received data in bit(4-bytes) to numpy array
    Data_array = np.frombuffer(data, dtype=np.float32).reshape(-1,params['num_channels']+1).T.astype(np.float32)
    
    label= Data_array[-1,:]
    Labels+=[label]
    Data += [Data_array[:-1,:]]
    
    if len(Data)>= params['step']//64:
        #Data_array=np.expand_dims(Data_array[:-1,:],0)
       
        Data_raw=np.hstack(Data)
        '''
        for ch in range(8):
            noise=create_noise(np.min(np.hstack(Data)[ch,:])/max_var,np.max(np.hstack(Data)[ch,:])*max_var,np.hstack(Data).shape[1])
            Data_raw[ch,:]=Data_raw[ch,:]+noise
        '''  
        
        window_EEG=pp.moving_window(window_EEG,Data_raw,params['step'])   
        Data=[] # reset Data 
        Labels=[]
        tic=time.time()
        #preprcessing
        
        eeg_norm=pp.NormMinMax(window_EEG,new_axis=1) # norm by sample
        #eeg_norm=pp.NormMinMax(window_EEG,new_axis=0) # norm by channel

        eeg_norm=np.expand_dims(eeg_norm,0)
        eeg_norm=np.moveaxis(eeg_norm,1,2)
        #emg_norm=pp.NormMinMax(np.expand_dims(window_EEG,0),new_axis=1)

        prep_time+=[time.time()-tic]
        

        #classification
        tic_model=time.time()

        y_AI_array = np.roll(y_AI_array,-1)

        model_pred=AI_tf_model.classify_data(eeg_norm)

        if np.argmax(model_pred[0]) == 1 and np.argmax(model_pred[1]) != 0 :
        #if np.argmax(model_pred[1]) != 0:
            
            #model_pred2=AI_tf_model2.classify_data(eeg_norm)
            y_AI_array[-1] = np.argmax(model_pred[1][1:])

            AI_time+=[time.time()-tic_model]

            #pos processing 
            y_votting=ppos.majority_voting(y_AI_array, params['num_voting'])
            #y_votting=y_AI_array[-1]


            all_pred+=[y_votting.item()+1]
            all_labels+=[label[-1]]
            # Acc  
            if all_pred[-1]== int(label[-1]):
                N_correct=N_correct+1

            N_interactions=N_interactions+1

    #log results
    if time.time() - update_time_duration>= log_update_time:
        if N_interactions!= 0:

            #reset log time 
            update_time_duration=time.time()
            #comput acc
            Acc=N_correct/N_interactions
            # update params60*
            #print(f'Preprocessing Latency: {round(np.mean(prep_time)*1000,4)}ms | Model Predict Latency: {round(np.mean(AI_time)*1000,4)}ms | Model  accuracy: {Acc}')

            ST.set_model_performance(np.mean(prep_time), np.mean(AI_time),Acc)
            AI_time=[]
            prep_time=[]
            #uncomment to save logs and display it 
            ST.log_info()
            
            N_correct=0
            N_interactions=0
        
print("Média de tempo no pré-processamento",np.array(prep_time).mean() *1000,'ms')
print("Máximo tempo no pré-processamento",np.array(prep_time).max() *1000,'ms')

print("Média de tempo do modelo AI",np.array(AI_time)[10:].mean() *1000,'ms')
print("Máximo tempo do modelo AI",np.array(AI_time)[10:].max() *1000,'ms')

subscriber.close()
context.term()


# In[ ]:




