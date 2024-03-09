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


model_source_path = os.path.abspath('./../../CCA_model/')
if(model_source_path not in sys.path):
    sys.path.append(model_source_path)

from CCA_classify  import CCA_SSVEP


# ## Python Packages

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


# ##  EDGE AI Machine learning model

# #### Signal Processing functions

# In[3]:


pp = preprocessing(axis=1)
ppos=posprocessing()


# #### Params and initial conditions

# In[4]:


#  EEG EDGE AI pre-definitions
pp = preprocessing(axis=1)
ppos=posprocessing()

window_length=2**11
channels=8
window_step=2**9

samples_per_batch=64
sfreq=1000
window_emg= np.random.uniform(0.00001, 0,size=(channels,window_length))
n_voting=5


# #### Load the ML model

# In[5]:


CCA=CCA_SSVEP(sfreq=sfreq,data_length=window_length,ref_freqs = [12, 8.57, 6.67, 5.45],N_harmonics=6)


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
log_update_time= 20 #seconds

# soak test total time
duration_time= 6 # hour
end_time= time.time()+ 60 * 60 * duration_time

update_time_duration=time.time()


# In[8]:


log_name=f'logs/system_monitor_cca_embedded_model_during_{duration_time}_hours'
ST=soak_test(log_name=log_name)


# In[9]:


import os, sys
processing_source_path = os.path.abspath('./../../../../EEG/Processing/')
if(processing_source_path not in sys.path):
    sys.path.append(processing_source_path)
import ProcessingPipeline as pp2


# In[10]:


low_pass =  pp2.LowPassFilter(44, 1000, 5)
high_pass = pp2.HighPassFilter(4, 1000, 5),
anti_aliasing = pp2.LowPassFilter(100, 1000, 5),


# In[11]:


pp2.ProcessingPipeline


# In[12]:


filter_processing = pp2.ProcessingPipeline([
    # band pass
    pp2.HighPassFilter(4, 1000, 5),
    pp2.LowPassFilter(44, 1000, 5),

    # anti aliasing
    pp2.LowPassFilter(100, 1000, 5),
]) 


# #### AI model in loop

# In[ ]:


EEG_Data=[]
prep_time=[]
AI_time=[]
y_CCA_array = np.zeros([n_voting])

All_Y_pred=[]
All_Y=[]
All_Y_pred_votting=[]
while time.time() < end_time:

    data = subscriber.recv()
    # Convert received data in bit(4-bytes) to numpy array
    Data_array = np.frombuffer(data, dtype=np.float64).reshape(samples_per_batch,channels+1).T

    label= Data_array[-1,:] -1
    #labels, counts=np.unique(label,return_counts=True)


    EEG_Data+=[Data_array[:-1,:,]] # wait the buffer increase
    EEG_Data_array=np.hstack(EEG_Data)

    if np.max(EEG_Data_array.shape)>=window_step:
        tic=time.time()

        #apply filters: High 1Hz, LOW 50Hz 
        #for ch in range(channels):
        #    EEG_Data_array[ch,:]= filter_processing(EEG_Data_array[ch,:].reshape(1,-1))
        EEG_Data_array = filter_processing(EEG_Data_array)

        #mean = np.mean(EEG_Data_array, axis=(1), keepdims=True)
        #std = np.std(EEG_Data_array, axis=(1), keepdims=True)
        # z_scale normalization
        #EEG_Data_array = (EEG_Data_array - mean) / std
        
        # windowing 
        window_eeg=pp.moving_window(window_emg,EEG_Data_array,window_step)

        EEG_Data=[]
        prep_time+=[time.time()-tic]
        
        tic_model=time.time()
        # CCA model 
        y_CCA_array = np.roll(y_CCA_array,-1)
        y_CCA_array[-1]=CCA.predict(window_eeg.T)
        #y_votting=CCA.predict(window_eeg.T)
        AI_time+=[time.time()-tic_model]
    
        #pos processing 
        y_votting=ppos.majority_voting(y_CCA_array, n_voting)

        All_Y_pred+=[y_CCA_array[-1]]
        All_Y_pred_votting+=[y_votting]
        All_Y+=[label[-1]]
        
        # Acc  
        if y_votting == label[-1]:
            N_correct=N_correct+1
        
        N_interactions=N_interactions+1

 #log results
    if time.time() - update_time_duration>= log_update_time:
        if N_interactions!= 0:

            #reset log time 
            update_time_duration=time.time()
            #comput acc
            Acc=N_correct/N_interactions
            print(f'Preprocessing Latency: {round(np.mean(prep_time)*1000,4)}ms | Model Predict Latency: {round(np.mean(AI_time)*1000,4)}ms | Model  accuracy: {Acc}')

            # update params60*
            ST.set_model_performance(np.mean(prep_time), np.mean(AI_time),Acc)
            AI_time=[]
            prep_time=[]
            # save params 
            
            #uncomment to save logs and display it 
            #ST.log_info()
            
            N_correct=0
            N_interactions=0

subscriber.close()
context.term()


# In[ ]:


y_CCA_array

