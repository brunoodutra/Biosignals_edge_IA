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


# ## python Packages

# In[2]:


#protocol communication liraries
import zmq
import logging
LOG_LEVEL=logging.DEBUG #change to logging.DEBUG to enable print logs
ZEROMQ_SOCKET="tcp://127.0.0.1:53421"

import numpy as np
import time
import pickle
#import matplotlib.pyplot as plt


# ##  EDGE AI Deep learning model

# #### Signal Processing functions

# In[3]:


pp = preprocessing(axis=1)
ppos=posprocessing()


# #### Params and initial conditions

# In[4]:


Ma_length=31
window_length=128
tke_length=1
img_width=8
img_height=8
channels=64
window_step=64
window_emg= np.random.uniform(0.00001, 0,size=(channels,window_length))
# initializing variables to initial conditions
data_prev = window_emg[:,-Ma_length-1:].copy()
prep_time=[]
AI_time=[]
all_emg_data=[]
n_voting=9
y_AI_array = np.zeros([n_voting])


# #### Load the ML model

# In[5]:


AI_tf_model=Embedded_Model('./../../quantized_model/Resnet2D')


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
classes=[0, 13, 14, 17, 20, 36, 58, 59, 63, 64]


# In[8]:


log_name=f'logs/system_monitor_deep_embedded_model_during_{duration_time}_hours'
ST=soak_test(log_name=log_name)


# #### AI model in loop

# In[ ]:


#while True:
while time.time() < end_time:
    data = subscriber.recv()
    # Convert received data in bit(4-bytes) to numpy array
    Data_array = np.frombuffer(data, dtype=np.float32).reshape(window_step,channels+1).T
    
    label= Data_array[-1,:]
    
    Data_array=np.expand_dims(Data_array[:-1,:],0)
    
    # windowing 
    window_emg=pp.moving_window(window_emg,Data_array,window_step)

    tic=time.time()
    #preprcessing

    emg_ret=pp.retification(window_emg)
    emg_ret= np.concatenate((data_prev, emg_ret),axis=1)
    data_prev = emg_ret[:,-Ma_length-1:].copy()

    emg_tke=pp.TKE(emg_ret)
    emg_MA=pp.MAfilter(emg_tke,N=Ma_length)
    
    emg_norm=pp.NormMinMax(np.expand_dims(emg_MA,0),new_axis=1)
    
    emg_2d_reshape=pp.DataReshape(np.expand_dims(emg_norm,0),(img_height,img_width,window_length))# reshape data to 2D (1,8,8,128)

    prep_time+=[time.time()-tic]
    

    #classification
    tic_model=time.time()

    y_AI_array = np.roll(y_AI_array,-1)
    
    y_AI_array[-1] = np.argmax(AI_tf_model.classify_data(emg_2d_reshape))

    AI_time+=[time.time()-tic_model]

    #pos processing 
    y_votting=ppos.majority_voting(y_AI_array, n_voting)

    # Acc  
    if y_votting == np.where(label[-1]==classes)[0]:
        N_correct=N_correct+1
    
    N_interactions=N_interactions+1

    #log results
    if time.time() - update_time_duration>= log_update_time:
        #reset log time 
        update_time_duration=time.time()
        #comput acc
        Acc=N_correct/N_interactions
        # update params60*
        ST.set_model_performance(np.mean(prep_time), np.mean(AI_time),Acc)
        print(f'Preprocessing Latency: {round(np.mean(prep_time)*1000,4)}ms | Model Predict Latency: {round(np.mean(AI_time)*1000,4)}ms | Model  accuracy: {Acc}')

        AI_time=[]
        prep_time=[]
        # save params 
        #ST.log_info()
        
               
        N_correct=0
        N_interactions=0

subscriber.close()
context.term()


# In[ ]:




