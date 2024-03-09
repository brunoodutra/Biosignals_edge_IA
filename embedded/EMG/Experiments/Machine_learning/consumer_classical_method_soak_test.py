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

# In[2]:


#protocol communication liraries
import zmq
import logging
LOG_LEVEL=logging.DEBUG #change to logging.DEBUG to enable print logs
ZEROMQ_SOCKET="tcp://127.0.0.1:53421"

import numpy as np
import time
import pickle


# ##  EDGE AI Machine learning model

# #### Params and initial conditions

# In[3]:


Ma_length=31
window_length=128
tke_length=1
window_length_init=window_length+Ma_length+tke_length
channels=64
window_step=64

window_emg= np.random.uniform(0.00001, 0,size=(1,channels,window_length))
# initializing variables to initial conditions
data_prev = window_emg[:,:,-Ma_length-1:].copy()
prep_time=[]
AI_time=[]
all_emg_data=[]
n_voting=5
y_AI_array = np.zeros([n_voting])
selected_features = ['dvar', 'iemg', 'mav', 'arv', 'cog', 'auc', 'M2', 'danv', 'var', 'dasdv']


# #### Signal Processing functions

# In[4]:


pp = preprocessing(axis=2)
ppos=posprocessing()
features =FeaturesExtract(axis=2, selected_features=selected_features)


# #### Load the ML model

# In[5]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
with open('./../../quantized_model/model_LDA_25295225.hdf5', 'rb') as file:
    AI_model = pickle.load(file)


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


log_name=f'logs/system_monitor_machine_learning_model_during_{duration_time}_hours'
ST=soak_test(log_name=log_name)


# #### AI model in loop

# In[9]:


while time.time() < end_time:
    
    data = subscriber.recv()
    #logging.debug(f"recv len {len(data)} bytes from publisher")
    #data processing start here

    # Convert received data in bit(4-bytes) to numpy array
    Data_array = np.frombuffer(data, dtype=np.float32).reshape(window_step,channels+1).T
    
    label= Data_array[-1,:]
    
    Data_array=np.expand_dims(Data_array[:-1,:],0)

    # windowing 
    window_emg=pp.moving_window(window_emg,Data_array,window_step)
    tic=time.time()
    #preprcessing

    emg_ret=pp.retification(window_emg)
    emg_ret= np.concatenate((data_prev, emg_ret),axis=2)

    data_prev = emg_ret[:,:,-Ma_length-1:].copy()

    emg_tke=pp.TKE(emg_ret)
    
    emg_MA=pp.MAfilter(emg_tke,N=Ma_length)
    
    emg_norm=pp.NormMinMax(emg_MA,new_axis=1)
    #emg_norm=emg_MA
    prep_time+=[time.time()-tic]
    
    # extract time features

    features_time=features.comput_features(emg_norm,axis=2)
    
    features_time_norm=pp.NormMinMax(features_time,new_axis=1)

    features_to_model=np.vstack(features_time_norm.flatten())

    #classification
    tic_model=time.time()

    y_AI_array = np.roll(y_AI_array,-1)
    
    y_AI_array[-1] = AI_model.predict(features_to_model.T)

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




