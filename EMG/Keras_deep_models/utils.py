#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, activations
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import h5py

def plot_history(history, model_checkpoint_path):
    fig, ax = plt.subplots(1,2, figsize=(16,8))

    ax[0].plot(history.history['loss'], color='b', label="Training loss",linewidth=3)
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0],linewidth=3)
    ax[0].set_title("Loss convergence ")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel('Categorical Crossentropy')
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['categorical_accuracy'], color='b', label="Training accuracy",linewidth=3)
    ax[1].plot(history.history['val_categorical_accuracy'], color='r',label="Validation accuracy",linewidth=3)
    ax[1].set_title("Acurracy convergence ")
    ax[1].set_ylabel("Categorical Acurracy")
    ax[1].set_xlabel("Epochs")

    legend = ax[1].legend(loc='best', shadow=True)

    plt.savefig(model_checkpoint_path+'/Train_validatin_model.png')
    plt.show()

    # plt.savefig('Figures/'+model_checkpoint_path.split('/')[-1]+': Train_validatin_model.png')

def plot_history_accuracy(history, model_checkpoint_path):
    fig, ax = plt.subplots(1,2, figsize=(16,8))

    ax[0].plot(history.history['loss'], color='b', label="Training loss",linewidth=3)
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0],linewidth=3)
    ax[0].set_title("Loss convergence ")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel('Categorical Crossentropy')
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy",linewidth=3)
    ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy",linewidth=3)
    ax[1].set_title("Acurracy convergence ")
    ax[1].set_ylabel("Acurracy")
    ax[1].set_xlabel("Epochs")

    legend = ax[1].legend(loc='best', shadow=True)

    
    plt.savefig(model_checkpoint_path+'/Train_validation_model.png')
    # plt.show()
    # plt.savefig('Figures/'+model_checkpoint_path.split('/')[-1]+': Train_validatin_model.png')
    
def generate_data(loader, model):
    y_test = np.array([])
    y_pred = np.array([])
    for i in tqdm(range(loader.__len__())):
        xData, yData = loader.__getitem__(i)

        y_pred = np.append(y_pred, np.argmax(model.predict(xData), axis=1))
        if len(yData.shape) == 2:
            y_test = np.append(y_test, np.argmax(yData,axis=1))
        else:
            y_test = np.append(y_test, yData)
        
    return y_test, y_pred


def plot_confusion_matrix(y_test, y_pred, gesture_list, model_checkpoint_path):
    cf_matrix = confusion_matrix(y_test, y_pred)

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    class_counts = np.sum(cf_matrix, axis=1)
    annotations = np.zeros((len(class_counts),len(class_counts)))

    for i in range(len(class_counts)):
        for j in range(len(class_counts)):
            annotations[i][j] = cf_matrix[i][j] / class_counts[i]

    group_percentages = ["{0:.2%}".format(value) for value in
                         annotations.flatten()]


    labels = [f"{v2}\n{v3}" for v2, v3 in
              zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(len(class_counts),len(class_counts))

    fig, ax = plt.subplots(figsize=(15,10))
    sns.set(font_scale=1.2) # Adjust to fit
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', cbar=True, xticklabels=gesture_list, yticklabels=gesture_list)
    plt.title('Confusion Matrix')
    plt.savefig(model_checkpoint_path+'/Confusion_Matrix.png')
    #plt.savefig('Figures/'+str(model_checkpoint_path.split('/')[-1])+': Confusion_Matrix.png')

    ax.set_xticklabels(gesture_list, rotation = 45)
    # plt.show()
    
    
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def train_val_test_split(h5_file_name, random_state=42):
    
    with h5py.File(h5_file_name, 'r') as hf:
        y = hf['label'][...]

    x = np.arange(len(y))
    
    x_full_train, x_val, y_full_train, y_val = train_test_split(x, y, test_size=0.2, random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(x_full_train, y_full_train, test_size=0.2, random_state=random_state)
    
    return x_train, x_val, x_test