#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, activations
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras import backend as K  # Import Keras backend

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import h5py

#from scipy.fftpack import fft 
from numpy.fft import fft

def get_fft(data, plot_data=True, get_data=False, sample_rate=1000):
    """
    Args:
    - data: Input data for FFT.
    - plot_data: Boolean, whether to plot the frequency spectrum.
    - get_data: Boolean, whether to return the frequency and amplitude data.
    - sample_rate: Sampling rate of the input data.

    Returns:
    If get_data is True:
    - x_freq: Frequencies corresponding to the FFT result.
    - y_freq: Amplitude values of the FFT result.
    """
    y_freq = fft(data)

    N = len(y_freq)
    domain = len(y_freq) // 2
    x_freq = np.linspace(0, sample_rate // 2, N // 2)

    if plot_data:
        plt.plot(x_freq, abs(y_freq[:domain]))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Frequency Amplitude |X(t)|")

    if get_data:
        return x_freq, abs(y_freq[:domain])

    # If get_data is False, return None
    return None

def majority_voting(labels_predicted, n_voting):
    y_pred=[]
    y_model = np.zeros(n_voting)

    for pred in labels_predicted:
        y_model = np.roll(y_model,-1)
        y_model[-1] = pred

        if len(y_pred) < n_voting:
            y_pred += [pred]
        else:
            values,count = np.unique(y_model,return_counts=True)
            if len(count) == n_voting:
                y_pred += [y_pred[-1]]
            else:
                y_pred += [values[np.argmax(count)]]
    return np.array(y_pred)


# Plot training and validation loss/accuracy over epochs
def plot_history_keys(history, keys=None, model_checkpoint_path=None):
    """
    Args:
    - history: Keras model training history.
    - model_checkpoint_path: Path to save the training and validation accuracy plot.
    """
    if keys == None:
        keys=list(history.history.keys())[:len(history.history.keys())//2]
        
    fig, ax = plt.subplots(1, len(keys), figsize=(len(keys)*5, len(keys)*2))
    
    print(keys)
    for idx,key in enumerate(keys): 
        if 'loss' in key:
            label_prefix='Loss'
            label_sufix = 'loss'
        elif 'accuracy' in key:
            label_prefix='Accuracy'
            label_sufix = 'acc'
           
        print(key,idx)
        ax[idx].plot(history.history[key], color='b', label="Training loss", linewidth=3)
        ax[idx].plot(history.history['val_'+key], color='r', label=f"Validation {label_sufix}", axes=ax[idx], linewidth=3)
        ax[idx].set_title(f"{key}")
        ax[idx].set_xlabel("Epochs")
        ax[idx].set_ylabel('Categorical Crossentropy')
        legend = ax[idx].legend(loc='best', shadow=True)
    
    if model_checkpoint_path != None:
        plt.savefig(model_checkpoint_path + '/Train_validation_model.png')

# Plot training and validation loss/accuracy over epochs focusing on accuracy
def plot_history_accuracy(history, model_checkpoint_path):
    """
    Args:
    - history: Keras model training history.
    - model_checkpoint_path: Path to save the training and validation accuracy plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].plot(history.history['loss'], color='b', label="Training loss", linewidth=3)
    ax[0].plot(history.history['val_loss'], color='r', label="Validation loss", axes=ax[0], linewidth=3)
    ax[0].set_title("Loss convergence ")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel('Categorical Crossentropy')
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy", linewidth=3)
    ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy", linewidth=3)
    ax[1].set_title("Accuracy convergence ")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Epochs")

    legend = ax[1].legend(loc='best', shadow=True)

    plt.savefig(model_checkpoint_path + '/Train_validation_model.png')

# Generate predictions and true labels for the given data loader and model
def generate_data(loader, model):
    """
    Args:
    - loader: Data loader providing input data.
    - model: Keras model for predictions.

    Returns:
    - y_test: True labels.
    - y_pred: Predicted labels.
    """
    y_test = np.array([])
    y_pred = np.array([])
    for i in tqdm(range(loader.__len__())):
        xData, yData = loader.__getitem__(i)

        y_pred = np.append(y_pred, np.argmax(model.predict(xData), axis=1))
        if len(yData.shape) == 2:
            y_test = np.append(y_test, np.argmax(yData, axis=1))
        else:
            y_test = np.append(y_test, yData)

    return y_test, y_pred

def generate_data_multihead(loader, model=None):
    '''Loading data into numpy array
        Args:
            loader (object) : A Data Loader object
            model (callable): Model object
        Returns:
            y_test (list[int]): List of data's true labels
            y_pred (list[int]): List of data's predicted labels
    '''
    y_test_freq = np.array([])
    y_test_task = np.array([])
    
    y_pred_freq = np.array([])
    y_pred_task = np.array([])
    
    for i in tqdm(range(loader.__len__())):
        xData, yData = loader.__getitem__(i)

        y_pred = model.predict(xData)
        
        y_pred_freq = np.append(y_pred_freq, np.argmax(y_pred[0], axis=1))
        y_pred_task = np.append(y_pred_task, np.argmax(y_pred[1], axis=1))
        
        y_test_freq = np.append(y_test_freq, np.argmax(yData[0],axis=1))
        y_test_task = np.append(y_test_task, np.argmax(yData[1],axis=1))
        
    return y_test_freq, y_test_task, y_pred_freq, y_pred_task

# Plot a confusion matrix
def plot_confusion_matrix(y_test, y_pred, gesture_list=None, model_checkpoint_path=None, Focus='percent'):
    """
    Args:
    - y_test: True labels.
    - y_pred: Predicted labels.
    - gesture_list: List of gesture labels.
    - model_checkpoint_path: Path to save the confusion matrix plot.
    - Focus: 'percent' for percentage values, 'samples' for sample counts.
    """
    if gesture_list is None:
        gesture_list = np.unique(y_test)

    cf_matrix = confusion_matrix(y_test, y_pred)

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

    class_counts = np.sum(cf_matrix, axis=1)
    annotations = np.zeros((len(class_counts), len(class_counts)))

    for i in range(len(class_counts)):
        for j in range(len(class_counts)):
            annotations[i][j] = cf_matrix[i][j] / class_counts[i]

    group_percentages = ["{0:.5%}".format(value) for value in annotations.flatten()]

    labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(len(class_counts), len(class_counts))

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.set(font_scale=1.5)  # Adjust font scale

    if Focus == 'samples':
        sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', cbar=True, xticklabels=gesture_list,
                    yticklabels=gesture_list)
        
        return cf_matrix
        
    elif Focus == 'percent':
        sns.heatmap(annotations, annot=labels, fmt='', cmap='Blues', cbar=True, xticklabels=gesture_list,
                    yticklabels=gesture_list)
        
        return annotations

    plt.title('Confusion Matrix')

    if model_checkpoint_path is not None:
        plt.savefig(model_checkpoint_path + '/Confusion_Matrix.png')
    ax.set_xticklabels(gesture_list, rotation=45)


# Custom recall metric
def recall_m(y_true, y_pred):
    """
    Args:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - Recall metric.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# Custom precision metric
def precision_m(y_true, y_pred):
    """
    Args:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - Precision metric.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Custom F1-score metric
def f1_m(y_true, y_pred):
    """
    Args:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - F1-score metric.
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Split the dataset into training, validation, and test sets
def train_val_test_split(h5_file_name, random_state=42):
    """
    Args:
    - h5_file_name: Path to the HDF5 file containing the dataset.
    - random_state: Random seed for reproducibility.

    Returns:
    - x_train, x_val, x_test: Indices for training, validation, and test sets.
    """
    with h5py.File(h5_file_name, 'r') as hf:
        y = hf['label'][...]

    x = np.arange(len(y))

    x_full_train, x_val, y_full_train, y_val = train_test_split(x, y, test_size=0.2, random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(x_full_train, y_full_train, test_size=0.2, random_state=random_state)

    return x_train, x_val, x_test
