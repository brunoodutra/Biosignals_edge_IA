import os
import sys

import numpy as np

import preprocessing as prep

from scipy import signal

import tensorflow as tf

class ProcessingPipeline:
    """Class used to perform a serie of processing over the data
    """
    def __init__(self, processing_list) -> None:
        """Class constructor to initialize the object

        Args:
            processing_list (list[callable]): list of callable object that accept two object as parameter
        """

        self.processingList = processing_list

    def __call__(self, x_data, y_data=None):
        """Apply the list of processing over the x_data and y_data

        Args:
            x_data (any): x_data to be processed
            y_data (any): y_data to be processed. Defaults to None.

        Returns:
            tuple[any, any]: the processed data 
            any: the processed x_data if the processed y_data == None
        """

        for p in self.processingList:

            x_data, y_data = p(x_data, y_data)

        if y_data is None:
            return x_data
        else:
            return x_data, y_data

    def getProcessingNameList(self):
        """Retrieve the names of the applied processing 

        Returns:
            list[str]: The list with the names
        """
        names = []
        for p in self.processingList:
            names.append(p.getName())
        
        return names
    
class plus_and_sum_constant:
    """Class used to plut and/or sum a constant in signal 
    """

    def __init__(self, Plus=1, Sum=0) -> None:
        """Class constructor to initialize the object
        """
        self.name = 'Plus_Sum'
        self.Plus=Plus
        self.Sum=Sum

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name

    def __call__(self, x_data, y_data=None):
        """Perform operation over the x_data

        Args:
            x_data (ndarray): n-D matrix to be processed
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: tuple with the filtered matrix in the first index
        """
        x_data=(x_data*self.Plus) + self.Sum
        return x_data, y_data
    
class Retification:
    """Class used to apply the retification filter over a matrix
    """

    def __init__(self) -> None:
        """Class constructor to initialize the object
        """
        self.name = 'Retification'

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name

    def __call__(self, x_data, y_data=None):
        """Perform the filter over the x_data

        Args:
            x_data (ndarray): n-D matrix to be processed
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: tuple with the filtered matrix in the first index
        """
        return np.abs(x_data), y_data

class EMGTKE:
    """Class used to apply the TKE filter over a 2D matrix
    """

    def __init__(self) -> None:
        """Class constructor to initialize the object
        """
        self.name = 'EMGTKE'

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name

    def __call__(self, x_data, y_data=None):
        """Perform the filter over the samples of x_data[i, :]

        Args:
            x_data (ndarray): 2D matrix of type float
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: tuple with the filtered 2D matrix in the first index
        """

        x_data = prep.emg_tkeo(x_data)

        return x_data, y_data

class MAFilter:
    """Class used to apply the moving average filter over a 2D matrix
    """

    def __init__(self, l) -> None:
        """Class constructor to initialize the object

        Args:
            l (int): 
        """

        self.l = l
        self.name = 'MAFilter'

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name + '(' + str(self.l) + ")"

    def __call__(self, x_data, y_data=None):
        """Perform the filter over the samples of x_data[i, :]

        Args:
            x_data (ndarray): 2D matrix of type float
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: tuple with the filtered 2D matrix in the first index
        """

        x_data = prep.MAFilter(x_data, self.l)
        
        return x_data, y_data

    
# Classes repetidas iguais NomrMinMax
class Norm:
    """ Class used to normalize a numpy 2D matrix
    """
    
    def __init__(self) -> None:
        """Class constructor to initialize the object
        """
        self.name = 'Norm'

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name

    def __call__(self, x_data, y_data=None):
        """Performs the normalization of the values in x_data

        Args:
            x_data (ndarray): 2D matrix
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: tuple with the normalized matrix in the first index
        """
            
        #x_data /= np.max(x_data)
        
        x_data= (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

        return x_data, y_data
    
# class normalize:
#     """ Class used to normalize a numpy 2D matrix
#     """
    
#     def __init__(self,axis=1) -> None:
#         """Class constructor to initialize the object
#         """
#         self.name = 'normalize'
#         self.axis = axis

#     def getName(self):
#         """Retrieve the name of the preprocessing

#         Returns:
#             str: The string with the name
#         """
#         return self.name

#     def __call__(self, x_data, y_data=None):
#         """Performs the normalization of the values in x_data

#         Args:
#             x_data (ndarray): 2D matrix
#             y_data (any): y_data. Defaults to None.

#         Returns:
#             tuple[ndarray, any]: tuple with the normalized matrix in the first index
#         """
            
#         # Compute minimum and maximum for each channel within each window
#         min_val = np.min(x_data, axis=self.axis, keepdims=True)
#         max_val = np.max(x_data, axis=self.axis, keepdims=True)

#         # Subtract minimum and divide by range for each channel within each window
#         # x_data = (x_data - min_val[:, np.newaxis, :]) / (max_val - min_val)[:, np.newaxis, :]
#         x_data = (x_data - min_val) / (max_val - min_val)
    
#         return x_data, y_data

class StandardScaler:
    """ Class used to StandardScaler a numpy 2D matrix based in mean and std
    # Subtract mean and divide by standard deviation for each channel within each window
    """
    
    def __init__(self, axis) -> None:
        """Class constructor to initialize the object
        
        Args:
            axis: axis to normalize
        """
        
        self.name = 'StandardScaler'
        self.axis = axis

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name

    def __call__(self, x_data, y_data=None):
        """Performs the normalization of the values in x_data

        Args:
            x_data (ndarray): 2D matrix
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: tuple with the normalized matrix in the first index
        """

        mean = np.mean(x_data, axis=self.axis)
        std = np.std(x_data, axis=self.axis)

        
        x_data = (x_data - mean[:, np.newaxis, :]) / std[:, np.newaxis, :]

        return x_data, y_data
    
class NormMinMax:
    """ Class used to normalize a numpy 2D matrix to [0, 1]
    """
    
    def __init__(self, axis, minimum=0, maximum=1) -> None:
        """Class constructor to initialize the object
        
        Args:
            axis: axis to normalize
        """
        
        self.name = 'NormMinMax'
        self.axis = axis
        self.minimum = minimum
        self.maximum = maximum

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name

    def __call__(self, x_data, y_data=None):
        """Performs the normalization of the values in x_data

        Args:
            x_data (ndarray): 2D matrix
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: tuple with the normalized matrix in the first index
        """

#         samples_min = x_data.min(axis=self.axis,keepdims=True)
#         x_data = (x_data - samples_min)
#         samples_max = x_data.max(axis=self.axis,keepdims=True)
#         x_data = x_data / samples_max


        samples_min = x_data.min(axis=self.axis,keepdims=True) + sys.float_info.epsilon
        samples_max = x_data.max(axis=self.axis,keepdims=True)
        x_data = (x_data - samples_min) * (self.maximum - self.minimum)
        x_data = (x_data / (samples_max - samples_min) ) + self.minimum
        
        return x_data, y_data

    
class BatchNormPerSample:
    """ Class used to normalize individually the samples of a batch
    """
    
    def __init__(self) -> None:
        """Class constructor to initialize the object
        
        """
        
        self.name = 'BatchNormPerSample'

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name

    def __call__(self, x_data, y_data=None):
        """Performs the normalization of the values in x_data

        Args:
            x_data (ndarray): n-D matrix
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: tuple with the normalized matrix in the first index
        """
        batch_shape = x_data.shape
        samples_max = x_data.reshape(batch_shape[0], -1).max(axis=1).reshape(batch_shape[0], 1)
        x_data = (x_data.reshape(batch_shape[0], -1) / samples_max).reshape(batch_shape)

        return x_data, y_data

class AxisPermutation:
    """Class used to permute axes of a numpy matrix
    """

    def __init__(self, axis) -> None:
        """Class constructor to initialize the object

        Args:
            axis (tuple): tuple of n elements indicating the axes to be permuted
        """

        self.axis = axis
        self.name = 'AxisPermutation'

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name + str(self.axis)

    def __call__(self, x_data, y_data=None):
        """performs the axis permutation in the x_data

        Args:
            x_data (ndarray): n-D matrix
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: tuple with the permuted n-D matrix in the first index
        """
        
        return np.transpose(x_data, self.axis), y_data

class DataReshape:
    """Class to reshape the data in the batch. The size of the first dimension (axis=0) remains the same
    """

    def __init__(self, new_shape):
        """Class constructor to initialize the object

        Args:
            new_shape (tuple): A n-D typle with the new shape of the data x_data (for axis > 0)
        """
        self.newShape = new_shape
        self.name = 'DataReshape'

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name + str(self.newShape)

    def __call__(self, x_data, y_data=None):
        """Performs the resize of the batch x_data

        Args:
            x_data (ndarray): n-D matrix to be resized
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: Tuple with the resized n-D matrix in first index
        """
        batch_shape = (x_data.shape[0],) + self.newShape

        return np.reshape(x_data, batch_shape), y_data

class DataLog10:
    """Class to transform the data to log10.
    """

    def __init__(self):
        """Class constructor to initialize the object

        Args:
            None
        """
        self.name = 'DataLog10'

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name 

    def __call__(self, x_data, y_data=None):
        """Performs the log10 calculus of the batch x_data

        Args:
            x_data (ndarray): n-D matrix to be resized
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: Tuple with the transformed n-D matrix in first index
        """

        return np.log10(x_data), y_data

class DataChannelsMean:
    """Class to calculate the mean of sample from all channels.
    """

    def __init__(self,axis):
        """Class constructor to initialize the object

        Args:
            None
        """
        self.name = 'DataChannelsMean'
        self.axis = axis

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name 

    def __call__(self, x_data, y_data=None):
        """Performs the log10 calculus of the batch x_data

        Args:
            x_data (ndarray): n-D matrix to be resized
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: Tuple with the transformed n-D matrix in first index
        """

        return np.mean(x_data,axis=self.axis), y_data
    
    
class PSD:
    """Class used to calculate psd over a 2D matrix
    """

    def __init__(self) -> None:
        """Class constructor to initialize the object
        Args:
        """
        self.name = 'PSD feature'

    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name

    def __call__(self, x_data, y_data=None):
        """Perform the filter over the samples of x_data[i, :]

        Args:
            x_data (ndarray): 2D matrix of type float
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: tuple with the psd 2D matrix in the first index
        """
        from scipy import signal
        freqs, psd = signal.welch(x_data)
                
        return psd, y_data

    
class SampleTrimmer:
    """Class to trim the leading and/or trailing samples from a 2D array.
    """

    def __init__(self, axis, front, back):
        """Class constructor to initialize the object

        Args:
            axis (int): the axis index to trim the data
            from (int): amount of samples to be removed from the front of the array
            back (int): amount of samples to be removed from the back of the array.
        """
        self.axis = axis
        self.front = front
        self.back = back
        self.name = 'SampleTrimmer'


    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name + '(' + str(self.axis) + ',' + str(self.front) + ',' + str(self.back) + ')'

    def __call__(self, x_data, y_data=None):
        """Performs the trimming of the batch x_data

        Args:
            x_data (ndarray): n-D matrix to be trimmed
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: Tuple with the trimmed n-D matrix in first index
        """
        if(self.axis == 0):
            if self.front + self.back < x_data.shape[0]:
                x_data = x_data[self.front:-self.back, :]
            else:
                print('SampleTrimmer: can not trim data. shape=', x_data.shape)
        
        elif(self.axis == 1):
            if self.front + self.back < x_data.shape[1]:
                x_data = x_data[:, self.front:-self.back]
            else:
                print('SamplesTrimmer: can not trim data. shape=', x_data.shape)
        else:
            print('SampleTrimmer: Axis', self.axis, 'not found.')

        return x_data, y_data


class DataRMS:
    """ Class used to normalize individually the samples of a batch
    """
    
    def __init__(self) -> None:
        """Class constructor to initialize the object
        
        """
        
        self.name = 'DataRMS'
    def getName(self):
        """Retrieve the name of the preprocessing
        Returns:
            str: The string with the name
        """
        return self.name
    
    def __call__(self, x_data, y_data=None):
        """Performs the normalization of the values in x_data
        Args:
            x_data (ndarray): n-D matrix
            y_data (any): y_data. Defaults to None.
        Returns:
            tuple[ndarray, any]: tuple with the normalized matrix in the first index
        """

        return np.sqrt(np.mean(np.power(x_data, 2),axis=-1,keepdims=True)), y_data   
    
class LabelOneHot:
    """ Class used to none hot encode of the values in y_data
    """
    
    def __init__(self, labels):
        """Class constructor to initialize the object
        
        """
        self.lb = np.array(labels)
        # label,idx = np.unique(labels,return_index=True)
        # self.lb={label[k]:idx[k] for k in np.arange(len(idx))}
        
        self.name = 'LabelOneHot'
        
    def getName(self):
        """Retrieve the name of the preprocessing
        Returns:
            str: The string with the name
        """
        return self.name
    
    def __call__(self, x_data, y_data):
        """Performs the one hot encode of the values in y_data
        Args:
            x_data (ndarray): n-D matrix. Default to None
            y_data (any): y_data. 
            labels (array): all labels present in dataset
        Returns:
            tuple[ndarray, any]: tuple with the encoded label
        """
        
        y_oh = np.zeros((y_data.shape[0],len(self.lb)),np.float32)
        # print(y_oh.shape, self.lb, y_data.shape, y_data[0])
        for i in np.arange(y_data.shape[0]):
            y_oh[i,y_data[i]] = 1.
                        
        return x_data, y_oh
    
class LowPassFilter:
    """ Class used to filter data
    """
    
    def __init__(self, freq, fs, order):
        """Class constructor to initialize the object
        
        """
        self.freq = freq
#         self.fs = fs
#         self.order = order
        
        self.normal_cutoff = freq / (fs / 2)
        
        # Get the filter coefficients 
        self.b, self.a = signal.butter(order, self.normal_cutoff, btype='low', analog=False)
        
        
        self.name = 'LowPassFilter'
        
    def getName(self):
        """Retrieve the name of the preprocessing
        Returns:
            str: The string with the name
        """
        return self.name  + str(self.freq)
    
    def butter_lowpass_filtfilt(self, data):

        y = signal.filtfilt(self.b, self.a, data)

        return y

    def __call__(self, x_data, y_data):
        """Performs the one hot encode of the values in y_data
        Args:
            x_data (ndarray): n-D matrix. Default to None
            y_data (any): y_data. 
            labels (array): all labels present in dataset
        Returns:
            tuple[ndarray, any]: tuple with the encoded label
        """
        # print('filter x data', x_data.shape)
        # batchsize = x_data.shape[0]
        nchannels = x_data.shape[0]
        # for nb in np.arange(batchsize):
        for nch in np.arange(nchannels):
            x_data[nch,:] = self.butter_lowpass_filtfilt(x_data[nch,:])
                        
        return x_data, y_data

class HighPassFilter:
    """ Class used to filter data
    """
    
    def __init__(self, freq, fs, order):
        """Class constructor to initialize the object
        
        """
        self.freq = freq
#         self.fs = fs
#         self.order = order
        
        self.normal_cutoff = freq / (fs / 2)
        
        # Get the filter coefficients 
        self.b, self.a = signal.butter(order, self.normal_cutoff, btype='high', analog=False)
        
        
        self.name = 'HighPassFilter'
        
    def getName(self):
        """Retrieve the name of the preprocessing
        Returns:
            str: The string with the name
        """
        return self.name  + str(self.freq)
    
    def butter_highpass_filtfilt(self, data):

        y = signal.filtfilt(self.b, self.a, data)

        return y

    def __call__(self, x_data, y_data):
        """Performs the one hot encode of the values in y_data
        Args:
            x_data (ndarray): n-D matrix. Default to None
            y_data (any): y_data. 
            labels (array): all labels present in dataset
        Returns:
            tuple[ndarray, any]: tuple with the encoded label
        """
        
        x_data = self.butter_highpass_filtfilt(x_data)
                        
        return x_data, y_data


class BandStopFilter:
    """ Class used to filter data
    """
    
    def __init__(self, lowcut, highcut, fs, order):
        """Class constructor to initialize the object
        
        """
        #self.freq = freq
        self.fs = fs
        self.freq =fs
#         self.order = order
        
        # self.normal_cutoff = freq / (fs / 2)
        
        # Get the filter coefficients 
        self.b, self.a = signal.iirfilter(order, [lowcut, highcut], fs=fs, btype='bandstop')
               
        self.name = 'BandStopFilter'
        
    def getName(self):
        """Retrieve the name of the preprocessing
        Returns:
            str: The string with the name
        """
        return self.name  + str(self.freq)
    
    def butter_bandstop_filtfilt(self, data):

        y = signal.filtfilt(self.b, self.a, data)

        return y

    def __call__(self, x_data, y_data):
        """Performs the one hot encode of the values in y_data
        Args:
            x_data (ndarray): n-D matrix. Default to None
            y_data (any): y_data. 
            labels (array): all labels present in dataset
        Returns:
            tuple[ndarray, any]: tuple with the encoded label
        """
        
        x_data = self.butter_bandstop_filtfilt(x_data)
                        
        return x_data, y_data

class AddNoise:
    """ Class used to add noise channel data
    """
    
    def __init__(self, nchannels):
        """Class constructor to initialize the object
        
        """
        self.max_nchannels = nchannels
        self.name = 'AddNoiseChannel'
        
    def getName(self):
        """Retrieve the name of the preprocessing
        Returns:
            str: The string with the name
        """
        return self.name  + str(self.freq)
    

    def __call__(self, x_data, y_data):
        """Performs the one hot encode of the values in y_data
        Args:
            x_data (ndarray): n-D matrix. Default to None
            y_data (any): y_data. 
            labels (array): all labels present in dataset
        Returns:
            tuple[ndarray, any]: tuple with the encoded label
        """
        
        if np.random.randint(2) == 1:
            n_channels = np.random.randint(0,self.max_nchannels,1)[0]
            idx_channel = np.random.randint(0,64,64)[:n_channels]

            for ns in np.arange(x_data.shape[0]):
                for ich  in idx_channel:
                    max_amp = np.random.randint(5,30,1)[0]/10
                    ss = x_data[ns,ich,:]
                    x_data[ns,ich,:] = prep.create_noise(np.min(ss),np.max(ss)*max_amp,ss.shape[0])

        return x_data, y_data

class MuLawQuantization:
    """ Class used to add noise channel data
    """
    
    def __init__(self, axis, mu):
        """Class constructor to initialize the object
        
        """

        self.name = 'MuLawQuantization'
        self.axis = axis
        self.mu = mu
        
    def getName(self):
        """Retrieve the name of the preprocessing
        Returns:
            str: The string with the name
        """
        return self.name 
    
    def normalization(self, x_data):
        
        samples_min = x_data.min(axis=self.axis,keepdims=True)
        samples_max = x_data.max(axis=self.axis,keepdims=True)
        x_data = (x_data - samples_min) * (2)
        x_data = (x_data / (samples_max - samples_min) ) - 1
        
        return x_data

    def muLaw(self, x_data):
        
        num = np.sign(x_data) * np.log(1+self.mu*np.abs(x_data))
        den = np.log(1+self.mu)                         
        
        return num/den
    
    def __call__(self, x_data, y_data):
        """Performs the one hot encode of the values in y_data
        Args:
            x_data (ndarray): n-D matrix. Default to None
            y_data (any): y_data. 
            labels (array): all labels present in dataset
        Returns:
            tuple[ndarray, any]: tuple with the encoded label
        """
        
        # x_data = self.normalization(x_data)
        x_data = self.muLaw(x_data)

        
        return x_data, y_data    


class CreatePatches(tf.keras.layers.Layer):
    
    
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.namefunc = 'CreatePatches'
    
    def getName(self):
        """Retrieve the name of the preprocessing
        Returns:
            str: The string with the name
        """
        return self.namefunc 
    
    def __call__(self, x_data, y_data):
        batch_size = tf.shape(x_data)[0]
        patches = tf.image.extract_patches(
            images=x_data,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches, y_data


class GetSequences():
    
    def __init__(self):
        self.namefunc = 'GetSequences'
        
        self.emtke = EMGTKE()
        self.lowpass = LowPassFilter(freq=3, fs=2048, order=1)
    
    def getName(self):
        """Retrieve the name of the preprocessing
        Returns:
            str: The string with the name
        """
        return self.namefunc 
    
    def __call__(self, x_data, y_data):
    

        filtered = x_data.copy()

        filtered = np.abs(x_data)
        filtered = self.emtke(filtered)
        filtered = filtered / np.max(filtered,axis=1)            
        filtered = self.lowpass(filtered)

        return self.process_emg_data(emg, filtered), y_data

        
        
        
class Downsample():
    """Class used to downsample data
    """
    
    def __init__(self, original_fs, new_fs):
        """Class constructor to initialize the object
        
        Args:
            original_fs (float): Taxa de amostragem original.
            new_fs (float): Nova taxa de amostragem desejada.
        """
        self.original_fs = original_fs
        self.new_fs = new_fs
        
        self.downsample_factor = int(original_fs / new_fs)
        
        self.name = 'Downsample'
        
    def getName(self):
        """Retrieve the name of the preprocessing
        Returns:
            str: The string with the name
        """
        return self.name + str(int(self.original_fs)) + "_" + str(int(self.new_fs))
    
    def downsample_signal(self, data):
        """Realiza o downsample de um sinal
        Args:
            data (ndarray): Sinal a ser downsampled.
        Returns:
            ndarray: Sinal downsampled.
        """
        downsampled_data = data[:, ::self.downsample_factor]
        return downsampled_data

    def __call__(self, x_data, y_data):
        """Executa o downsample dos dados em x_data
        Args:
            x_data (ndarray): Sinal a ser downsampled.
            y_data (any): y_data. 
        Returns:
            tuple[ndarray, any]: Tupla com o sinal downsampled e y_data não alterado.
        """
        x_data_downsampled = self.downsample_signal(x_data)
        return x_data_downsampled, y_data
    
class expand_dims:
    """Class used to expand_dims of array 
    """

    def __init__(self,axis=0) -> None:
        """Class constructor to initialize the object
        """
        self.name = 'expand_dims'
        self.axis= axis
    def getName(self):
        """Retrieve the name of the preprocessing

        Returns:
            str: The string with the name
        """
        return self.name

    def __call__(self, x_data, y_data=None):
        """Perform operation over the x_data

        Args:
            x_data (ndarray): n-D matrix to be processed
            y_data (any): y_data. Defaults to None.

        Returns:
            tuple[ndarray, any]: tuple with the new dimension of array x_data
        """
        x_data=np.expand_dims(x_data,self.axis)
        return x_data, y_data