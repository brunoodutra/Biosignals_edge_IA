
import tensorflow as tf
import numpy as np
from os.path import isfile, join
from os import listdir
import h5py
import threading
from keras.utils.np_utils import to_categorical

class EEGDataGen(tf.keras.utils.Sequence):
    """Class responsible to retrieve the EMG data from the H5 files during the model training 
    """
    
    def __init__(self, folder_path, idx, batch_size, window_size, window_step, num_channels, sfreq=1000, categorical=True, shuffle=True, processing=None, FilterProcessing=None, fft=False):
        """Class constructor to initialize the object

        Args:
            folder_path (str): folder path of the H5 files
            batch_size (int): size of the batch
            window_size (int): size of the EMG window
            window_step (int): step between EMG windows
            num_channels (int): number of EMG channels
            shuffle (bool, optional): flag to indicate if the data have to be shuffle between epochs. Defaults to True.
            processing (callable, optional): Callable object to process the emg data. Defaults to None.
            FilterProcessing (callable, optional): Callable object to filter process the EEG data. Defaults to None
        """
        self.list_IDs = idx    

        self.windowSize = window_size
        self.windowStep = window_step
        self.numChannels = num_channels
        self.dataProcessing = processing
        self.dataFilterProcessing = FilterProcessing
        
        self.fft = fft
        self.categorical = categorical
        self.folder_path = folder_path

        self.computeSegments(folder_path)

        self.batchSize = batch_size
        self.shuffle = shuffle
        
        self.numSamples = len(self.xData)
        self.indices = np.arange(self.numSamples)
        self.sfreq = sfreq
        
        if self.shuffle == True:
            np.random.shuffle(self.indices)

        self.len = self.__len__()

    def computeSegments(self, folder_path):
        """Use EMG gestures in the H5 files to create EMG segment data with the same window size. 
        Remap the gesture classes to be a set of sequencial values

        Args:
            folder_path (str): folder path with the H5 files
        """
        self.xData, self.ylData, self.ytData = [], [], []
        # all_gestures = []

        h5_file = h5py.File(folder_path, 'r')
        # print('list',self.list_IDs)
        
        if self.list_IDs == []:
            self.list_IDs = np.arange(len(h5_file['label'][...]))
        
        for r in self.list_IDs:
            try:
                segments = self.extract_segments(h5_file['eeg'][r,:,:].shape[-1], self.windowSize, self.windowStep)
                for seg in segments:
                    sample = dict(seg)
                    sample.update({'ID':(int(r))})
                    self.xData.append(sample)

 
            except:
                #repetitions.remove(r)
                continue
                
                
    def extract_segments(self, eeg_signal_length, segment_size, step_size):
        """Compute the (start, end) indices of the EMG segments

        Args:
            emg_signal_length (int): the total number of samples of the EMG
            segment_size (int): the size of the segment (window)
            step_size (int): the step between the segments

        Returns:
            list[dict[int, int]]: dictionary list with the (start, end) indices of all the segments created
        """
        segments = []

        for seg_start in np.arange(0, (eeg_signal_length + 1) - segment_size, step_size):
            segments.append({'start': seg_start, 'end': seg_start + segment_size})
        
        return segments
    
    def on_epoch_end(self):
        """Override the superclass method to shuffle the data on the end of the epoch
        """
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def __getitem__(self, index):
        """Override the superclass method to retrieve the batches during the training

        Args:
            index (int): batch index

        Returns:
            tuple[ndarray, ndarray]: (x[samples, ch], y[class]) data
        """
        
        batch_indices = self.indices[index * self.batchSize:(index + 1) * self.batchSize]
        x_data = np.zeros((self.batchSize, self.windowSize, self.numChannels))
        y_freq = np.zeros((self.batchSize))
        y_task = np.zeros((self.batchSize))

        self.getBatchData(x_data, y_freq, y_task, batch_indices)

        y_freq = tf.convert_to_tensor(y_freq, dtype=tf.float32)
        y_task = tf.convert_to_tensor(y_task, dtype=tf.float32)
        
        if self.categorical == True:
            y_freq = (to_categorical(y_freq,num_classes=5))
            y_task = (to_categorical(y_task,num_classes=2))

        if self.dataFilterProcessing != None:
            x = np.zeros((self.batchSize, self.windowSize, self.numChannels))
            for nch in np.arange(x_data.shape[-1]):
                x[:,:,nch] = self.dataFilterProcessing(x_data[:,:,nch])
            
            x_data=x
            
        if self.fft == True:
            x = np.zeros((self.batchSize, 56, self.numChannels))

            for nch in np.arange(x_data.shape[-1]):
                x[:,:,nch] = tf.math.log(tf.math.abs(tf.signal.rfft(x_data[:,:,nch], fft_length=[self.sfreq//2])[:,2:58]))
            x_data = x

        if(self.dataProcessing != None):
            x_data, y_data = self.dataProcessing(x_data, (y_freq, y_task))
        

        return x_data, (y_freq, y_task)

 
    
    def getBatchData(self, x_data, y_freq,  y_task, batch_indices):
        """Method executed by a thread to create part of the batch data from the segment data 

        Args:
            x_data (ndarray): [i, samples, ch] batch where the EMG data is stored
            batch_indices (ndarray): 1D array with the indices of the EMG segments to be processed
        """
        for i, j in enumerate(batch_indices):
            x_data[i, :, :],  y_freq[i],  y_task[i] = self.getEEGSignal(self.xData[j])
        
            
    def getEEGSignal(self, sample):
        
        with h5py.File(self.folder_path, 'r') as hf:
            
            start, end = sample['start'], sample['end']
            rep = sample['ID']
            eeg_signal = hf['eeg'][rep,:,start:end].T
            
            y1 = hf['label'][rep]
            y2 = hf['ep_flag'][rep]

            
        return eeg_signal, y1, y2

    def __len__(self):
        """Private method to tetrieve the number of batches

        Returns:
            int: number of batches
        """
        num_batches = self.numSamples // self.batchSize
        return num_batches

    def getitem(self, index):
        """Public method to retrieve the batches during the training

        Args:
            index (int): batch index

        Returns:
            tuple[ndarray, ndarray]: (x[samples, ch], y[class]) data
        """
        return self.__getitem__(index)
