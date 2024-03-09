
import tensorflow as tf
import numpy as np
from os.path import isfile, join
from os import listdir
import h5py
import threading

class EMGGestureDataGen(tf.keras.utils.Sequence):
    """Class responsible to retrieve the EMG data from the H5 files during the model training 
    """
    
    def __init__(self, folder_path, batch_size, window_size, window_step, num_channels, shuffle=True, processing=None, armband=False, dataset_name=None, model_output_mode='multiclass',subjects='All'):
        """Class constructor to initialize the object

        Args:
            folder_path (str): folder path of the H5 files
            batch_size (int): size of the batch
            window_size (int): size of the EMG window
            window_step (int): step between EMG windows
            num_channels (int): number of EMG channels
            shuffle (bool, optional): flag to indicate if the data have to be shuffle between epochs. Defaults to True.
            processing (callable, optional): Callable object to process the emg data. Defaults to None.
            armband (boolean, optional): Convert EMG data to armband matrix . Defaults to False.
            dataset_name (str, optional): EMG dataset name. Defaults to None.
        """
        self.windowSize = window_size
        self.windowStep = window_step
        self.numChannels = num_channels
        self.dataProcessing = processing
        self.model_output_mode = model_output_mode
        self.dataset_name = dataset_name

        self.computeSegments(folder_path, subjects)

        self.batchSize = batch_size
        self.shuffle = shuffle
        
        self.numSamples = len(self.xData)
        self.indices = np.arange(self.numSamples)
        
        if self.shuffle == True:
            np.random.shuffle(self.indices)

        self.armband = armband

        self.len = self.__len__()
        

    def computeSegments(self, folder_path, subjects):
        """Use EMG gestures in the H5 files to create EMG segment data with the same window size. 
        Remap the gesture classes to be a set of sequencial values

        Args:
            folder_path (str): folder path with the H5 files
        """
        if subjects == 'All':
            files = [f for f in listdir(folder_path) if (isfile(join(folder_path, f)) and ".hdf5" in f)]
        else:
            files = [f for f in listdir(folder_path) if (isfile(join(folder_path, f)) and ".hdf5" in f and subjects in f)]
            
        self.xData, self.yData = [], []
        all_gestures = []

        for f in files:
            h5_file = h5py.File(join(folder_path, f), 'r')

            for s in h5_file['sessions'][:]:
                session = h5_file['session_' + str(int(s))]

                all_gestures = set().union(all_gestures, session['gestures'][:])
 
                for g in session['gestures'][:]:
                    repetitions = session['gesture_' + str(int(g))]['repetitions'][:]
                    gesture = session['gesture_' + str(int(g))]
                    if self.model_output_mode == 'multilabel':
                        labels = [session['gesture_' + str(int(g))]['labels'][:].tolist()]


                    for r in repetitions:
                        try:
                            segments = self.extract_segments(gesture['repetition_' + str(int(r))]['emg'].shape[1], self.windowSize, self.windowStep)
                            for seg in segments:
                                sample = dict(seg)
                                sample.update({'s': str(int(s)), 'g': str(int(g)), 'r': str(int(r)), 'emg': gesture['repetition_' + str(int(r))]['emg']})
                                self.xData.append(sample)

                            if self.model_output_mode == 'multilabel':
                                self.yData += labels * len(segments)
                            elif self.model_output_mode == 'multiclass':
                                self.yData += [int(g)] * len(segments)
                        except:
                            continue
        
        # print('pre map',self.yData)
        if self.model_output_mode == 'multiclass':
            all_gestures = list(all_gestures)
            all_gestures = np.sort(all_gestures)


            self.GestureRemap = {g: new_g for new_g, g in enumerate(all_gestures)}
            self.yData = np.array([self.GestureRemap[g] for g in self.yData])


        self.yData = np.array(self.yData)
        # print('mapped',self.yData)
        # print('emg', np.min(self.xData), np.max(self.xData))

    def extract_segments(self, emg_signal_length, segment_size, step_size):
        """Compute the (start, end) indices of the EMG segments

        Args:
            emg_signal_length (int): the totanl number of samples of the EMG
            segment_size (int): the size of the segment (window)
            step_size (int): the step between the segments

        Returns:
            list[dict[int, int]]: dictionary list with the (start, end) indices of all the segments created
        """
        segments = []

        for seg_start in np.arange(0, (emg_signal_length + 1) - segment_size, step_size):
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
        x_data = np.zeros((self.batchSize, self.numChannels, self.windowSize), dtype=np.float32)

        self.getBatchData(x_data, batch_indices)
        
        if self.armband == True:
            idx = self.getArmbandData()
            x_data = np.delete(x_data,idx,axis=1)

        y_data = self.yData[batch_indices]
        
        if self.model_output_mode == 'multilabel':
            rest_label = ((y_data.sum(axis=1) == 0) * 1)
            y_data = np.concatenate((np.expand_dims(rest_label, axis=1), y_data), axis=1)

        # print('emg pré preprocess', np.min(x_data), np.max(x_data))

        if(self.dataProcessing != None):
            x_data, y_data = self.dataProcessing(x_data, y_data)

        # print('emg  preprocess', np.min(x_data), np.max(x_data))

        return x_data, y_data
    
    def getArmbandData(self):
        """Read data from electrode positions selected for armband.

        """
        if self.dataset_name == 'Hyser':
            """Hyser matrix contains Flexor+Extensor data. Also, the channels increase perpendicular
            the muscle direction.
            """ 
            #idx = [np.arange(8), np.arange(56,88), np.arange(104,128)]
            idx = [np.arange(24), np.arange(40,72), np.arange(120,128)]
            idx = np.hstack(idx)

        elif self.dataset_name == '65-HD-sEMG' or  self.dataset_name == '65HDsEMG':
            """65-HD-sEMG matrix contains Extensor+Flexor data. Also, the channels increase at same
            direction to the muscle direction.
            """ 
            idx = [np.arange(0,121,8), np.arange(1,58,8),np.arange(2,59,8), np.arange(5,62,8), np.arange(6,63,8), np.arange(7,129,8)]
            idx = np.hstack(idx)
        else:
            raise Exception("No valid dataset defined")
            
        return idx


    def getBatchData(self, x_data, batch_indices):
        """Method executed by a thread to create part of the batch data from the segment data 

        Args:
            x_data (ndarray): [i, samples, ch] batch where the EMG data is stored
            batch_indices (ndarray): 1D array with the indices of the EMG segments to be processed
        """
        for i, j in enumerate(batch_indices):
            x_data[i, :] = self.getEMGSignal(self.xData[j]).astype(np.float32)
            
    def getEMGSignal(self, sample):
        
        start, end = sample['start'], sample['end']
        emg_signal = sample['emg'][:, start:end]
        return emg_signal

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
