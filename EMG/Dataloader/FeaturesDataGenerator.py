import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
import tensorflow as tf
import time
from tqdm import tqdm

class FeaturesDataGenerator:

    def __init__(self, dataset_generator, axis=2, processing=None, selected_features=None):
        """
        Args:
            Features dataset_generator: The dataset generator providing input and output data.
            axis (int): Axis for feature computation.
            processing: Optional data processing function.
            selected_features (list): List of features to compute. If None, compute all features.
        """
        self.datasetGenerator = dataset_generator
        self.inputShape = dataset_generator.__getitem__(0)[0].shape
        self.outputShape = dataset_generator.__getitem__(0)[1].shape
        self.processing = processing
        self.axis = axis
        self.selected_features = selected_features

    def __len__(self):
        return len(self.datasetGenerator)
    
    def compute_myopulse_percentage_rate(self, emg_data, threshold=0.2):
        # Calculate the Myopulse Percentage Rate here
        # Example:
        myopulse_rate = (np.sum(emg_data > threshold, axis=self.axis) / emg_data.shape[0]) * 100
        return myopulse_rate
    
    def compute_slope_sign_change(self,emg_data, axis=-1):
        # Calculate the slope of the signal
        slope = np.diff(emg_data, axis=axis)

        # Calculate the sign change in the slope
        sign_change = np.diff(np.sign(slope), axis=axis)

        # Count the number of sign changes
        slope_sign_change = np.sum(np.abs(sign_change), axis=axis)

        return slope_sign_change

    # Example of Maximum and Minimum Amplitude in Different Channels calculation
    def compute_amplitude_difference(self,emg_data):
        max_amp = np.max(emg_data, axis=self.axis)  # Calculate the maximum amplitude in each channel
        min_amp = np.min(emg_data, axis=self.axis)  # Calculate the minimum amplitude in each channel
        amp_diff = max_amp - min_amp  # Calculate the amplitude difference in each channel
        return amp_diff

    # Example of Width of Distribution calculation
    def compute_width_distribution(self,emg_data):
        # Assume that the width is the difference between the 25th and 75th percentiles of amplitudes in each channel
        percentile_25 = np.percentile(emg_data, 25, axis=self.axis)
        percentile_75 = np.percentile(emg_data, 75, axis=self.axis)
        width = percentile_75 - percentile_25
        return width

    # Example of Active Electrodes calculation
    def compute_active_electrodes(self,emg_data, threshold=0.2):
        active_electrodes = np.sum(emg_data > threshold, axis=self.axis)  # Count how many values in the channel are greater than a threshold
        return active_electrodes
    
    def autorregressive_coefs(self,emg_data,p=3):
        channels=emg_data.shape(self.axis) 

        phi=np.zeros([emg_data.shape[0],emg_data.shape[1],p])

        ar_coefs=np.zeros([channels,p])

        y_init=[]
        for ch in range(channels):
            y= train_dataset[0][0][0][ch]

            aux=np.zeros(p)
            for k in range(p): 
                aux[p-k:]=y[:k]
                y_init.append(aux.copy())

            phi[ch,:,:]=np.vstack([y[i-p:i] if i-p>=0 else y_init[i]  for i in range(0, len(y))])

            ar_coefs[ch,:]=np.linalg.inv(phi[ch].T.dot(phi[ch])).dot(phi[ch].T.dot(emg_data[ch]))
            
        return ar_coefs
    
    def comput_features(self, emg_data):
        """
        Args:
            emg_data (numpy.ndarray): Input EMG data.

        Returns:
            numpy.ndarray: Feature matrix computed from the input data.
        """
        
        # list with the all features. Bag of features 
        all_features = {
            
            # time domain features
            #Features based on energy information 
            'iemg': np.sum(np.abs(emg_data), axis=self.axis),  # Integrated EMG (IEMG)
            'mav': np.mean(np.abs(emg_data), axis=self.axis),  # MAV (Mean Absolute Value)
            'ssi': np.sum(emg_data**2, axis=self.axis),  # Simple Square Integral (SSI)
            'var': np.var(emg_data, axis=self.axis),  # VAR (Variance)
            'rms': np.sqrt(np.mean(emg_data**2, axis=self.axis)),  # RMS (Root Mean Square)
            'myop': self.compute_myopulse_percentage_rate(emg_data),# Myopulse Percentage Rate
            # first difference of EMG time series
            'wl': np.sum(np.abs(np.diff(emg_data, axis=self.axis)), axis=self.axis),  # WL (Waveform Length)
            'danv': np.mean(np.abs(np.diff(emg_data, axis=self.axis)), axis=self.axis),  # Difference Absolute Mean ValueValue)
            'M2': np.sum(np.diff(emg_data, axis=self.axis)**2, axis=self.axis), # Second-Order Moment
            'dvar': np.var(np.diff(emg_data), axis=self.axis),# Difference Variance
            'dasdv': np.std(np.diff(emg_data), axis=self.axis), # Difference absolute standard deviation value
            'willson_amplitude': np.max(emg_data, axis=self.axis) - np.min(emg_data, axis=self.axis),  # Willson Amplitude
            
            'kurt': kurtosis(emg_data, axis=self.axis),  # Kurtosis
            'skewness': skew(emg_data, axis=self.axis),  # Skewness 
            'zc': np.sum(np.abs(np.diff(np.sign(emg_data), axis=self.axis)), axis=self.axis) / 2,  # ZC (Zero Crossing)
            'slope_sign_change': self.compute_slope_sign_change(emg_data, axis=self.axis),  # Slope Sign Change Coefficients

            'LOG': np.exp(np.mean(np.log(emg_data),axis=self.axis)),
            
                        
            # spatial features          
            'cog': np.mean(emg_data, axis=self.axis),  # Center of Gravity (COG)
            'amplitude_diff': self.compute_amplitude_difference(emg_data), # Amplitude Máxima e Mínima em Canais Diferentes
            'width_distribution': self.compute_width_distribution(emg_data), # Largura da Distribuição
            'auc': np.trapz(emg_data, axis=self.axis), # Area Under the Curve (AUC)
            'active_electrodes': self.compute_active_electrodes(emg_data) # Active electrodes

            }   
                           

        if self.selected_features is None:
            selected_features = all_features.keys()
        else:
            selected_features = self.selected_features

        features = np.hstack([all_features[feature][:, np.newaxis] for feature in selected_features])

        self.features_name = selected_features

        if self.processing is not None:
            features, _ = self.processing(features, features)

        return features
        
    def __getFeaturesName__(self):
        return self.features_name

    def __getitem__(self, i):
        x, y = self.datasetGenerator.__getitem__(i)
        features = self.comput_features(x)
        return features, y

    def __getAllData__(self, verbose=0):
        data = []
        labels = []
        time_to_comput_features = []

        for i in tqdm(range(self.__len__())):
            start_time = time.time()
            items = self.__getitem__(i)
            time_to_comput_features.append(time.time() - start_time)

            data.append(np.vstack(items[0].flatten()))
            labels.append(items[1])

        labels = np.hstack(labels)
        data = np.array(data).squeeze().astype("float32")
        time_to_comput_features = np.array(time_to_comput_features)
        if verbose == 1:
            print('Average time required to extract features ', time_to_comput_features.mean())
        return data, labels
