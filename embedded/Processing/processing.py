import numpy as np
import sys
np.seterr(divide='ignore', invalid='ignore')
from scipy.stats import kurtosis, skew

class preprocessing:
    def __init__(self, axis=1):
        self.axis = axis
        self.y_pred = []

    def set_axis(self, axis):
        """
        Args:
            axis (int): New axis value.
        """
        self.axis = axis

    def AxisPermutation(self, data):
        """
        Args:
            data (numpy.ndarray): Input data array.

        Returns:
            numpy.ndarray: Permuted data array.
        """
        return np.transpose(data, self.axis)

    def moving_window(self, window_array, step_array, window_step):
        """
        Args:
            window_array (numpy.ndarray): Window array.
            step_array (numpy.ndarray): Step array.
            window_step (int): Window step size.

        Returns:
            numpy.ndarray: Updated window array.
        """
        window_array = np.roll(window_array, -window_step)
        if self.axis == 2:
            window_array[:, :, -window_step:] = step_array
        else:
            window_array[:, -window_step:] = step_array
        return window_array

    def retification(self, data):
        """
        Args:
            data (numpy.ndarray): Input data array.

        Returns:
            numpy.ndarray: Rectified data array.
        """
        return np.abs(data)

    def TKE(self, data):
        """
        Args:
            data (numpy.ndarray): Input data array.

        Returns:
            numpy.ndarray: TKEO (Teager-Kaiser Energy Operator) computed data array.
        """
        if self.axis == 2:
            tkeo = data[:,:,1:-1]*data[:,:,1:-1] - data[:,:,:-2]*data[:,:,2:]
        else:
            tkeo = data[:,1:-1]*data[:,1:-1] - data[:,:-2]*data[:,2:]
        return tkeo

    def MAfilter(self, data, N):
        """
        Args:
            data (numpy.ndarray): Input data array.
            N (int): Window size for moving average.

        Returns:
            numpy.ndarray: MA (Moving Average) filtered data array.
        """
        cs = np.cumsum(data, axis=self.axis, dtype=float)
        if self.axis == 2:
            cs[:,:,N:] = cs[:,:,N:] - cs[:,:,:-N]
            return cs[:,:,N-1:]/N
        else:
            cs[:,N:] = cs[:,N:] - cs[:,:-N]
            return cs[:,N-1:]/N

    def NormMinMax(self, data, new_axis=None):
        """
        Args:
            data (numpy.ndarray): Input data array.
            new_axis (int, optional): New axis value (default: None).

        Returns:
            numpy.ndarray: Normalized data array.
        """
        if new_axis is None:
            new_axis = self.axis
        samples_min = data.min(axis=new_axis, keepdims=True)
        data = (data - samples_min)
        samples_max = data.max(axis=new_axis, keepdims=True) + sys.float_info.epsilon
        data = data / samples_max
        return data
    
    def NormMinMax_2D(self,x_data, new_axis, minimum, maximum):
            
        samples_min = x_data.min(axis=new_axis,keepdims=True) + sys.float_info.epsilon
        samples_max = x_data.max(axis=new_axis,keepdims=True)
        x_data = (x_data - samples_min) * (maximum - minimum)
        x_data = (x_data / (samples_max - samples_min) ) + minimum

        return x_data

    def four_step_preprocessing(self, data):
        """
        Args:
            data (numpy.ndarray): Input data array.

        Returns:
            numpy.ndarray: Preprocessed data array using four steps.
        """
        emg_ret = self.retification(data)
        emg_tke = self.TKE(emg_ret)
        emg_MA = self.MAfilter(emg_tke, N=31)
        emg_norm = self.NormMinMax(emg_MA)
        return emg_norm

    def RMS(self, data, window_size):
        """
        Args:
            data (numpy.ndarray): Input data array.
            window_size (int): Size of the moving window.

        Returns:
            numpy.ndarray: RMS (Root Mean Square) computed data.
        """
        a2 = np.power(data, 2)
        window = np.ones(window_size) / float(window_size)
        return np.sqrt(np.convolve(a2, window, 'valid'))

    def DataReshape(self, data, new_shape):
        """
        Args:
            data (numpy.ndarray): Input data array.
            new_shape (tuple): New shape for data.

        Returns:
            numpy.ndarray: Reshaped data array.
        """
        batch_shape = (data.shape[0],) + new_shape
        return np.reshape(data, batch_shape)

class posprocessing:
    def __init__(self, axis=1):
        self.axis = axis
        self.y_pred = []

    def majority_voting(self, y_model, n_voting):
        """
        Args:
            y_model (list): Model predictions.
            n_voting (int): Number of voting iterations.

        Returns:
            numpy.ndarray: Majority voted prediction.
        """
        if len(self.y_pred) < n_voting:
            self.y_pred += [y_model[-1]]
        else:
            values, count = np.unique(y_model, return_counts=True)
            if len(count) == n_voting:
                self.y_pred += [self.y_pred[-1]]
            else:
                self.y_pred += [values[np.argmax(count)]]

            self.y_pred = self.y_pred[-n_voting:]

        return np.array(self.y_pred[-1])

class FeaturesExtract:
    def __init__(self, axis=1, selected_features=None):
        self.axis = axis
        self.selected_features = selected_features
        
    def compute_myopulse_percentage_rate(self, emg_data, threshold=0.6):
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

    def comput_features(self, emg_data, axis=1):
        
        """
        Args:
            emg_data (numpy.ndarray): Input EMG data.

        Returns:
            numpy.ndarray: Feature matrix computed from the input data.
        """
        self.axis=axis
        # list with the all features. Bag of features 
        all_features = {
            
            # time domain features
            'mav': np.mean(np.abs(emg_data), axis=self.axis),  # MAV (Mean Absolute Value)
            'rms': np.sqrt(np.mean(emg_data**2, axis=self.axis)),  # RMS (Root Mean Square)
            'wl': np.sum(np.abs(np.diff(emg_data, axis=self.axis)), axis=self.axis),  # WL (Waveform Length)
            'zc': np.sum(np.abs(np.diff(np.sign(emg_data), axis=self.axis)), axis=self.axis) / 2,  # ZC (Zero Crossing)
            'var': np.var(emg_data, axis=self.axis),  # VAR (Variance)
            'arv': np.mean(np.abs(emg_data), axis=self.axis),  # ARV (Average Rectified Value)
            'iemg': np.sum(np.abs(emg_data), axis=self.axis),  # Integrated EMG (IEMG)
            'kurt': kurtosis(emg_data, axis=self.axis),  # Kurtosis
            'skewness': skew(emg_data, axis=self.axis),  # Skewness
            'ssi': np.sum(emg_data**2, axis=self.axis),  # Simple Square Integral (SSI)
            'willson_amplitude': np.max(emg_data, axis=self.axis) - np.min(emg_data, axis=self.axis),  # Willson Amplitude
            
            'slope_sign_change': self.compute_slope_sign_change(emg_data, axis=self.axis),  # Slope Sign Change Coefficients
            
            # first difference of EMG time series
            'dasdv': np.std(np.diff(emg_data), axis=self.axis), # Difference absolute standard deviation value
            'dvar': np.var(np.diff(emg_data), axis=self.axis),# Difference Variance
            'danv': np.mean(np.abs(np.diff(emg_data)), axis=self.axis),# Difference Absolute Mean Value
            'M2': np.mean(emg_data ** 2, axis=self.axis), # Second-Order Moment
            'myop': self.compute_myopulse_percentage_rate(emg_data),# Myopulse Percentage Rate
            
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

        return features
