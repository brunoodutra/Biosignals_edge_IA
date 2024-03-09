aimport numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
import tensorflow as tf
import time
from tqdm import tqdm

class FeaturesDataGenerator:

    def __init__(self, dataset_generator, axis=2, processing=None):
        """
        Args:
            Features dataset_generator: The dataset generator providing input and output data.
            axis (int): Axis for feature computation.
            processing: Optional data processing function.
        """
        self.datasetGenerator = dataset_generator
        self.inputShape = dataset_generator.__getitem__(0)[0].shape
        self.outputShape = dataset_generator.__getitem__(0)[1].shape
        self.processing = processing
        self.axis = axis

    def __len__(self):
        return len(self.datasetGenerator)

    def comput_features(self, emg_data):
        """
        Args:
            emg_data (numpy.ndarray): Input EMG data.

        Returns:
            numpy.ndarray: Feature matrix computed from the input data.
        """
        # Compute MAV (Mean Absolute Value)
        mav = np.mean(np.abs(emg_data), axis=self.axis)

        # Compute RMS (Root Mean Square)
        rms = np.sqrt(np.mean(emg_data**2, axis=self.axis))

        # Compute WL (Waveform Length)
        wl = np.sum(np.abs(np.diff(emg_data, axis=self.axis)), axis=self.axis)

        # Compute ZC (Zero Crossing)
        zc = np.sum(np.abs(np.diff(np.sign(emg_data), axis=self.axis)), axis=self.axis) / 2

        # Compute VAR (Variance)
        var = np.var(emg_data, axis=self.axis)

        # Compute ARV (Average Rectified Value)
        arv = np.mean(np.abs(emg_data), axis=self.axis)

        # Integrated EMG (IEMG)
        iemg = np.sum(np.abs(emg_data), axis=self.axis)

        # Kurtosis
        kurt = kurtosis(emg_data, axis=self.axis)

        # Skewness
        skewness = skew(emg_data, axis=self.axis)

        # Simple Square Integral (SSI)
        ssi = np.sum(emg_data**2, axis=self.axis)

    
        features = np.hstack((mav[:, np.newaxis], rms[:, np.newaxis], wl[:, np.newaxis], zc[:, np.newaxis], var[:, np.newaxis], arv[:, np.newaxis],
                              iemg[:, np.newaxis], kurt[:, np.newaxis], skewness[:, np.newaxis], ssi[:, np.newaxis]))

        self.features_name = ['mav', 'rms', 'wl', 'zc', 'var', 'arv', 'iemg', 'kurt', 'skewness', 'ssi']

        if self.processing is not None:
            features, _ = self.processing(features, features)

        return features'rms': np.sqrt(np.mean(emg_data**2, axis=self.axis),
            'wl': np.sum(np.abs(np.diff(emg_data, axis=self.axis)), axis=self.axis),
            'zc': np.sum(np.abs(np.diff(np.sign(emg_data), axis=self.axis)), axis=self.axis) / 2,
            'var': np.var(emg_data, axis=self.axis),
            'arv': np.mean(np.abs(emg_data), axis=self.axis),
            'iemg': np.sum(np.abs(emg_data), axis=self.axis),
            'kurt': kurtosis(emg_data, axis=self.axis),
            'skewness': skew(emg_data, axis=self.axis),
            'ssi': np.sum(emg_data**2, axis=self.axis)

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
