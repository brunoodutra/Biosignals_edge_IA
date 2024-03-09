import numpy as np
import sys
np.seterr(divide='ignore', invalid='ignore')
from sklearn.cross_decomposition import CCA

class CCA_SSVEP:

    def __init__(self,sfreq=1000,data_length=500,ref_freqs = [12, 8.57, 6.67, 5.45],N_harmonics=2,n_components=1):
        """
        Initialize the CCA_SSVEP classifier.

        Args:
            sfreq (int): Sampling frequency of the EEG data.
            data_length (int): Length of the EEG data.
            ref_freqs (list): List of reference frequencies for the SSVEP classes.
            N_harmonics (int): Number of harmonics to consider for each reference frequency.
            n_components (int): number of components utilized with CCA.
        """
        self.n_components=n_components
        self.n_classes=len(ref_freqs)
        self.set_ref_signals(sfreq,data_length,ref_freqs,N_harmonics)

    def cca_max_correlation(self,X, Y):
        """
        Calculate the maximum canonical correlation between two datasets using CCA.

        Args:
            X (numpy.ndarray): The first dataset.
            Y (numpy.ndarray): The second dataset.

        Returns:
            float: The maximum canonical correlation between X and Y.
        """
        cca = CCA(n_components=self.n_components)  # Specify the number of canonical components to extract
        cca.fit(X, Y)  # Fit CCA model to the data
        
        x_scores,y_scores = cca.transform(X,Y)
        
        max_correlation = np.corrcoef(x_scores.T, y_scores.T)[0, 1]  # Maximum canonical correlation
        
        return max_correlation
    
    def predict(self,eeg_data):
        """
        Predict the SSVEP class label for the given EEG data raw.

        Args:
            eeg_data (numpy.ndarray): EEG data to classify.

        Returns:
            int: Predicted class label.
        """
        ccc=[]
        for i in range(self.n_classes):
            max_P=self.cca_max_correlation(eeg_data, self.ref_signals[:,i*(self.harmonics*2):i*(self.harmonics*2)+self.harmonics])
            ccc+=[max_P]
        y_pred=np.argmax(np.nan_to_num(np.array(ccc))) 
        return y_pred
    
    def get_proba(self,eeg_data):
        """
        Get the SSVEP class label probabilities for the given EEG data raw.

        Args:
            eeg_data (numpy.ndarray): EEG data to classify.

        Returns:
            array: Predicted probabilities of classes.
        """
        ccc=[]
        for i in range(self.n_classes):
            max_P=self.cca_max_correlation(eeg_data, self.ref_signals[:,i*(self.harmonics*2):i*(self.harmonics*2)+self.harmonics])
            ccc+=[max_P]
        y_pred=np.nan_to_num(np.array(ccc))
        return y_pred
    
    def set_ref_signals(self,sfreq=1000,data_length=500,ref_freqs = [12, 8.57, 6.67, 5.45],N_harmonics=2):
        """
        Set the reference signals for the CCA classifier.

        Args:
            sfreq (int): Sampling frequency of the EEG data.
            data_length (int): Length of the EEG data.
            ref_freqs (list): List of reference frequencies for the SSVEP classes.
            N_harmonics (int): Number of harmonics to consider for each reference frequency.
        """
        
        self.harmonics=N_harmonics
        self.ref_freqs=ref_freqs
        t = np.arange(data_length) / sfreq

        ref_signals=[]
        for f in ref_freqs:
                for har in range(1,self.harmonics+1):
                    
                    ref_signals += [np.sin(2 * np.pi * (f*har) * t),
                                    np.cos(2 * np.pi * (f*har) * t)]
        ref_signals=np.vstack(ref_signals).T

        self.ref_signals = ref_signals
    
    def get_ref_signals(self):

        return self.ref_signals
   
    def delta_predict(self,eeg_data,th1, th2):
        """
        CCA SSVEP  predict the delta value for multitask.

        Args:
        - th1: Threshold for the difference between the top two probabilities.
        - th2: Threshold for the maximum probability.
        - eeg_data: Is the probability predition in range 0 to 1 obtained from CCA.get_proba

        Returns:
        - delta_pred: Binary array indicating the delta prediction.
        - y_pred_delta: Modified prediction array based on the delta technique.
        """
        prob_pred = self.get_proba(eeg_data)
        
        if np.diff(np.sort(prob_pred)[-2:]) >= th1 and prob_pred.max() >= th2:
                delta_pred = 1
                y_pred_delta=np.argmax(prob_pred)
        else:
                delta_pred = 0
                y_pred_delta = 0
        return delta_pred, y_pred_delta
        
    def get_delta_pred(self, y_pred, prob_pred,th1, th2):
        """
        CCA SSVEP delta technique.

        Args:
        - th1: Threshold for the difference between the top two probabilities.
        - th2: Threshold for the maximum probability.
        - prob_pred: Is the probability predition in range 0 to 1 obtained from CCA.get_proba
        - y_pred: Predctio classification obtained from CCA.predict

        Returns:
        - delta_pred: Binary array indicating the delta prediction.
        - y_pred_delta: Modified prediction array based on the delta technique.
        """
        delta_pred = np.zeros(y_pred.shape)
        y_pred_delta = y_pred.copy()

        for idx in range(len(y_pred)):
            if np.diff(np.sort(prob_pred[idx])[-2:]) >= th1 and prob_pred[idx].max() >= th2:
                delta_pred[idx] = 1
            else:
                delta_pred[idx] = 0
                y_pred_delta[idx] = 0

        return delta_pred, y_pred_delta