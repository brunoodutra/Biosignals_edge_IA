import numpy as np
from scipy import signal

def remove_N_seconds(data, Fs, second):
    Ts=1/Fs
    #samples in 2 seconds
    seconds_samples= int(second/Ts)

    data= data[:,seconds_samples:-seconds_samples]
    
    return data 

def check_value(v, mean_rms):
    """Check if value is greater than a given threshold
        Args:
            v (float): EMG value
            mean_rms (float): Mean RMS value used as threshold
        Returns:
    """
    if(v > mean_rms):
        return v
    return 0

def window_rms(a, window_size):
    """Calculate the RMS window of a signal
        Args:
            a (list[float]): EMG signal
            window_size (int): RMS window size  
        Returns:
            rms (list[float]): RMS values of a given signal
    """
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))

def removing_noise(data, windows_size):
    """Remove signal rest noise, this function filters the signal per channel
        Args:
            data (list[float]): EMG signal 
            windows_size (int): RMS Window size
        Returns:
            emg_clean (list[float]): Rest Noise removed EMG signal
    """
    emg_rms = data.copy()

    for ch in np.arange(data.shape[0]):
        rms = window_rms(data[ch,:], windows_size)
        mean_rms = np.mean(rms)
        rms_value = [check_value(v, mean_rms) for v in rms]
        rms = np.concatenate([np.zeros(windows_size-1,dtype=float),np.array(rms_value)])
        emg_rms[ch,:] = rms>0

    return np.argwhere(np.sum(emg_rms,axis=0)>data.shape[0]//2).flatten()

def get_data_resample(x, factor):
    """
    resample data with a factor. When factor >0 is downsample otherside is upsample
    
    :x: input sequency (Array of EMG or Force)
    :return: constant that indicate a upsample or dowsample

    The function recieves a raw array of EMG signals or Force signals and resample the data with considering a factor.
    """
    signal_length=np.max(x.shape)
    ranges_resample=np.arange(0,signal_length,round(factor,5)).astype(int)
   
    if np.argmax(x.shape) == 1:
        data_resampled= x[:,ranges_resample]
    else:
        data_resampled= x[ranges_resample,:]
        
    return data_resampled

def get_emg_from_index(x, index_range, remove_outliers=True):
    """
    Slicing EMG data to valid index only
    Args:
        x (array): input sequency (Array of EMG)
        index_range (list): List of valid indexes
    Returns:
         emg_RMS (array): Sliced RMS-filtered EMG Array

    The function recieves a raw array of EMG signals, remove outliers with a RMS-based filter and the slices the data.
    """
    emg = []
    
    for d in x:
        if(remove_outliers):
            e = np.array(remove_emg_outliers(d))
        else:
            e = np.array(d)
        emg.append(e)
    emg = np.array(emg)
    return emg[:, index_range[0]:index_range[-1]]

def get_force_from_index(x, index_range, remove_outliers=True):
    """
    Slicing Force data to valid index only
    :x: input sequency (Array of Force)
    :index_range: List of valid indexes
    :return: Sliced RMS-filtered Force Array

    The function recieves a raw array of Force signals, remove outliers with a RMS-based filter and the slices the data.
    """
    force = []
    
    for d in x:
        if(remove_outliers):
            e = np.array(remove_emg_outliers(d))
        else:
            e = np.array(d)
        force.append(e)
    force = np.array(force)
    return force[:,index_range[0]:index_range[-1]]
   
def remove_emg_outliers(emg):
    """Remove outliers datapoints from EMG signal data
    Args:
        emg(array float): input sequence (single EMG signal)
    Returns
        emg(array float): Clean EMG signal array
    """
    
    std = np.std(emg)
    mean = np.mean(emg)
    emg_clean = list(map(lambda x: x if (mean - 2*std) < x < (mean + 2*std) else 0, emg)) # Checking for outliers

    cont = 0

    # Filling nan values with the mean of the previous 20 values
    while(cont < len(emg_clean)):                                                         
        if(emg_clean[cont] == 0):
            emg_clean[cont] = np.mean(emg_clean[cont-20:cont])
        cont += 1
    
    return np.array(emg_clean)


def check_kinematics(kinematics):
    """ Check where kinematics are different from resting
    Args:
        kinematics (array float): Kinecamtics data
    Returns:
        activation (array int): Array with indices where kinematics are greather than threshold.
    """
    kine_bool1 = kinematics.copy()
    for ch in np.arange(kinematics.shape[1]):
        kine_bool1[:,ch] = kinematics[:,ch]-(kinematics[0,ch]) 
        
    kine = np.sum(np.abs(kine_bool1),axis=1)
    
    return np.argwhere(kine > (0.2*(kinematics.shape[1])))


def resample_signal(data, orig_fs, new_fs, final_num_points, resampling_type):
    """
    Resample signals given current and desired sampling frequencies 
    Args:
        data: channels x samples 2D array
        orig_fs: sampling frequency of the data
        new_fs: sampling frequency to achieve
        final_num_points: how many points the resampled array should have
    Returns
        resampled data: resampled data to the new sampling frequency
    """
    
    if resampling_type == 'poly': 
        # resample using polynomial interpolation
        if orig_fs > new_fs:
            resampled_data = signal.resample_poly(data, 1, (orig_fs//new_fs), axis=0, padtype='line', window=29)
        else:
            # add evenly spaced samples 
            resampled_data = signal.resample_poly(data, (new_fs//orig_fs), 1, axis=0, padtype='line', window=29)
    elif resampling_type == 'repeat':
        # resample by repeating data
        # works fine with glove data 
        resampled_data = np.repeat(data, (new_fs//orig_fs),axis=0)
    elif resampling_type == 'fourier':
        # resample from scipy uses fft for interpolation
        if orig_fs > new_fs:
            resampled_data = signal.resample(data, data.shape[0]//(orig_fs//new_fs))
        else:
            t = np.linspace(0, data.shape[-1]/orig_fs, data.shape[-1], endpoint=True)
            resampled_data = signal.resample(data, num=int(new_fs/orig_fs)*data.shape[-1], t=t, axis=0)[0]

    return resampled_data[:final_num_points+1,:]

def emg_tkeo(emg):
    """
    Calculates the Teager–Kaiser Energy operator.
    Args:
        emg (array): raw EMG signal.
    Returns
        tkeo (1D array_like): signal processed by the Teager–Kaiser Energy operator.
        
    Notes
    -----
    *Authors*
    - Marcos Duarte
    *See Also*
    See this notebook [1]_.
    References
    ----------
    .. [1] https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
    """

    # Teager–Kaiser Energy operator
    tkeo = emg[:,1:-1]*emg[:,1:-1] - emg[:,:-2]*emg[:,2:]

    return(tkeo)

def MAFilter(data,L):
    """
    Moving average filter using scipy function
    Args:
        data (2D array): 2D data to be filtered on axis 1
        L (int): filter kernel size
    Returns:
        filtered data (2D array): Data filtered by Moving average filter

    """
    
    cs = np.cumsum(data,axis=1,dtype=float)
    cs[:,L:] = cs[:,L:] - cs[:,:-L]
    return cs[:,L-1:]/L 


def scipy_func(data,function):
    """
    functions from scipy: butterworth filter, spectogram and wavelet 
    Args:
        data (1D array): 1D data to be filtered
        function (str): scipy filter to be applied
    Returns:
        filtered data (1D array): Data filtered by function
    
    """
    
    if function == 'butter':   
        powerline = signal.butter(2, [48,52], 'bandstop', fs=2048, output='sos')
        data_filt = signal.sosfilt(powerline, data)
        
    elif function == 'spectrogram':
        f, t, data_filt = signal.spectrogram(data,fs=2048)
    elif function == 'cwt':
        data_filt = signal.cwt(data, signal.ricker, widths=np.arange(1,33))

    return data_filt


