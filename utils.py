import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import itertools
from scipy.signal import resample

ACTIVITIES = {
    0: 'WALK',
    1: 'WALK_UP',
    2: 'WALK_DOWN',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING',
}

def confusion_matrix_pd(Y_true, Y_pred):
    """
    This function prints the confusion matrix
    """
    Y_true = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_true, axis=1)])
    Y_pred = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_pred, axis=1)])
    return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])


def plot_confusion_matrix(conf_mat, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix
    """
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, conf_mat[i, j], horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def downsample_AccSignals(signal, num): #signal (128,3)
    """
    This function resample signal to a new length of num
    The resampled signal starts at the same value as x 
    but is sampled with a spacing of len(x) / num * (spacing of x).
    """
    return resample(signal, num, axis=0)
     

def downsample_batch_Signals(batch_signal, num):
    N, w, ch = batch_signal.shape
    signal_aftersample = np.zeros((N, num, ch))
    for i in range(N):
        signal_aftersample[i]=downsample_AccSignals(batch_signal[i], num)
      
    return signal_aftersample    
        

def split_3Darray(array2d, L_window):    
    """
    This function split 2Darray to 3Darray with non-overlapping sliding window
    width of window = L_window
    """
    N, ch = array2d.shape
    n_windows = N//L_window
    array3d = np.zeros((n_windows, L_window, ch))
    for i in range(n_windows):
        array3d[i]=array2d[i*L_window: (i+1)*L_window,:] 
        
    return array3d
    

def get_netAcc_splitset(X_train):
    n, t, ch = X_train.shape
    netAcc=np.zeros((n,t))
    for i in range(n):
        tmp = X_train[i]
        netAcc[i] = np.sqrt(np.sum(tmp**2,axis=1))
        
    return netAcc
        
def get_netAcc_datalist(Acc_data_list):
    netAcc_data_list=[]
    for i in range(len(Acc_data_list)):
        tmp = Acc_data_list[i] # (n,82,3)
        tmp2 = get_netAcc_splitset(tmp)
        netAcc  = np.expand_dims(tmp2, axis=2)
        netAcc_data_list.append(netAcc)
        
    return netAcc_data_list

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    Acc2D = Acc_2Drawdata_list[i]
    netAcc_list
        