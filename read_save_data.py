# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:07:37 2020

@author: lwang
"""

import pickle
from prepare_data import load_data

#%% previously saved dict data
X_train, X_test, Y_train, Y_test = load_data()

#%% save dataset into a dic for later use
Acc_6class_UCI={
    'X_train': X_train,
    'X_test': X_test,
    'Y_train':Y_train,
    'Y_test':Y_test,
    }
a_file = open("Acc_6class_UCI.pkl", "wb")
pickle.dump(Acc_6class_UCI, a_file)
a_file.close()


#%% read saved UCI Acceleromter (Acc) data
# a_file = open("Acc_6class_UCI.pkl", "rb")
# data = pickle.load(a_file)
# a_file.close()

