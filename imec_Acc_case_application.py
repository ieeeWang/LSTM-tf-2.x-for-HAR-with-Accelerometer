# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:06:06 2020
test a trained model from a UCI dataset on imec 4-subj dataset
@author: lwang
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from utils import split_3Darray
import pandas

#%% read Acc data from excel files
# make sure that all excel files are under the folder named 'files'
file_dir = os.path.join(os.getcwd(), 'files')
subj1, subj2, subj3, subj4= '075', '246', '248', '323'
# choose a subject
subj = subj1

Acti_a = 'Chestpatch_subj_' + subj +'_a.xlsx' 
Acti_b = 'Chestpatch_subj_' + subj +'_b.xlsx' 
Acti_c = 'Chestpatch_subj_' + subj +'_c.xlsx' 
Acti_x = 'Chestpatch_subj_' + subj +'_x.xlsx' 

act_list = [Acti_a,Acti_b,Acti_c,Acti_x]
# act_list = [Acti_a,Acti_b,Acti_c]
L_window = 82 # ~ 2.5 sec sliding window
scale = 68 # np.sqrt(68) #normalize our data to the unit: 'g'
Acc_rawdata_list=[]
Acc_2Drawdata_list=[] # store 2D data of 4 activities,
for i, name in enumerate(act_list):
    print('load subject:',i)
    print(name)    
    filepath = os.path.join(file_dir, name)
    excel_data_df = pandas.read_excel(filepath, sheet_name='accelerometer')
    Acc1 = excel_data_df.to_numpy()
    Acc1_3D = split_3Darray(Acc1, L_window)
    Acc1_3D =Acc1_3D/scale
    Acc_rawdata_list.append(Acc1_3D)
    Acc_2Drawdata_list.append(Acc1)

#%% visualize netAcc of groups: a,b,c,x
netAcc_mean_list=[] # store mean net Acc of 4 activities
netAcc_std_list=[] # store std net Acc of 4 activities
for i in range(len(Acc_2Drawdata_list)):
    Acc2D = Acc_2Drawdata_list[i]
    net_acc = np.sqrt(np.sum(Acc2D**2,axis=1))
    netAcc_mean = np.mean(net_acc)
    netAcc_std = np.std(net_acc)
    netAcc_mean_list.append(netAcc_mean)
    netAcc_std_list.append(netAcc_std)


plt.figure()
plt.subplot(211)
xbar = ['a','b','c','x']
plt.bar(xbar, netAcc_mean_list)
plt.ylabel('mean')
plt.title('net Acc, subj#'+ subj)
plt.subplot(212)
plt.bar(xbar, netAcc_std_list)
plt.ylabel('sd')

#%% visualize one original sample
# choose a activity type, 0:a, 1:b, 2:c, 3:x
acti = 1
# choose the starting point to plot a sliding window
t = 10 # plot starts from t*2.5 sec

acc3axis_sample = Acc_rawdata_list[acti][t]
net_acc = np.sqrt(np.sum(acc3axis_sample**2,axis=1))

plt.figure()
plt.plot(acc3axis_sample[:,0],label='x',linestyle='-')
plt.plot(acc3axis_sample[:,1],label='y',linestyle='-')
plt.plot(acc3axis_sample[:,2],label='z',linestyle='-')
plt.plot(net_acc,label='net Acc',linestyle='--')
plt.legend(loc='best', fontsize='x-large')
plt.grid(True)
plt.title('one sliding window (2.5 s)')

#%% laod trainded model
# Re-load the model with the best validation accuracy
model_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'LSTM_best_trained_model_3Acc_32Hz_90.1.h5'
best_model= keras.models.load_model(os.path.join(model_dir, model_name))

#%% predict our Acc data
ACTIVITIES = {
    0: 'WALK',
    1: 'WALK_UP',
    2: 'WALK_DOWN',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'}

X_test = Acc_rawdata_list[3]
y_pre = best_model.predict(X_test)
y_pre_mean = np.mean(y_pre, 0)

plt.figure()
plt.bar(range(6),y_pre_mean)
plt.xlabel('activities')
plt.ylabel('probability')
plt.title('recoganize Acc data as:'+ ACTIVITIES[np.argmax(y_pre_mean)])

