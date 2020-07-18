# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:03:00 2020
train a UCI dataset for transfer learning on imec 4-subj dataset
@author: lwang
"""
import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import confusion_matrix_pd, plot_confusion_matrix
from utils import downsample_AccSignals, downsample_batch_Signals

#%% read saved UCI Acceleromter (Acc) data
a_file = open("Acc_6class_UCI.pkl", "rb")
data = pickle.load(a_file)
a_file.close()

X_train, X_test, Y_train, Y_test = data['X_train'], data['X_test'], \
                    data['Y_train'], data['Y_test']

#%% use only tri-axis accelerometer data (last 3 dim)
X_train = X_train[:,:,-3:]
X_test = X_test[:,:,-3:]

#%% plot Acc data
ACTIVITIES = {
    0: 'WALK',
    1: 'WALK_UP',
    2: 'WALK_DOWN',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'}

activity2plot = 0
Y_train_label = np.argmax(Y_train, axis=1)
locs = np.where(Y_train_label==activity2plot)[0]
# choose the first sample to plot
acc3axis_sample = X_train[locs[0]]
net_acc = np.sqrt(np.sum(acc3axis_sample**2,axis=1))
# visualize one original sample
plt.figure()
plt.plot(acc3axis_sample[:,0],label='x',linestyle='-')
plt.plot(acc3axis_sample[:,1],label='y',linestyle='-')
plt.plot(acc3axis_sample[:,2],label='z',linestyle='-')
plt.plot(net_acc,label='net Acc',linestyle='--')
plt.legend(loc='best', fontsize='x-large')
plt.grid(True)
plt.title(ACTIVITIES[activity2plot])

# visualize the sample after downsampling
num_downsample = 82 # 82/32 = 2.5625 sec ~ 2.56 sec used for UCI data
acc3axis_sample2 = downsample_AccSignals(acc3axis_sample, num_downsample)
plt.figure()
plt.plot(acc3axis_sample2[:,0],label='x',linestyle='--')
plt.plot(acc3axis_sample2[:,1],label='y',linestyle='--')
plt.plot(acc3axis_sample2[:,2],label='z',linestyle='--')
plt.legend(loc='best', fontsize='x-large')
plt.grid(True)
plt.title(ACTIVITIES[activity2plot] + ' : downsampling')

#%% downsampling all X data for transfer learning, to match 32 Hz sampling ratio
num_downsample = 82 # 82/32 = 2.5625 sec ~ 2.56 sec used for UCI data 
X_train = downsample_batch_Signals(X_train,num_downsample)
X_test = downsample_batch_Signals(X_test,num_downsample)


#%% define network LSTM
timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = 6
n_hidden = 32

model = models.Sequential()
model.add(layers.LSTM(n_hidden, input_shape=(timesteps, input_dim)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_classes, activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#%% define the training loop for saving the best trained model
def train_network(network, x_training, y_training, x_validation, y_validation, 
                  n_epoch, batch_size, network_save_path):

    # lists where we will be storing values during training
    tra_losses = [] # list for training loss
    tra_accs = [] # list for training accuracy
    val_losses = [] # list for validation loss
    val_accs = [] # list for validation accuracy
    
    # we want to save the parameters that give the best performance on the validation set
    # so we store the best validation accuracy, and save the parameters to disk
    best_validation_accuracy = 0 # best validation accuracy
    
    for epoch in range(n_epoch):  
        print('epoch:',epoch,'/',n_epoch)
        # Train the network
        results = network.fit(x_training, y_training, epochs=1, batch_size = batch_size)
        
        # Get training loss and accuracy
        training_loss = results.history['loss']
        training_accuracy = results.history['accuracy']
        # Add to list
        tra_losses.append(training_loss)
        tra_accs.append(training_accuracy)
        
        # Evaluate performance (loss and accuracy) on validation set
        scores = network.evaluate(x_validation, y_validation, batch_size = batch_size)
        validation_loss = scores[0]
        validation_accuracy = scores[1]
        # Add to list
        val_losses.append(validation_loss)
        val_accs.append(validation_accuracy)
        
        # (Possibly) update best validation accuracy and save the network
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            network.save(model_path)
            print('Saved trained model at %s ' % model_path)

    # for Visualization of the learning curves
    return tra_losses, tra_accs, val_losses, val_accs

    
#%% (option B) training using keras.model.fit API. 
# Simpler but not good for save the best model
# epochs = 100
# history = model.fit(X_train,
#           Y_train,
#           batch_size= 8,
#           validation_data=(X_test, Y_test),
#           epochs=epochs)

#%% training using own-defined training loop
# choose the path for saving the trained models
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'LSTM_best_trained_model.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
model_path = os.path.join(save_dir, model_name)

n_epoch = 20
batch_size= 16 # best < 16
start_time = time.time()

tra_losses, tra_accs, val_losses, val_accs = train_network(model, X_train, 
                     Y_train, X_test, Y_test, n_epoch, batch_size, model_path)

print('elapsed_time:',  time.time() - start_time) # ~517 s for 100 epochs

# visualize the learning curves
plt.figure()
plt.plot(tra_losses,label='loss',linestyle='-')
plt.plot(tra_accs,label='acc',linestyle='-')
plt.plot(val_losses,label='val_loss',linestyle='--')
plt.plot(val_accs,label='val_acc',linestyle='--')
plt.legend(loc='best', fontsize='x-large')
plt.grid(True)

#%% Evaluate the saved 'best model'
# Re-load the model with the best validation accuracy
best_model= keras.models.load_model(model_path)
y_pre = best_model.predict(X_test)

# plt.figure()
# plt.plot(y_pre[:,3],label='pred',linestyle='-')

Y_true = [ACTIVITIES[y] for y in np.argmax(Y_test, axis=1)]
Y_pred = [ACTIVITIES[y] for y in np.argmax(y_pre, axis=1)]

# get overall accuracy
accuracy = accuracy_score(Y_true, Y_pred)
print ('Accuracy: {:.2f}%'.format(100.0*accuracy))
# confusion matrix
conf_mat = confusion_matrix(Y_true, Y_pred)
# visualize the confusion matrix
activityTypes = ['WALKING', 'WALKING_UP', 'WALKING_DOWN', 'SITTING', 'STANDING', 'LAYING']
plt.figure(figsize=(6,6))
plot_confusion_matrix(conf_mat, classes=activityTypes, 
                      title='Confusion matrix, Accuracy: {:.2f}%'.format(100.0*accuracy))




