# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:11:22 2020
train a UCI dataset for transfer learning on imec 4-subj dataset
This file is updated from 'LSTM_UCI2012Acc_TransferLearning.py'
but this uses keras build-in API (i.e., tf.keras.callbacks.ModelCheckpoint) for
automated saving model weights (only the best 'metric') after each epoch.
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
print(tf.version.VERSION)

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
def create_model(timesteps, input_dim, n_classes):
    # Define a simple sequential model
    n_hidden = 32
    model = models.Sequential()
    model.add(layers.LSTM(n_hidden, input_shape=(timesteps, input_dim)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_classes, activation='sigmoid'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

#%% Create a basic model instance
timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = 6
# Create a basic model instance
model = create_model(timesteps, input_dim, n_classes) 
model.summary()

#%% (option B) training using keras.model.fit API. 
# choose the path for saving the trained model weights
checkpoint_filepath = os.path.join(os.getcwd(), 'saved_checkpoints/ckpt')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

epochs = 30
start_time = time.time()
history = model.fit(X_train,Y_train,
          batch_size= 8,
          validation_data=(X_test, Y_test),
          epochs=epochs,
          callbacks=[model_checkpoint_callback])
print('elapsed_time:',  time.time() - start_time) # ~517 s for 100 epochs

#%% visualize the learning curves
learning_dict = history.history
tra_losses = learning_dict['loss']
tra_accs = learning_dict['accuracy']
val_losses = learning_dict['val_loss']
val_accs = learning_dict['val_accuracy']

plt.figure()
plt.plot(tra_losses,label='loss',linestyle='-')
plt.plot(tra_accs,label='acc',linestyle='-')
plt.plot(val_losses,label='val_loss',linestyle='--')
plt.plot(val_accs,label='val_acc',linestyle='--')
plt.legend(loc='best', fontsize='x-large')
plt.grid(True)

#%% Evaluate the automated saved 'best model'
# Create a basic model instance
best_model = create_model(timesteps, input_dim, n_classes) 
# Display the model's architecture
best_model.summary()
# Evaluate the model
loss, acc = best_model.evaluate(X_test,  Y_test, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Re-load the model weights from the saved check points
best_model.load_weights(checkpoint_filepath) # Loads the weights
# Evaluate the saved model
loss, acc = best_model.evaluate(X_test,  Y_test, verbose=2)
print("saved model, accuracy: {:5.2f}%".format(100*acc))

#%% plot confusion_matrix on test set
y_pre = best_model.predict(X_test)
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


