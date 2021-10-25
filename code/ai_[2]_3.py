import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import fftpack
from numpy.fft import *
import warnings
warnings.filterwarnings(action='ignore')

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers import Dense, LSTM,Bidirectional,Dropout
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import backend as K 
from sklearn.model_selection import KFold,StratifiedKFold

#Get CSV File
#-----------------------------------------------------------------------------
train=pd.read_csv('../data/train_features.csv')
train_labels=pd.read_csv('../data/train_labels.csv')
test=pd.read_csv('../data/test_features.csv')

#Feature
#-----------------------------------------------------------------------------
#get energy feature
def energy_feature(data):
    return ((data.iloc[:, 0] ** 2) + (data.iloc[:, 1] ** 2) + (data.iloc[:, 2] ** 2)) ** (1 / 3)

#get gradient feature
def gradient_feature(signal):
    gradient_signal = []
    for i in range(len(signal) - 1):
        gradient_signal.append((signal[i + 1] - signal[i]) / 0.02)
    return np.array(gradient_signal)

#get fourier transform
def fourier_transform(signal):
    fourier_signal = fftpack.fft(signal)
    return np.abs(fourier_signal)

#detect energy feature
#train
train_acc = train.iloc[:, 2:5]
train['acc_energy'] = energy_feature(train_acc)

train_gy = train.iloc[:, 5:8]
train['gy_energy'] = energy_feature(train_gy)

#train_diff = train_gy - train_acc
train_diff_x = train_gy.iloc[:, 0] - train_acc.iloc[:, 0]
train_diff_y = train_gy.iloc[:, 1] - train_acc.iloc[:, 1]
train_diff_z = train_gy.iloc[:, 2] - train_acc.iloc[:, 2]
train_diff = pd.concat([train_diff_x, train_diff_y, train_diff_z], axis=1)
train['diff_energy'] = energy_feature(train_diff)
   
#test
test_acc = test.iloc[:, 2:5]
test['acc_energy'] = energy_feature(test_acc)

test_gy = test.iloc[:, 5:8]
test['gy_energy'] = energy_feature(test_gy)

test_diff_x = test_gy.iloc[:, 0] - test_acc.iloc[:, 0]
test_diff_y = test_gy.iloc[:, 1] - test_acc.iloc[:, 1]
test_diff_z = test_gy.iloc[:, 2] - test_acc.iloc[:, 2]
test_diff = pd.concat([test_diff_x, test_diff_y, test_diff_z], axis=1)
test['diff_energy'] = energy_feature(test_diff)
    
#detect gradient feature
#train
train_gf = []
print("--->Train_Gradient_Feature_Detect")
for i in tqdm(train['id'].unique()):
    train_per_id = train.loc[train['id']==i]
    for column in train.columns[2:]:
        series = gradient_feature(train_per_id[column].values)
        series = np.insert(series, 0, 0)
        train_per_id.loc[:, column+'_gf'] = series
    train_gf.append(train_per_id)
train = pd.concat(train_gf)

#test    
test_gf = []
print("--->Test_Gradient_Feature_Detect")
for i in tqdm(test['id'].unique()):
    test_per_id = test.loc[test['id']==i]
    for column in test.columns[2:]:
        series = gradient_feature(test_per_id[column].values)
        series = np.insert(series, 0, 0)
        test_per_id.loc[:, column+'_gf'] = series
    test_gf.append(test_per_id)
test = pd.concat(test_gf)
    
#detect fourier transform
#train
print("--->Train : Fourier Transform")
train_ft = []
for i in tqdm(train['id'].unique()):
    train_per_id = train.loc[train['id']==i]
    for i in train.columns[2:8]:
        train_per_id[i] = fourier_transform(train_per_id[i].values)
    train_ft.append(train_per_id)
train = pd.concat(train_ft)

#test
print("--->Test : Fourier Transform")
test_ft = []
for i in tqdm(test['id'].unique()):
    test_per_id = test.loc[test['id']==i]
    for i in test.columns[2:8]:
        test_per_id[i] = fourier_transform(test_per_id[i].values)
    test_ft.append(test_per_id)
test = pd.concat(test_ft)

#Normalization
#-----------------------------------------------------------------------------
train_copy = train.copy()
test_copy = test.copy()

standardScaler = StandardScaler()
train_copy.iloc[:, 2:] = standardScaler.fit_transform(train_copy.iloc[:, 2:])
train_std = pd.DataFrame(data=train_copy, columns=train.columns)

test_copy.iloc[:, 2:] = standardScaler.fit_transform(test_copy.iloc[:, 2:])
test_std = pd.DataFrame(data=test_copy, columns=test.columns)

#Model
#-----------------------------------------------------------------------------

#trainset
train_x=np.array(train_std.iloc[:,2:]).reshape(3125, 600, -1)
test_x=np.array(test_std.iloc[:,2:]).reshape(782, 600, -1)

train_y = tf.keras.utils.to_categorical(train_labels['label']) 

#define model
def define_model(input_shape, classes):
    
    input_layer = keras.layers.Input(input_shape)
    conv1 = keras.layers.Conv1D(filters=128, kernel_size=9, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)
    conv1 = keras.layers.Dropout(rate=0.3)(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=6, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    conv2 = keras.layers.Dropout(rate=0.4)(conv2)
    
    conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    conv3 = keras.layers.Dropout(rate=0.5)(conv3)
    
    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    
    output_layer = keras.layers.Dense(classes, activation='softmax')(gap)
    
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(), 
        metrics=['accuracy'])
    
    return model

accuracy = []
losses=[]
models=[]
kFold = StratifiedKFold(n_splits = 10, shuffle = True)
for i, (t, val) in enumerate(kFold.split(train_x, train_y.argmax(1))):
    print(str(i+1)+" fold")
    model = define_model((train_x.shape[1], train_x.shape[2]), train_y.shape[1])
    history = model.fit(
        train_x[t], train_y[t],
        epochs = 100,
        validation_data = (train_x[val], train_y[val]),
        verbose = 1,
        batch_size = 64,
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, mode='min'),
            ModelCheckpoint(f'../model/{i + 1}.h5',save_best_only=True, verbose=0, monitor = 'val_loss', mode = 'min', save_weights_only=True),
            ReduceLROnPlateau(patience = 4,verbose = 1,factor = 0.5) 
            ]
        )
    model.load_weights(f'../model/{i + 1}.h5')
    accuracy.append('%.4f' % (model.evaluate(train_x[val], train_y[val])[1]))
    losses.append('%.4f' % (model.evaluate(train_x[val], train_y[val])[0]))
    models.append(model)

print('\nK-fold validation Auc: {0} ---> {1}'.format(accuracy, sum([float(i) for i in accuracy])/10))
print('\nK-fold validation loss: {0} ---> {1}'.format(losses, sum([float(i) for i in losses])/10))

#predict
#-----------------------------------------------------------------------------
test_x=np.array(test_std.iloc[:,2:]).reshape(782, 600, -1)

preds = []
for model in models:
    pred = model.predict(test_x)
    preds.append(pred)
pred = np.mean(preds, axis=0)

submission=pd.read_csv('../data/sample_submission.csv')
submission.iloc[:,1:]=pred
submission.to_csv('result.csv',index=False)