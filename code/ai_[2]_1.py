import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, fftpack
import scipy
from transforms3d.axangles import axangle2mat
from tqdm import tqdm
from numpy.fft import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import seaborn as sns
from math import atan, sqrt
import random
import warnings
warnings.filterwarnings(action='ignore')

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM,Bidirectional,Dropout
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import backend as K 
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.model_selection import KFold,StratifiedKFold
from numpy.random import seed
import keras

train=pd.read_csv('../data/train_features.csv')
train_labels=pd.read_csv('../data/train_labels.csv')
test=pd.read_csv('../data/test_features.csv')

#mag
#------------------------------------------------------------------------------
train['acc_mag'] = (train['acc_x'] ** 2 + train['acc_y'] ** 2 + train['acc_z'] ** 2) 
test['acc_mag'] = (test['acc_x'] ** 2 + test['acc_y'] ** 2 + test['acc_z'] ** 2)

train['gy_mag'] = (train['gy_x'] ** 2 + train['gy_y'] ** 2 + train['gy_z'] ** 2)
test['gy_mag'] = (test['gy_x'] ** 2 + test['gy_y'] ** 2 + test['gy_z'] ** 2)

train['gy_acc_mag'] = ((train['gy_x'] - train['acc_x']) ** 2 + (train['gy_y'] - train['acc_y']) ** 2 + (train['gy_z'] - train['acc_z']) ** 2)
test['gy_acc_mag'] = ((test['gy_x'] - test['acc_x']) ** 2 + (test['gy_y'] - test['acc_y']) ** 2 + (test['gy_z'] - test['acc_z']) ** 2)

#mul
#------------------------------------------------------------------------------
train['acc_mul'] = (train['acc_x'] * train['acc_y'] * train['acc_z'])
test['acc_mul'] = (test['acc_x'] * test['acc_y'] * test['acc_z'])

train['gy_mul'] = (train['gy_x'] * train['gy_y'] * train['gy_z'])
test['gy_mul'] = (test['gy_x'] * test['gy_y'] * test['gy_z'])

#roll
# #------------------------------------------------------------------------------
# train['acc_roll'] = (train['acc_y'] / train['acc_x'] ** 2 + train['acc_y'] ** 2).apply(lambda x : sqrt(x)).apply(lambda x : atan(x))*180/np.pi
# test['acc_roll'] = (test['acc_y'] / test['acc_x'] ** 2 + test['acc_y'] ** 2).apply(lambda x : sqrt(x)).apply(lambda x : atan(x))*180/np.pi

# train['gy_roll'] = (train['gy_y'] / train['gy_x'] ** 2 + train['gy_y'] ** 2).apply(lambda x : sqrt(x)).apply(lambda x : atan(x))*180/np.pi
# test['gy_roll'] = (test['gy_y'] / test['gy_x'] ** 2 + test['gy_y'] ** 2).apply(lambda x : sqrt(x)).apply(lambda x : atan(x))*180/np.pi
#시간 대비 변화량
#------------------------------------------------------------------------------
def jerk_signal(signal): 
    dt=0.02 
    return np.array([(signal[i + 1] - signal[i]) / dt for i in range(len(signal) - 1)])

train_dt=[]
for i in tqdm(train['id'].unique()):
    temp = train.loc[train['id'] == i]
    for v in train.columns[2:]:
        values = jerk_signal(temp[v].values)
        values = np.insert(values,0,0)
        temp.loc[:,v+'_dt'] = values
    train_dt.append(temp)
    
test_dt=[]
for i in tqdm(test['id'].unique()):
    temp=test.loc[test['id']==i]
    for v in train.columns[2:]:
        values=jerk_signal(temp[v].values)
        values=np.insert(values,0,0)
        temp.loc[:,v+'_dt']=values
    test_dt.append(temp)
    
#푸리에 변환
#------------------------------------------------------------------------------
def fourier_transform_one_signal(t_signal):
    complex_f_signal= fftpack.fft(t_signal)
    amplitude_f_signal=np.abs(complex_f_signal)
    return amplitude_f_signal

train=pd.concat(train_dt)

fft=[]
for i in tqdm(train['id'].unique()):
    temp=train.loc[train['id']==i]
    for i in train.columns[2:8]:
        temp[i]=fourier_transform_one_signal(temp[i].values)
    fft.append(temp)
train=pd.concat(fft)

test=pd.concat(test_dt)

fft_t=[]
for i in tqdm(test['id'].unique()):
    temp=test.loc[test['id']==i]
    for i in test.columns[2:8]:
        temp[i]=fourier_transform_one_signal(temp[i].values)
    fft_t.append(temp)
test=pd.concat(fft_t)

#Normalization
#------------------------------------------------------------------------------
col=train.columns
train_s=train.copy()
test_s=test.copy()

scaler = StandardScaler()

train_s.iloc[:,2:]= scaler.fit_transform(train_s.iloc[:,2:])
train_sc = pd.DataFrame(data = train_s,columns =col)

test_s.iloc[:,2:]= scaler.transform(test_s.iloc[:,2:])
test_sc = pd.DataFrame(data = test_s,columns =col)

#Model
#------------------------------------------------------------------------------
X=np.array(train_sc.iloc[:,2:]).reshape(3125, 600, -1)
test_x=np.array(test_sc.iloc[:,2:]).reshape(782, 600, -1)

# y = train_labels['label'].values
y = tf.keras.utils.to_categorical(train_labels['label']) 

def cnn_model(input_shape, classes):
    # seed(2021)
    # tf.random.set_seed(2021)
    
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

# skf = StratifiedKFold(n_splits = 10, random_state = 2021, shuffle = True)
skf = StratifiedKFold(n_splits = 10, shuffle = True)
reLR = ReduceLROnPlateau(patience = 4,verbose = 1,factor = 0.5) 
es =EarlyStopping(monitor='val_loss', patience=8, mode='min')

accuracy = []
losss=[]
models=[]

for i, (train, validation) in enumerate(skf.split(X, y.argmax(1))) :
    mc = ModelCheckpoint(f'../model_kf/cv_study{i + 1}.h5',save_best_only=True, verbose=0, monitor = 'val_loss', mode = 'min', save_weights_only=True)
    print("-" * 20 +"Fold_"+str(i+1)+ "-" * 20)
    model = cnn_model((X.shape[1], X.shape[2]),61)
    history = model.fit(X[train], y[train], epochs = 100, validation_data= (X[validation], y[validation]), 
                        verbose=1,batch_size=64,callbacks=[es,mc,reLR])
    model.load_weights(f'../model_kf/cv_study{i + 1}.h5')
    
    k_accuracy = '%.4f' % (model.evaluate(X[validation], y[validation])[1])
    k_loss = '%.4f' % (model.evaluate(X[validation], y[validation])[0])
    
    accuracy.append(k_accuracy)
    losss.append(k_loss)
    models.append(model)

print('\nK-fold cross validation Auc: {}'.format(accuracy))
print('\nK-fold cross validation loss: {}'.format(losss))

#성능 확인
#------------------------------------------------------------------------------
print(sum([float(i) for i in accuracy])/10)
print(sum([float(i) for i in losss])/10)

#결과 출력
#------------------------------------------------------------------------------
test_X=np.array(test_sc.iloc[:,2:]).reshape(782, 600, -1)
preds = []
for model in models:
    pred = model.predict(test_X)
    preds.append(pred)
pred = np.mean(preds, axis=0)

submission=pd.read_csv('../data/sample_submission.csv')
submission.iloc[:,1:]=pred
submission.to_csv('../result/sub_kfold_stratified_10_adam_fft_0.5.csv',index=False)