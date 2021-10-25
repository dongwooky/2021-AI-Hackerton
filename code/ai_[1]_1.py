import pandas as pd
import numpy as np
import scipy
from transforms3d.axangles import axangle2mat
import matplotlib.pyplot as plt
from math import atan, sqrt
from scipy.integrate import cumtrapz
import sklearn
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings("ignore")

np.random.seed(10)

train = pd.read_csv("../data/train_features.csv")
train_acc, train_gy  = train.iloc[:, 2:5], train.iloc[:, 5:]
train_time = train.time[:600]/50

train_label = pd.read_csv("../data/train_labels.csv")
train_y = train_label.label

test = pd.read_csv("../data/test_features.csv")
submission = pd.read_csv("../data/sample_submission.csv")

#data aug
#-----------------------------------------------------------------------------
def rolling(data):
    for j in np.random.choice(data.shape[0], int(data.shape[0]*2/3)):
        data[j] = np.roll(data[j], np.random.choice(data.shape[1]), axis= 0)
    return data

def rotation(data):
    axis = np.random.uniform(low=-1, high=1, size=data.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(data , axangle2mat(axis,angle))

def permutation(data, nPerm=4, mSL=10):
    data_new = np.zeros(data.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(mSL, data.shape[0]-mSL, nPerm-1))
        segs[-1] = data.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > mSL:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        data_temp = data[segs[idx[ii]]:segs[idx[ii]+1],:]
        data_new[pp:pp+len(data_temp),:] = data_temp
        pp += len(data_temp)
    return(data_new)

def combine_aug(data, k, aug_P = 0):
    data_ = data.copy()
    if aug_P == 0:
        if (k+1) % 2 == 0:
            for i in np.random.choice(int(data.shape[0]/600), int(data.shape[0]/600*2/3)):
                data_[600*i:600*(i+1)] = rotation(np.array(data_[600*i:600*(i+1)]))
        if (k+1) % 2 == 1:
            for i in np.random.choice(int(data.shape[0]/600), int(data.shape[0]/600*2/3)):
                data_[600*i:600*(i+1)] = permutation(np.array(data_[600*i:600*(i+1)]))
                
    if aug_P != 0:
        pass
    return data_

#feature
#-----------------------------------------------------------------------------
def get_mag(data):
    return (data.iloc[:, 0]**2) + (data.iloc[:, 1]**2) + (data.iloc[:, 2]**2)

def get_mul(data):
    return data.iloc[:, 0] * data.iloc[:, 1] * data.iloc[:, 2]

def get_roll_pitch(data):
    roll = (data.iloc[:,1]/(data.iloc[:,0]**2 + data.iloc[:,2]**2).apply(lambda x : sqrt(x))).apply(lambda x : atan(x))*180/np.pi
    pitch = (data.iloc[:,0]/(data.iloc[:,1]**2 + data.iloc[:,2]**2).apply(lambda x : sqrt(x))).apply(lambda x : atan(x))*180/np.pi
    return pd.concat([roll, pitch], axis= 1)

def setting(data, data_, case = 0):
    if case == 0:
        for i in range(0, data.shape[0], 600):
            data[i] = data_[i] - data_[i+599]
    else:
        for i in range(0, data.shape[0], 600):
            data[i: i+5] = data_[i: i+5].values - data_[i+594:i+599].values
    return data
        
def get_diff(data, case = 0):
    if case == 0:
        x_dif, y_dif, z_dif = data.iloc[:, 0].diff(), data.iloc[:, 1].diff(), data.iloc[:, 2].diff()
    else:
        x_dif, y_dif, z_dif = data.iloc[:, 0].diff(5), data.iloc[:, 1].diff(5), data.iloc[:, 2].diff(5)
    return pd.concat([setting(x_dif, data.iloc[:, 0], case),
                      setting(y_dif, data.iloc[:, 1], case),
                      setting(z_dif, data.iloc[:, 2], case)], axis= 1)

def get_cumtrapz(acc):
    acc_x, acc_y, acc_z = [], [], []
    ds_x, ds_y, ds_z = [], [], []
    for i in range(int(acc.shape[0]/600)):
        acc_x.append(pd.DataFrame(cumtrapz(acc.iloc[600*i:600*(i+1), 0], train_time, initial=0)))
        acc_y.append(pd.DataFrame(cumtrapz(acc.iloc[600*i:600*(i+1), 1], train_time, initial=0)))
        acc_z.append(pd.DataFrame(cumtrapz(acc.iloc[600*i:600*(i+1), 2], train_time, initial=0)))
        ds_x.append(pd.DataFrame(cumtrapz(cumtrapz(acc.iloc[600*i:600*(i+1), 0], train_time, initial=0), train_time, initial=0)))
        ds_y.append(pd.DataFrame(cumtrapz(cumtrapz(acc.iloc[600*i:600*(i+1), 1], train_time, initial=0), train_time, initial=0)))
        ds_z.append(pd.DataFrame(cumtrapz(cumtrapz(acc.iloc[600*i:600*(i+1), 2], train_time, initial=0), train_time, initial=0)))
    return (pd.concat([pd.concat(acc_x), pd.concat(acc_y), pd.concat(acc_z)], axis = 1).reset_index(drop=True),
           pd.concat([pd.concat(ds_x), pd.concat(ds_y), pd.concat(ds_z)], axis= 1).reset_index(drop = True))

#make dataset
#-----------------------------------------------------------------------------
def train_dataset(acc_data, gy_data, i, aug_P = 0):

    aug_acc = combine_aug(acc_data, i, aug_P)
    aug_gy = combine_aug(gy_data, i, aug_P)
    
    diff_acc = get_diff(aug_acc)
    #diff_acc_5 = get_diff(aug_acc, 1)
    
    roll_pitch_acc = get_roll_pitch(aug_acc)
    mag_acc, mul_acc = get_mag(aug_acc), get_mul(aug_acc)
    mag_mul_acc = pd.concat([mag_acc, mul_acc], axis= 1)
    #accvel, disp = get_cumtrapz(aug_acc)

    diff_gy = get_diff(aug_gy)
    #diff_gy_5 = get_diff(aug_gy, 1)
    mag_gy, mul_gy = get_mag(aug_gy), get_mul(aug_gy)
    mag_mul_gy = pd.concat([mag_gy, mul_gy], axis= 1)

    return pd.concat([aug_acc, diff_acc, roll_pitch_acc, mag_mul_acc,
                     aug_gy, diff_gy, mag_mul_gy], axis= 1)

def test_dataset(acc_data, gy_data):
    
    diff_acc = get_diff(acc_data)
    #diff_acc_5 = get_diff(acc_data, 1)
    
    roll_pitch_acc = get_roll_pitch(acc_data)
    mag_acc, mul_acc = get_mag(acc_data), get_mul(acc_data)
    mag_mul_acc = pd.concat([mag_acc, mul_acc], axis= 1)
    #accvel, disp = get_cumtrapz(acc_data)

    diff_gy = get_diff(gy_data)
    #diff_gy_5 = get_diff(gy_data, 1)
    mag_gy, mul_gy = get_mag(gy_data), get_mul(gy_data)
    mag_mul_gy = pd.concat([mag_gy, mul_gy], axis= 1)

    return pd.concat([acc_data, diff_acc, roll_pitch_acc, mag_mul_acc,
                      gy_data, diff_gy, mag_mul_gy], axis= 1)

#Normalization
#-----------------------------------------------------------------------------
data_for_scaler = test_dataset(train_acc, train_gy) # train data만 사용
scaler = StandardScaler().fit(np.array(data_for_scaler))

data_for_scaler = np.array(data_for_scaler).reshape(-1, 600, data_for_scaler.shape[1])
test_x = test_dataset(test.iloc[:, 2:5], test.iloc[:, 5:])

test_X = scaler.transform(np.array(test_x)).reshape(-1, 600, test_x.shape[1])

#Model
#-----------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def First_model():
    inputs = L.Input(shape = (data_for_scaler.shape[1], data_for_scaler.shape[2]))
    gru1 = L.GRU(256, return_sequences = True, dropout = 0.2)(inputs)
    ap = L.AveragePooling1D()(gru1)
    gru2 = L.GRU(150, return_sequences = True)(ap)
    GAP = L.GlobalAveragePooling1D()(gru2)
    dense = L.Dense(61, activation = "softmax")(GAP)
    return keras.models.Model(inputs, dense)

#training
#-----------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import random

def train_model(model_ = None, epochs = 40, first_rlr = 15, second_rlr = 30, r_seed = 10, aug_P = 0, seed_ = 1):
    # first_rlr : 첫번째로 learning_rate이 감소
    # second_rlr : 두번째로 learning_rate이 감소
    # r_seed : StratifiedKFold seed
    # seed_ : numpy/random seed
    
    result_model = []
    cnt = 0
    array_acc = np.array(train_acc).reshape(-1, 600, 3)
    array_gy = np.array(train_gy).reshape(-1, 600, 3)
    
    random.seed(seed_)
    tf.random.set_seed(21)

    split = StratifiedKFold(n_splits=10, shuffle = True, random_state = r_seed)
    for train_idx, valid_idx in split.split(data_for_scaler, train_y):
        
        train_Y, valid_Y = np.array(pd.get_dummies(train_y))[train_idx], np.array(pd.get_dummies(train_y))[valid_idx]

        valid_ACC, valid_GY = array_acc[valid_idx].reshape(-1, 3), array_gy[valid_idx].reshape(-1, 3)
        valid_x = test_dataset(pd.DataFrame(valid_ACC), pd.DataFrame(valid_GY))
        valid_X = scaler.transform(np.array(valid_x)).reshape(-1, 600, valid_x.shape[1])

        model = model_()
        model.compile(optimizer=keras.optimizers.RMSprop(0.003),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        val_score = 0
        seed_ += 1

        for i in range(epochs):
            
            np.random.seed(seed_*47 + i)
            
            train_ACC, train_GY = array_acc[train_idx].reshape(-1, 3), array_gy[train_idx].reshape(-1, 3)
            train_x = train_dataset(pd.DataFrame(train_ACC), pd.DataFrame(train_GY), i, aug_P)
            train_X = scaler.transform(np.array(train_x)).reshape(-1, 600, valid_x.shape[1])

            train_X_ = train_X.copy()

            train_X_ = rolling(train_X_)

            hist = model.fit(train_X_, train_Y, epochs = 1, validation_data = (valid_X, valid_Y), verbose = 0)

            train_accuracy = hist.history["accuracy"]
            new_val_score = accuracy_score(np.argmax(valid_Y, axis = 1), np.argmax(model.predict(valid_X), axis = 1))
            val_loss = hist.history["val_loss"]

            if i == first_rlr:
                model.compile(optimizer=keras.optimizers.RMSprop(0.003*0.2),
                              loss='categorical_crossentropy', metrics=['accuracy'])

            if i == second_rlr:
                model.compile(optimizer = keras.optimizers.RMSprop(0.003*0.2*0.4),
                             loss='categorical_crossentropy', metrics=['accuracy'])

            print("epoch {} - train_accuracy : {} - validation_loss : {} - validation_accuracy : {}".format(i,
                                                                                                            train_accuracy,
                                                                                                            val_loss,
                                                                                                            new_val_score,
                                                                                                            ))

            if i == 0:
                val_loss_score = val_loss[0]
        
            if val_loss_score >= val_loss[0]:
                val_loss_score = val_loss[0]
                best_model = model
                print("####best_val####")
                    
            if new_val_score >= val_score:
                val_score = new_val_score
                best_model = model
                print("####best_acc####")
        print("####################################################### cycle {} is done".format(cnt))
        result_model.append(best_model)
        cnt+=1
    return result_model


def predict_(model):
    result = []
    for mod in model:
        result.append(mod.predict(test_X))
    predict = np.array(result).mean(axis = 0)
    return predict

def save_model(models, name = '1'):
    cnt = 1
    for model in models:
        model.save(path + "submission/last/weight/" + name + '-{}.h5'.format(cnt))
        cnt +=1
        
first_result = train_model(First_model, r_seed = 47, seed_ = 1)

