import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from transforms3d.axangles import axangle2mat

import warnings
warnings.filterwarnings("ignore")

PATH = "../data/"

class Augmentation:
    def __init__(self, data, nPerm = 4, mSL = 10, aug_P=0):
        self.data = data
        self.nPerm = nPerm
        self.mSL = mSL
        self.aug_P = aug_P
    
    def rolling(self):
        for j in np.random.choice(self.data.shape[0], int(self.data.shape[0]*2/3)):
            self.data[j] = np.roll(self.data[j], np.random.choice(self.data.shape[1]), axis=0)
        return self.data
    
    def rotation(self):
        axis = np.random.uniform(low=-1, high = 1, size = self.data.shape[1])
        angle = np.random.unifrom(low=-np.pi, high=np.pi)
        return np.matmul(self.data, axangle2mat(axis, angle))
    
    def shuffle(self):
        data_new = np.zeros(self.data.shape)
        idx = np.random.permutation(self.nPerm)
        continue_while = True
        while continue_while == True:
            segs = np.zeros(self.nPerm+1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(self.mSL, self.data.shape[0]-self.mSL, self.nPerm-1))
            segs[-1] = self.data.shape[0]
            if np.min(segs[-1:]-segs[0:-1]) > self.mSL:
                continue_while = False
        pp = 0
        for ii in range(self.nPerm):
            data_temp = self.data[segs[idx[ii]]:segs[idx[ii]+1], :]
            data_new[pp:pp+len(self.data_temp), :] = data_temp
            pp += len(data_temp)
        return (data_new)
    
    def combine_aug(self, k):
        data_ = self.data.copy()
        if self.aug_P == 0:
            if (k+1) % 2 == 0:
                for i in np.random.choice(int(self.data.shape[0]/600), int(self.data.shape[0]/600*2/3)):
                    data_[600*i:600*(i+1)] = self.rotation(np.array(data_[600*i:600*(i+1)]))
                if (k + 1) % 2 == 1:
                    for i in np.random.choice(int(self.data.shape[0]/600), int(self.data.shape[0]/600*2/3)):
                        data_[600*i:600*(i+1)] = self.permutation(np.array(data_[600*i:600*(i+1)]))
        if self.aug_P != 0:
            pass
        return data_                
        
train = pd.read_csv(PATH + "train_features.csv")
train_acc, train_gy = train.iloc[:, 2:5], train.iloc[:, 5:]
train_time = train.time[:600]/50

train_label = pd.read_csv(PATH + "train_labels.csv")
train_y = train_label

test = pd.read_csv(PATH + "test_features.csv")
submission = pd.read_csv(PATH + "sample_submission.csv")

