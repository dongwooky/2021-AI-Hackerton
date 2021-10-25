import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

from transforms3d.axangles import axangle2mat
from math import atan, sqrt
from scipy.integrate import cumtrapz

import sklearn
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

PATH = "../data/"

#Data Augmentation 
#-----------------------------------------------------------------------------------------
class Augmentation:
    def __init__(self, perm_idx = 4, least_size = 10, doCombine=0):
        self.perm_idx = perm_idx
        self.least_size = least_size
        self.doCombine = doCombine
    
    #rolling augmentation
    def rolling(self, rolling_data):
        for j in np.random.choice(rolling_data.shape[0], int(rolling_data.shape[0]*2/3)): #전체 데이터의 2/3을 roll
            self.data[j] = np.roll(rolling_data[j], np.random.choice(rolling_data.shape[1]), axis=0)  #x, y, z값을 rolling
        return self.data
    
    #rotation augmentation
    def rotation(self, rotation_data):
        axis = np.random.uniform(low=-1, high = 1, size = rotation_data.shape[1]) #random axis 생성
        angle = np.random.uniform(low=-np.pi, high=np.pi)   #random angle 생성
        return np.matmul(rotation_data, axangle2mat(axis, angle))   #random axis, random angle로 signal 변형
    
    #permutation augmentation
    def permutation(self, permutation_data):
        permutation_data = np.array(permutation_data)
        perm_data = np.zeros(permutation_data.shape)    #permutation 후 들어갈 빈 데이터 공간 생성
        idx = np.random.permutation(self.perm_idx) #나눌 인덱싱 선택 
        continue_while = True
        while continue_while == True:
            #least_size 이상의 차이가 나도록 segemtation할 인덱싱 생성
            segs = np.zeros(self.perm_idx+1, dtype=int) 
            segs[1:-1] = np.sort(np.random.randint(self.least_size, permutation_data.shape[0]-self.least_size, self.perm_idx-1))
            segs[-1] = permutation_data.shape[0]
            if np.min(segs[-1:]-segs[0:-1]) > self.least_size:
                continue_while = False
        #permutation
        start_idx = 0
        for p_idx in range(self.perm_idx):
            data_temp = permutation_data[segs[idx[p_idx]]:segs[idx[p_idx]+1], :]
            perm_data[start_idx:start_idx+len(data_temp), :] = data_temp
            start_idx += len(data_temp)
        return perm_data
    
    #augmentation combine
    def combine_aug(self, comb_data, k):
        combined_data = comb_data.copy()
        if self.doCombine == 0:
            if (k+1) % 2 == 0:
                for i in np.random.choice(int(comb_data.shape[0]/600), int(comb_data.shape[0]/600*2/3)):
                    combined_data[600*i:600*(i+1)] = self.rotation(np.array(combined_data[600*i:600*(i+1)]))
                if (k + 1) % 2 == 1:
                    for i in np.random.choice(int(comb_data.shape[0]/600), int(comb_data.shape[0]/600*2/3)):
                        combined_data[600*i:600*(i+1)] = self.permutation(np.array(combined_data[600*i:600*(i+1)]))
        if self.doCombine != 0:
            pass
        return combined_data                
#Feature
#---------------------------------------------------------------------------------------
class Feature:
    def __init__(self, case = 0):
        self.case = case
        
    def get_mag(self, data):
        return (data.iloc[:, 0]**2) + (data.iloc[:, 1]**2) + (data.iloc[:, 2]**2)
    
    def get_mul(self, data):
        return data.iloc[:, 0] * data.iloc[:, 1] * data.iloc[:, 2]
    
    def get_roll_pitch(self, data):
        roll = (data.iloc[:, 1] / (data.iloc[:, 0]**2 + data.iloc[:, 2]**2).apply(lambda x : sqrt(x))).apply(lambda x : atan(x)) * 180 / np.pi
        pitch = (data.iloc[:, 0]/(data.iloc[:, 1]**2 + data.iloc[:, 2]**2).apply(lambda x : sqrt(x))).apply(lambda x : atan(x)) * 180 / np.pi
        return pd.concat([roll, pitch], axis= 1)
    
    def setting(self, data, data_):
        if self.case == 0:
            for i in range(0, data.shape[0], 600):
                data[i] = data_[i] - data_[i+599]
        else:
            for i in range(0, data.shape[0], 600):
                data[i: i+5] = data_[i: i+5].values - data_[i+594:i+599].values
        return data
    
    def get_diff(self, data):
        if self.case == 0:
            x_dif, y_dif, z_dif = data.iloc[:, 0].diff(), data.iloc[:, 1].diff(), data.iloc[:, 2].diff()
        else:
            x_dif, y_dif, z_dif = data.iloc[:, 0].diff(5), data.iloc[:, 1].diff(5), data.iloc[:, 2].diff(5)
        return pd.concat([self.setting(x_dif, data.iloc[:, 0]),
                          self.setting(y_dif, data.iloc[:, 1]),
                          self.setting(z_dif, data.iloc[:, 2])], axis= 1)
    
    def get_cumtrapz(self, acc):
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
 
#get train, test data
#-----------------------------------------------------------------------------------------
train = pd.read_csv(PATH + "train_features.csv")
train_acc, train_gy = train.iloc[:, 2:5], train.iloc[:, 5:]
train_time = train.time[:600]/50

train_label = pd.read_csv(PATH + "train_labels.csv")
train_y = train_label

test = pd.read_csv(PATH + "test_features.csv")
submission = pd.read_csv(PATH + "sample_submission.csv")

#make augmentation function
#-----------------------------------------------------------------------------------------
np.random.seed(10)

augmentation = Augmentation()
feature = Feature()

#plot train data
#-----------------------------------------------------------------------------------------
# f, axes = plt.subplots(1, 3, sharex=True, sharey=True)

# f.set_size_inches((40, 6))
# f.patch.set_facecolor("white")

# axes[0].plot(train_acc[:600])
# axes[0].set_title("ORIGINAL", fontsize = 20)
# axes[1].plot(augmentation.rotation(train_acc[:600]))
# axes[1].set_title("ROTATION", fontsize = 20)
# axes[2].plot(augmentation.permutation(train_acc[:600]))
# axes[2].set_title("PERMUTATION", fontsize = 20)
# plt.show()

#make Dataset
#-----------------------------------------------------------------------------------------
def train_dataset(acc_data, gy_data, i, aug_P = 0):
    aug_acc = augmentation.combine_aug(acc_data, i)
    aug_gy = augmentation.combine_aug(gy_data, i)
    
    diff_acc = feature.get_diff(aug_acc)
    
    roll_pitch_acc = feature.get_roll_pitch(aug_acc)
    mag_acc, mul_acc = feature.get_mag(aug_acc), feature.get_mul(aug_acc)
    mag_mul_acc = pd.concat([mag_acc, mul_acc], axis = 1)
    
    diff_gy = feature.get_diff(aug_gy)
    mag_gy, mul_gy = feature.get_mag(aug_gy), feature.get_mul(aug_gy)
    mag_mul_gy = pd.concat([mag_gy, mul_gy], axis= 1)
    
    return pd.concat([aug_acc, diff_acc, roll_pitch_acc, mag_mul_acc,
                     aug_gy, diff_gy, mag_mul_gy], axis= 1)

def test_dataset(acc_data, gy_data):
    diff_acc = feature.get_diff(acc_data)
    
    roll_pitch_acc = feature.get_roll_pitch(acc_data)
    mag_acc, mul_acc = feature.get_mag(acc_data), feature.get_mul(acc_data)
    mag_mul_acc = pd.concat([mag_acc, mul_acc], axis = 1)
    
    diff_gy = feature.get_diff(gy_data)
    mag_gy, mul_gy = feature.get_mag(gy_data), feature.get_mul(gy_data)
    mag_mul_gy = pd.concat([mag_gy, mul_gy], axis = 1)
    
    diff_gy = feature.get_diff(gy_data)
    mag_gy, mul_gy = feature.get_mag(gy_data), feature.get_mul(gy_data)
    mag_mul_gy = pd.concat([mag_gy, mul_gy], axis=1)
    
    return pd.concat([acc_data, diff_acc, roll_pitch_acc, mag_mul_acc,
                      gy_data, diff_gy, mag_mul_gy], axis= 1)

#Scaler
#------------------------------------------------------------------------------------------
data_for_scaler = test_dataset(train_acc, train_gy)
scaler = StandardScaler().fit(np.array(data_for_scaler))

data_for_scaler = np.array(data_for_scaler).reshape(-1, 600, data_for_scaler.shape[1])

test_x = test_dataset(test.iloc[: 2:5], test.iloc[:, 5:])
test_x_scaler = scaler.transform(np.array(test_x)).reshape(-1, 600, test_x.shape[1])

#Model
#------------------------------------------------------------------------------------------
