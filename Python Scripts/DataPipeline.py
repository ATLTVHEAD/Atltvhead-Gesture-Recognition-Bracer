#   Copyright 2020 Nate Damen
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np 
import pandas as pd 
import datetime
import re
import os, os.path
import time
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf


#combine all data into one file 
def allDatatoDataframe(FolderNavigation,DataFolders):
    files =[]
    completedf = pd.DataFrame(columns=['gesture','acceleration'])
    for idx1,folder in enumerate(DataFolders):
        files = os.listdir(FolderNavigation+folder)
        for idx2,file in enumerate(files):
            df_temp = pd.read_csv(folder+'/'+file)
            x=df_temp[['Acc_X','Acc_Y','Acc_Z']].to_numpy()
            series = pd.Series(data={'gesture': folder, 'acceleration':x.tolist()})
            df_temp2= pd.DataFrame([series])
            completedf=pd.concat([completedf,df_temp2], ignore_index=True)  
    completedf.to_csv('complete_data.csv', index=False)
    return completedf


#split data into training, validation, and testing sets
def splitDataSets(data, training_ratio, validation_ratio, testing_ratio):
    train_set, test_set = train_test_split(data, test_size=1 - training_ratio, random_state=0)
    val_set, test_set = train_test_split(test_set, test_size=test_ratio/
                                     (testing_ratio + validation_ratio), random_state=0) 
    print('len of train_set: '+ str(len(train_set)))
    print('len of test_set: '+ str(len(test_set)))
    print('len of val_set: '+ str(len(val_set)))
    train_set.to_csv('train_set.csv', index=False)
    test_set.to_csv('test_set.csv', index=False)
    val_set.to_csv('val_set.csv', index=False)
    return train_set, val_set, test_set


#Augment the training set of data
def gestureMagnitudeShifting(training,accel_sets, fract):
    magnitudedf=pd.DataFrame(columns=['gesture','acceleration'])
    for idx1, aset in enumerate(accel_sets):
        for molecule, denominator in fract:
            magSeries = pd.Series(data={'gesture': training['gesture'][idx1],
                                    'acceleration':(np.array(aset, dtype=np.float32) * 
                                                    molecule / denominator).tolist()})
            magnitudedf_temp=pd.DataFrame([magSeries])
            magnitudedf=pd.concat([magnitudedf,magnitudedf_temp], ignore_index=True) 
    return magnitudedf


# Time stretch and shrink
def time_wrapping(molecule, denominator, data):
    """Generate (molecule/denominator)x speed data."""
    tmp_data = [[0 for i in range(len(data[0]))] 
                for j in range((int(len(data) / molecule) - 1) * denominator)]
    for i in range(int(len(data) / molecule) - 1):
        for j in range(len(data[i])):
            for k in range(denominator):
                tmp_data[denominator * i +
                         k][j] = (data[molecule * i + k][j] * (denominator - k) +
                                  data[molecule * i + k + 1][j] * k) / denominator
    return tmp_data

def gestureStretchShrink(train_set, accel_sets, fract):
    timedf=pd.DataFrame(columns=['gesture','acceleration'])
    for idx1, aset in enumerate(accel_sets):
        shiftedAccels =[]
        for molecule, denominator in fract:
            shiftedAccels=time_wrapping(molecule, denominator, aset)
            timeSeries = pd.Series(data={'gesture': train_set['gesture'][idx1],
                                     'acceleration':shiftedAccels})
            timedf_temp=pd.DataFrame([timeSeries])
            timedf=pd.concat([timedf,timedf_temp], ignore_index=True) 
    return timedf

# Add Noise
def gestureWithNoise(train_set,accel_sets):
    noisedf=pd.DataFrame(columns=['gesture','acceleration'])
    for idx1, aset in enumerate(accel_sets):
        for t in range(5):
            tmp_data = [[0 for i in range(len(aset[0]))] for j in range(len(aset))]
            for q in range(len(aset)):
                for j in range(len(aset[q])):
                    tmp_data[q][j] = aset[q][j] + 4 * random.random()
            noiseSeries = pd.Series(data={'gesture': train_set['gesture'][idx1],
                                      'acceleration':tmp_data})  
            noisedf_temp=pd.DataFrame([noiseSeries])
            noisedf=pd.concat([noisedf,noisedf_temp], ignore_index=True)
    return noisedf

# Shift data uniformily up or down in mag
def gestureTimeShift(train_set, accel_sets):
    shiftdf=pd.DataFrame(columns=['gesture','acceleration'])
    for idx1, aset in enumerate(accel_sets):
        for i in range(5):
            shiftSeries = pd.Series(data={'gesture': train_set['gesture'][idx1],
                                      'acceleration':(np.array(aset, dtype=np.float32)+
                                                      ((random.random()- 0.5)*50)).tolist()})
            shiftdf_temp=pd.DataFrame([shiftSeries])
            shiftdf=pd.concat([shiftdf,shiftdf_temp], ignore_index=True)
    return shiftdf

def pad(data, seq_length, dim):
    """Get neighbour padding."""
    noise_level = 1
    padded_data = []
    # Before- Neighbour padding
    tmp_data = (np.random.rand(seq_length, dim) - 0.5) * noise_level + data[0]
    tmp_data[(seq_length -
              min(len(data), seq_length)):] = data[:min(len(data), seq_length)]
    padded_data.append(tmp_data)
    # After- Neighbour padding
    tmp_data = (np.random.rand(seq_length, dim) - 0.5) * noise_level + data[-1]
    tmp_data[:min(len(data), seq_length)] = data[:min(len(data), seq_length)]
    padded_data.append(tmp_data)
    return padded_data


def dataToLength(data_set,seq_length,dim):
    proc_acc = data_set['acceleration'].to_numpy()
    pad_train_df = pd.DataFrame(columns=['gesture','acceleration'])
    for idx4, proacc in enumerate(proc_acc):
        pad_acc = pad(proacc,seq_length,dim)
        for half in pad_acc:
            padSeries = pd.Series(data={'gesture': data_set['gesture'][idx4],
                                      'acceleration': half.tolist()})
            paddf_temp=pd.DataFrame([padSeries])
            pad_train_df=pd.concat([pad_train_df,paddf_temp], ignore_index=True)
    return pad_train_df

if __name__=='__main__':

    gest_id = {'single_wave': 0, 'fist_pump': 1, 'random_motion': 2, 'speed_mode': 3}
    folders = ["fist_pump","single_wave","speed_mode","random_motion"]
    prefolder = "../Training_Data/"
    CompleteData = allDatatoDataframe(prefolder,folders)

    train_ratio = 0.75
    val_ratio = 0.15
    test_ratio = 0.10
    trainingData, validationData, testingData = splitDataSets(CompleteData,train_ratio, val_ratio, test_ratio)


    #see the makeup of each set per gesture, have to add more gestures here later, pandas query with folder loop wasn't working at first pass.
    print('len of trainingData Speed mode: '+ str(len(trainingData.query('gesture == "speed_mode"'))))
    print('len of validationData Speed mode: '+ str(len(validationData.query('gesture == "speed_mode"'))))
    print('len of testingData Speed mode: '+ str(len(testingData.query('gesture == "speed_mode"'))))

    print('len of trainingData fist_pump: '+ str(len(trainingData.query('gesture == "fist_pump"'))))
    print('len of validationData fist_pump: '+ str(len(validationData.query('gesture == "fist_pump"'))))
    print('len of testingData fist_pump: '+ str(len(testingData.query('gesture == "fist_pump"'))))

    print('len of trainingData single_wave: '+ str(len(trainingData.query('gesture == "single_wave"'))))
    print('len of validationData single_wave: '+ str(len(validationData.query('gesture == "single_wave"'))))
    print('len of testingData single_wave: '+ str(len(testingData.query('gesture == "single_wave"'))))

    print('len of trainingData random_motion: '+ str(len(trainingData.query('gesture == "random_motion"'))))
    print('len of validationData random_motion: '+ str(len(validationData.query('gesture == "random_motion"'))))
    print('len of testingData random_motion: '+ str(len(testingData.query('gesture == "random_motion"'))))


    #Data Augmenting
    training_accelerations = trainingData['acceleration'].to_numpy()
    shiftingFractions=[(3, 2), (5, 3), (2, 3), (3, 4), (9, 5), (6, 5), (4, 5)]
    train_Mag_Data = gestureMagnitudeShifting(trainingData, training_accelerations, shiftingFractions)
    train_TimeSS_data = gestureStretchShrink(trainingData, training_accelerations, shiftingFractions)
    train_Noise_data = gestureWithNoise(trainingData, training_accelerations)
    train_Shift_data = gestureTimeShift(trainingData, training_accelerations)

    #combine all the Data sets into one
    processedTrain_set = pd.DataFrame(columns=['gesture','acceleration'])
    processedTrain_set = pd.concat([trainingData,train_Mag_Data, train_TimeSS_data, train_Noise_data, train_Shift_data], ignore_index=True)

    #Set all data to exactly 760 datapoints
    train_final_data = dataToLength(processedTrain_set,760,3)
    val_final_data = dataToLength(validationData,760,3)
    test_final_data = dataToLength(testingData,760,3)

    #Convert the gesture names to id numbers 0-nth gesture
    train_final_data['gesture'] = train_final_data['gesture'].apply(lambda x: gest_id[x])
    test_final_data['gesture'] = test_final_data['gesture'].apply(lambda x: gest_id[x])
    val_final_data['gesture'] = val_final_data['gesture'].apply(lambda x: gest_id[x])

    #save all data to csv
    val_final_data.to_csv('processed_val_set.csv', index=False)
    test_final_data.to_csv('processed_test_set.csv', index=False)
    train_final_data.to_csv('processed_train_set.csv', index=False)