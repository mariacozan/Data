# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:19:52 2022

@author: LABadmin
"""

import numpy as np 
from matplotlib import pyplot as plt
import random
import sklearn
import seaborn as sns
import scipy as sp
from matplotlib import rc
import matplotlib.ticker as mtick
import matplotlib as mpl

import os
import glob
import pickle

metadataDirectory = 'Z:\\RawData\\SS113\\2022-07-11\\1\\'
# metadataDirectory = 'Z:\\RawData\\Hedes\\2022-07-19\\1\\'
saveDirectory = os.path.join(metadataDirectory,'PreprocessedFiles')

if not os.path.isdir(saveDirectory):
    os.makedirs(saveDirectory)

nidaq,nidaqt = GetNidaqChannels(metadataDirectory,5,plot=True)
frameclock = nidaq[:,1]
frames = AssignFrameTime(frameclock,plot=True)
photodiode = nidaq[:,0]
st = DetectPhotodiodeChanges(photodiode,plot=True)
sparse = GetSparseNoise(metadataDirectory,size = (8,9))
sparse = sparse[st]
np.save(os.path.join(saveDirectory,'sparse.times.npy'),st)
np.save(os.path.join(saveDirectory,'sparse.map.npy'),sparse)
np.save(os.path.join(saveDirectory,'sparse.id.npy'),np.range(sparse.shape[0]))
# props = ['Ori','SFreq','TFreq','Contrast']
# stimProps = GetStimulusInfo(metadataDirectory+'Log0.csv',props)
#%%

metadataDirectory = 'Z:/RawData/Hedes/2022-08-04/1'
# metadataDirectory = 'Z:\\RawData\\SS113\\2022-07-11\\1\\'

nidaq,chans,nt = GetNidaqChannels(metadataDirectory, plot=False)
arduino,ardChans, at = GetArduinoData(metadataDirectory ,plot=False)

# arduinoSync = np.round(arduino[:,-1]).astype(bool)
arduinoSync = arduino[:,ardChans == 'sync'][:,0]
arduinoSync2 = arduino[:,-1]
niSync =  nidaq[:,chans=='sync'][:,0]
niSync2 =  nidaq[:,-1]  

# niSync = np.round(nidaq[:,-1]).astype(bool)

#%%
plt.close('all')
at_new = arduinoDelayCompensation(niSync,arduinoSync, nt,at)
plt.plot(nt,niSync)
plt.plot(at_new,arduinoSync)
#%%
movement1 =  arduino[:,ardChans == 'rotary1'][:,0]
movement2 =  arduino[:,ardChans == 'rotary2'][:,0]
v,d = DetectWheelMove(movement1,movement2,at_new)

# np.save(os.path.join(saveDirectory,'running.timestamps.npy'),at_new)
# np.save(os.path.join(saveDirectory,'running.speed.npy'),v)



#%%


#%%
camera1 = arduino[:,ardChans == 'camera1']
st = AssignFrameTime(camera1,plot=True)

import cv2
cap = cv2.VideoCapture(metadataDirectory+'/Video00.avi')
#%%
b,a = sp.signal.butter(1, 30, btype='low', fs=1000)
sigFilt = sp.signal.filtfilt(b,a,photodiode)
sigFilt = sp.signal.medfilt(sigFilt,101)
st = DetectPhotodiodeChanges(sigFilt,plot=True,downThreshold =0.4)