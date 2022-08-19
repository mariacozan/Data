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

directory = 'Z:\\RawData\\SS113\\2022-07-11\\2\\'
# directory = 'Z:\\RawData\\Hedes\\2022-07-19\\1\\'

nidaq,nidaqt = GetNidaqChannels(directory + 'NiDaqInput0.bin',5,plot=True)

frameclock = nidaq[:,1]

frames = AssignFrameTime(frameclock,plot=True)

photodiode = nidaq[:,0]

st = DetectPhotodiodeChanges(photodiode,plot=True)


props = ['Ori','SFreq','TFreq','Contrast']

stimProps = GetStimulusInfo(directory+'Log1.csv',props)

sparse = GetSparseNoise(directory+'SparseNoise0.bin',size = (8,9))

#%%
arduino,arduinoTime = GetArduinoData(directory+'ArduinoInput0.csv',plot=True)
arduinoSync = np.round(arduino[:,-1]).astype(bool)
niSync = np.round(nidaq[:,-1]).astype(bool)
niTimes = np.arange(len(niSync))/1000

#%%
newTime = arduinoDelayCompensation(niSync,arduinoSync, niTimes,arduinoTime)

#%%
movement1 =  arduino[:,0]
movement2 =  arduino[:,1]
v,d = DetectWheelMove(movement1,movement2,arduinoTime)

#%%
camera1 = arduino[:,2]
st = AssignFrameTime(camera1,plot=True)

import cv2
cap = cv2.VideoCapture(directory+'Video0.avi')
#%%
b,a = sp.signal.butter(1, 30, btype='low', fs=1000)
sigFilt = sp.signal.filtfilt(b,a,photodiode)
sigFilt = sp.signal.medfilt(sigFilt,101)
st = DetectPhotodiodeChanges(sigFilt,plot=True,downThreshold =0.4)