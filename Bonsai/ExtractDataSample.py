# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:19:52 2022

@author: LABadmin
"""

directory = 'Z:\\RawData\\SS113\\2022-07-11\\2\\'

nidaq = GetNidaqChannels(directory + 'NiDaqInput0.bin',5,plot=True)

frameclock = nidaq[:,1]

frames = AssignFrameTime(frameclock,plot=True)

photodiode = nidaq[:,0]

st = DetectPhotodiodeChanges(photodiode,plot=True)

props = ['Ori','SFreq','TFreq','Contrast']

stimProps = GetStimulusInfo(directory+'Log0.csv',props)
#%%
arduino,arduinoTime = GetArduinoData(directory+'ArduinoInput0.csv',plot=False)

#%%
movement1 =  arduino[:,0]
movement2 =  arduino[:,1]
DetectWheelMove(movement1,movement2)

#%%
niTick = np.round(nidaq[:,-1]).astype(bool)
ardTick = np.round(arduino[:,-1]).astype(bool)

niChange = np.where(np.diff(niTick,prepend=True)>0)[0][1:]
niTimes = np.arange(len(niTick))/1000
niChangeTime = niTimes[niChange]
niChangeDuration = np.round(np.diff(niChangeTime),4)

ardChange = np.where(np.diff(ardTick,prepend=True)>0)[0][1:]
ardchangeTime = arduinoTime[ardChange]
ardChangeDuration = np.round(np.diff(ardchangeTime),4)

lags = np.arange(-len(changeDuration) + 1, len(ardChangeDuration))
corr = np.correlate(niChangeDuration,ardChangeDuration,mode='full')

timeShift = lags[np.argmax(corr)]

temporalShift = -np.sign(timeShift)*(niChangeTime[np.abs(timeShift)]-ardChangeTime[0])

arduinoTime+=temporalShift