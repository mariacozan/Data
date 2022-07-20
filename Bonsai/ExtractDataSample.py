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
arduinoSync = arduino[:,-1]
niSync = nidaq[:,-1]
niTimes = np.arange(len(niSync))/1000


newTime = arduinoDelayCompensation(niSync,arduinoSync, niTimes,arduinoTime)

#%%
movement1 =  arduino[:,0]
movement2 =  arduino[:,1]
DetectWheelMove(movement1,movement2,arduinoTime)

#%%
