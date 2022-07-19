"""Pre-process data recorded with Bonsai."""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 09:26:53 2022

@author: Liad J. Baruchin
"""

import numpy as np
from matplotlib import pyplot as plt
import csv

def GetNidaqChannels(niDaqFilePath, numChannels = 5, plot = False):
    """
    Get the nidaq channels

    Parameters
    ----------
    niDaqFilePath : string
        the path of the nidaq file.
    numChannels : int, optional
        Number of channels in the file. The default is 7.

    Returns
    -------
    niDaq : matrix
        the matrix of the niDaq signals [time X channels]

    """
    niDaq = np.fromfile(niDaqFilePath, dtype= np.float64)
    niDaq = np.reshape(niDaq,(int(len(niDaq)/numChannels),numChannels))
    
    if (plot):
        f,ax = plt.subplots(numChannels,sharex=True)
        for i in range(numChannels):
            ax[i].plot(niDaq[:,i])
            
    return niDaq

def AssignFrameTime(frameClock,th = 0.5,plot=False,fs = 1000):
    """
    The function assigns a time in ms to a frame time.
    
    Parameters:
    frameClock: the signal from the nidaq of the frame clock
    th : the threshold for the tick peaks, default : 3, which seems to work 
    plot: plot to inspect, default = False
    fs: the frame rate of acquisition default is 1000Hz
    returns frame start Times (s)
    """
    #Frame times
    # pkTimes,_ = sp.signal.find_peaks(-frameClock,threshold=th)
    # pkTimes = np.where(frameClock<th)[0]
    # fdif = np.diff(pkTimes)
    # longFrame = np.where(fdif==1)[0]
    # pkTimes = np.delete(pkTimes,longFrame)
    # recordingTimes = np.arange(0,len(frameClock),0.001)
    # frameTimes = recordingTimes[pkTimes]
    
    # threshold = 0.5
    pkTimes = np.where(np.diff(frameClock > th, prepend=False))[0]    
    # pkTimes = np.where(np.diff(np.array(frameClock > 0).astype(int),prepend=False)>0)[0]
       
    
    if (plot):
        f,ax = plt.subplots(1)
        ax.plot(frameClock)
        ax.plot(pkTimes,np.ones(len(pkTimes))*np.min(frameClock),'r*')
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude (V)')
        
        
    return pkTimes[::2]/fs


def DetectPhotodiodeChanges(photodiode,plot=False,kernel = 101,upThreshold = 0.2, downThreshold = 0.4,fs=1000, waitTime=5000):
    """
    The function detects photodiode changes using a 'Schmitt Trigger', that is, by
    detecting the signal going up at an earlier point than the signal going down,
    the signal is filtered and smootehd to prevent nosiy bursts distorting the detection.W
    
    Parameters: 
    photodiode: the signal from the nidaq of the photodiode    
    lowPass: the low pass signal for the photodiode signal, default: 30,
    kernel: the kernel for median filtering, default = 101.
    fs: the frequency of acquisiton, default = 1000
    plot: plot to inspect, default = False   
    waitTime: the delay time until protocol start, default = 5000
    
    returns: diode changes (s) up to the user to decide what on and off mean
    """    
    
    # b,a = sp.signal.butter(1, lowPass, btype='low', fs=fs)
    sigFilt = photodiode
    # sigFilt = sp.signal.filtfilt(b,a,photodiode)
    sigFilt = sp.signal.medfilt(sigFilt,kernel)
   
  
    maxSig = np.max(sigFilt)
    minSig = np.min(sigFilt)
    thresholdU = (maxSig-minSig)*upThreshold
    thresholdD = (maxSig-minSig)*downThreshold
    threshold =  (maxSig-minSig)*0.5
    
    # find thesehold crossings
    crossingsU = np.where(np.diff(np.array(sigFilt > thresholdU).astype(int),prepend=False)>0)[0]
    crossingsD = np.where(np.diff(np.array(sigFilt > thresholdD).astype(int),prepend=False)<0)[0]
    crossingsU = np.delete(crossingsU,np.where(crossingsU<waitTime)[0])     
    crossingsD = np.delete(crossingsD,np.where(crossingsD<waitTime)[0])   
    crossings = np.sort(np.unique(np.hstack((crossingsU,crossingsD))))
  
    
    if (plot):
        f,ax = plt.subplots(1,1,sharex=True)
        ax.plot(photodiode,label='photodiode raw')
        ax.plot(sigFilt,label = 'photodiode filtered')        
        ax.plot(crossings,np.ones(len(crossings))*threshold,'g*')  
        ax.hlines([thresholdU],0,len(photodiode),'k')
        ax.hlines([thresholdD],0,len(photodiode),'k')
        # ax.plot(st,np.ones(len(crossingsD))*threshold,'r*')  
        ax.legend()
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Amplitude (V)')    

    return crossings/fs

def DetectWheelMove(moveA,moveB,rev_res = 1024, total_track = 598.47,plot=False):
    """
    The function detects the wheel movement. 
    At the moment uses only moveA.    
    
    Parameters: 
    moveA,moveB: the first and second channel of the rotary encoder
    rev_res: the rotary encoder resoution, default =1024
    total_track: the total length of the track, default = 598.47 (mm)
    kernel: the kernel for median filtering, default = 101.
    
    plot: plot to inspect, default = False   
    
    returns: distance
    """
    
    
    # make sure all is between 1 and 0
    moveB = -moveB
    moveA /= np.max(moveA)
    moveA -= np.min(moveA)
    moveB -= np.min(moveB)
    moveB /= np.max(moveB)
    moveA = np.round(moveA).astype(bool)
    moveB = np.round(moveB).astype(bool)
    
    # detect A move
    ADiff = np.where(np.diff(moveA>0,prepend=True))[0]
    
    # Ast = np.where(ADiff >0.5)[0]
    # Aet = np.where(ADiff <-0.5)[0]
    
    # detect B move
    BDiff = np.where(np.diff(moveB>0,prepend=True))[0]#np.diff(moveB)
    
    # Bst = np.where(BDiff >0.5)[0]
    # Bet = np.where(BDiff <-0.5)[0]
    
    #Correct possible problems for end of recording
    if (len(Ast)>len(Aet)):
        Aet = np.hstack((Aet,[len(moveA)]))
    elif (len(Ast)<len(Aet)):
        Ast = np.hstack(([0],Ast))   
    
    
    dist_per_move = total_track/rev_res
    
    # Make into distance
    track = np.zeros(len(moveA))
    track[Ast] = dist_per_move
    
    distance = np.cumsum(track)
        
    if (plot):
        f,ax = plt.subplots(3,1,sharex=True)
        ax[0].plot(moveA)
        # ax.plot(np.abs(ADiff))
        ax[0].plot(Ast,np.ones(len(Ast)),'k*')
        ax[0].plot(Aet,np.ones(len(Aet)),'r*')
        ax[0].set_xlabel('time (ms)')
        ax[0].set_ylabel('Amplitude (V)')
        
        ax[1].plot(distance)
        ax[1].set_xlabel('time (ms)')
        ax[1].set_ylabel('distance (mm)')
        
        ax[2].plot(track)
        ax[2].set_xlabel('time (ms)')
        ax[2].set_ylabel('Move')
    
    # movFirst = Amoves>Bmoves
    
    return distance
  
def GetSparseNoise(filePath, size=(20,25)):
    """
    Pulls the sparse noise from the directory
    
    Parameters: 
    filePath: The full file path for the sparse noise file
    size: a tuple for the size of the screen. default = (20,25)         
    
    returns: an array of size [frames X size[0] X size[1]]
    """
    sparse = np.fromfile(filePath, dtype= np.dtype('b'))
    sparse = np.reshape(sparse,(int(len(sparse)/(size[0]*size[1])),size[0],size[1]))
    # sparse = np.reshape(sparse,(size[0],size[1],int(len(sparse)/(size[0]*size[1]))))
    return sparse

def GetLogEntry(filePath,entryString):
    """
    

    Parameters
    ----------
    filePath : str
        the path of the log file.
    entryString : the string of the entry to look for

    Returns
    -------
    StimProperties : list of dictionaries
        the list has all the extracted stimuli, each a dictionary with the props and their values.

    """
    
    

    StimProperties  = []
    
    with open(filePath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            a = []
            for p in range(len(props)):
                # m = re.findall(props[p]+'=(\d*)', row[np.min([len(row)-1,p])])
                m = re.findall(entryString, row[np.min([len(row)-1,p])])
                if (len(m)>0):
                    a.append(m[0])            
            if (len(a)>0):
                stimProps = {}
                for p in range(len(props)):
                    stimProps[props[p]] = a[p]
                StimProperties.append(stimProps)
    return StimProperties

def GetStimulusInfo(filePath,props):
    """
    

    Parameters
    ----------
    filePath : str
        the path of the log file.
    props : array-like
        the names of the properties to extract. Must be exact string

    Returns
    -------
    StimProperties : list of dictionaries
        the list has all the extracted stimuli, each a dictionary with the props and their values.

    """  

    StimProperties  = []
    
    with open(filePath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            a = []
            for p in range(len(props)):
                # m = re.findall(props[p]+'=(\d*)', row[np.min([len(row)-1,p])])
                m = re.findall(props[p]+'=([a-zA-Z0-9_.-]*)', row[np.min([len(row)-1,p])])
                if (len(m)>0):
                    a.append(m[0])            
            if (len(a)>0):
                stimProps = {}
                for p in range(len(props)):
                    stimProps[props[p]] = a[p]
                StimProperties.append(stimProps)
    return StimProperties

def GetArduinoData(arduinoFilePath,th = 3,plot=False):
    '''
    Retrieves the arduino data, regularises it (getting rid of small intervals)
    Always assume last entry is the timepoints

    Parameters
    ----------
    arduinoFilePath : str
        The path of the arduino file.
    plot : bool, optional
        Whether or not to plot all the channels.. The default is False.

    Returns
    -------
    csvChannels : array-like [time X channels]
        all the channels recorded by the arduino.

    '''
    csvChannels = np.genfromtxt(arduinoFilePath,delimiter=',')
    
    
    arduinoTime = csvChannels[:,-1]
    arduinoTimeDiff = np.diff(arduinoTime,prepend=True)
    normalTimeDiff = np.where(arduinoTimeDiff>3)[0]
    csvChannels = csvChannels[normalTimeDiff,:]
    # convert time to second (always in ms)
    arduinoTime = csvChannels[:,-1]/1000 
    arduinoTime-=arduinoTime[0]
    csvChannels = csvChannels[:,:-1]
    numChannels = csvChannels.shape[1]
    if (plot):
        f,ax = plt.subplots(numChannels,sharex=True)
        for i in range(numChannels):
            ax[i].plot(arduinoTime,csvChannels[:,i])
            
    
    return csvChannels,arduinoTime

