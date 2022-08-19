# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:22:36 2022

@author: LABadmin
"""
import os
import suite2p
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp


directory = 'D:\\Suite2Pprocessedfiles\\Hedes\\2022-03-23\\suite2p\\plane0\\'

F = np.load(os.path.join(directory,'F.npy')).T
N = np.load(os.path.join(directory,'Fneu.npy')).T
F_corr, regPars, F_binValues, N_binValues = correct_neuropil(F,N)
#%% Z Stack creation
zdir = 'Z:\\RawData\\Hedes\\2022-03-23\\3\\file_00003_00001.tif'
piezoDir = 'Z:\\RawData\\Hedes\\2022-03-23\\1\\'
nidaq,nt = GetNidaqChannels(piezoDir + 'NiDaqInput0.bin',4,plot=False)
piezo = nidaq[:,-1]
# piezo =  sp.signal.medfilt(piezo,41)
w = np.hanning(10)
w/=np.sum(w)
piezo =  np.convolve(piezo,w)
frameclock = nidaq[:,1]
# piezo/=np.nanmax(piezo)
# piezo*=45
frames = AssignFrameTime(frameclock,plot=False)
plane0ind = np.where((nt>=frames[0]) & (nt<frames[1]))
# plane0piezo = sp.signal.medfilt(piezo[plane0ind],21) 
plane0piezo = piezo[plane0ind]
#%%
zstack = register_zstack(zdir, spacing = 1, piezo=plane0piezo, save = False)
#%%
zstack = io.imread('Z:\\RawData\\Hedes\\2022-03-23\\3\\file_00003_00001_angled0.tif')
#%%
zprofilesC, zprofiles, neuropil = extract_zprofiles(directory, zstack,neuropil_correction =regPars[1,:] )
#%%
ops = np.load(os.path.join(directory,'ops.npy'),allow_pickle=True).item()
zTrace = np.argmax(ops['zcorr'],0)
#%%
F_zcorrected = correct_zmotion(F_corr, zprofilesC, zTrace)
#%%
n = 10
f,ax = plt.subplots(2,1,sharex=True)
ax[0].plot(F_corr[:,n],'k--')
ax[0].plot(F_zcorrected[:,n],'r')
ax[1].plot(zTrace)
#%%
n = 10
F = np.load(os.path.join(directory,'F.npy')).T
plt.plot(F[:,n],'r')
plt.plot(F_corr[:,n],'k')
plt.plot(N[:,n],'b')

#%%
plt.close('all')
stat = np.load(os.path.join(directory,'stat.npy'),allow_pickle=True)
ops = np.load(os.path.join(directory,'ops.npy'),allow_pickle=True).item()
refImg = ops['refImg']
n = 4


for n in range(zprofiles.shape[1]):

    im = np.zeros((ops['Ly'], ops['Lx']))
    # im = refImg
    ypix = stat[n]['ypix']#[~stat[n]['overlap']]
    
    xpix = stat[n]['xpix']#[~stat[n]['overlap']]
    im[ypix,xpix] = 1
    # im[nypix,nxpix] = 0
    plt.figure()
    r = refImg.copy()
    r[im.astype(bool)] = -10000
    plt.imshow(r)
    plt.figure()
    plt.plot(zprofiles[:,n],'k')
    # plt.vlines([np.min(zTrace),np.max(zTrace)],np.min(zprofiles[:,n])-100,np.max(zprofiles[:,n])+100,'r')
    plt.vlines([np.nanpercentile(zTrace,2.5),np.nanpercentile(zTrace,97.5)],np.min(zprofiles[:,n])-100,np.max(zprofiles[:,n])+100,'r')
    plt.ylim(np.min(zprofiles[:,n]),np.max(zprofiles[:,n]))
    plt.savefig('D:\\Suite2Pprocessedfiles\\Hedes\\2022-03-23\\zprofiles\\neuron'+str(n)+'_corrected.png')
    plt.close('all')
    plt.figure()
    plt.plot(zprofilesC[:,n],'k')
    plt.plot(neuropil[:,n],'b')
    # plt.vlines([np.min(zTrace),np.max(zTrace)],np.min(zprofilesC[:,n])-100,np.max(zprofilesC[:,n])+100,'r')
    plt.vlines([np.nanpercentile(zTrace,2.5),np.nanpercentile(zTrace,97.5)],np.min(zprofilesC[:,n])-100,np.max(zprofilesC[:,n])+100,'r')
    plt.savefig('D:\\Suite2Pprocessedfiles\\Hedes\\2022-03-23\\zprofiles\\neuron'+str(n)+'_raw.png')
    plt.close('all')

#%% Fitting Tryouts
plt.close('all')
plt.plot(zprofilesC[:,n],'k--')
n = 16



def _moffat(r,B,A,r0,alpha,beta):
    return B+A*(1+ (((r-r0)**2)/alpha**2))**(-beta)

def _gauss(x, A,mu,sigma):    
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

z = zprofilesC[:,n]
zprofilesC[:,n]
z/=np.max(z)
z-=np.min(z)

res, _ = optimize.curve_fit(_moffat, np.arange(0,101), z,
                            p0=(0,1,50,1,1), bounds=([0, 0,-np.inf,-np.inf,-np.inf], [np.inf, np.inf,np.inf,np.inf,np.inf]))


plt.plot(_moffat(np.arange(0,101),*res),'r')