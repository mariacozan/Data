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


# directory = 'D:\\Suite2Pprocessedfiles\\Hedes\\2022-03-23\\suite2p\\plane1\\'
directory = 'D:\\Suite2Pprocessedfiles\\SS113\\2022-07-11\\suite2p\\plane1\\'
frequency = 58
#%% Load suite2p outputs
F = np.load(os.path.join(directory,'F.npy'), allow_pickle=True).T
N = np.load(os.path.join(directory,'Fneu.npy')).T
isCell = np.load(os.path.join(directory,'iscell.npy')).T
stat = np.load(os.path.join(directory,'stat.npy'),allow_pickle=True)
ops = np.load(os.path.join(directory,'ops.npy'),allow_pickle=True).item()
processing_metadata = {}

fs = ops['fs']
F = F[:,isCell[0,:].astype(bool)]
N = N[:,isCell[0,:].astype(bool)]
Fc, regPars, F_binValues, N_binValues = correct_neuropil(F,N)
F0  = get_F0(F_corr,fs)
dF = get_delta_F_over_F(Fc,F0)
#TODO: save 

#%% get Piezo Trace
# zdir = 'Z:\\RawData\\Hedes\\2022-03-23\\3\\file_00003_00001.tif'
# piezoDir = 'Z:\\RawData\\Hedes\\2022-03-23\\1\\'
zdir = 'Z:\\RawData\\SS113\\2022-07-11\\4\\file_00001_00001.tif'
piezoDir = 'Z:\\RawData\\SS113\\2022-07-11\\1\\'
nidaq,nt = GetNidaqChannels(piezoDir + 'NiDaqInput0.bin',5,plot=False)
frameclock = nidaq[:,1]
frames = AssignFrameTime(frameclock,plot=False)
piezo = nidaq[:,-2].copy()
planePiezo = get_piezo_trace_for_plane(piezo,plane = 1,maxDepth = 10)

#%% Xreate the z-stack with a piezo angle
zstack = register_zstack(zdir, spacing = 1, piezo = planePiezo, target_image = refImg)
#TODO : save z-stack
#%%
zstack = io.imread('Z:\\RawData\\SS113\\2022-07-11\\4\\file_00001_00001_angled.tif')
#%% Retrieve z-profiles and retrieve corrected F
refImg = ops['refImg'][1]
zprofiles = extract_zprofiles(directory,zstack,neuropil_correction =regPars[1,:] ,metadata = processing_metadata)
zTrace = np.argmax(ops['zcorr'],0)
Fcz = correct_zmotion(dF, zprofiles, zTrace)

#TODO: save registered z-stack
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
refImg = ops['refImg'][0]
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
    plt.savefig(directory+'Zprofiles\\neuron'+str(n)+'_corrected.png')
    plt.close('all')
    plt.figure()
    plt.plot(zprofilesC[:,n],'k')
    plt.plot(neuropil[:,n],'b')
    # plt.vlines([np.min(zTrace),np.max(zTrace)],np.min(zprofilesC[:,n])-100,np.max(zprofilesC[:,n])+100,'r')
    plt.vlines([np.nanpercentile(zTrace,2.5),np.nanpercentile(zTrace,97.5)],np.min(zprofilesC[:,n])-100,np.max(zprofilesC[:,n])+100,'r')
    plt.savefig(directory+'Zprofiles\\neuron'+str(n)+'_raw.png')
    plt.close('all')

#%% Hunt for the ROI
plt.close('all')
stat = np.load(os.path.join(directory,'stat.npy'),allow_pickle=True)
stat = stat[isCell[0,:].astype(bool)]
ops = np.load(os.path.join(directory,'ops.npy'),allow_pickle=True).item()
refImg = ops['refImg'][1]



n =0 
im = np.zeros((ops['Ly'], ops['Lx']))
# im = refImg
ypix = stat[n]['ypix'][~stat[n]['overlap']]

xpix = stat[n]['xpix'][~stat[n]['overlap']]
im[ypix,xpix] = 1
# im[nypix,nxpix] = 0
plt.figure()
r = refImg.copy()
r[im.astype(bool)] = -10000
plt.imshow(im,cmap = 'bone')

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