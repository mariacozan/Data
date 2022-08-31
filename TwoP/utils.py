# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:37:13 2022

@author: LABadmin
"""

def get_piezo_trace_for_plane(piezo,plane,maxDepth,winSize=20):    
    w = np.hanning(winSize)
    w/=np.sum(w)
    piezo =  np.convolve(piezo,w,'same')
    plane0ind = np.where((nt>=frames[5+plane]) & (nt<frames[5+plane+1]))    
    piezo-=np.min(piezo)
    piezo/=np.nanmax(piezo)
    piezo*=maxDepth
    planePiezo = piezo[plane0ind]
    return planePiezo