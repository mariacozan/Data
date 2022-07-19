"""Pre-process tiff files."""


import os
import numpy as np
import pandas as pd
import skimage
from skimage import io
from skimage import data
from skimage import metrics
from skimage.util import img_as_float
import tifftools as tt
from pystackreg import StackReg
from suite2p.extraction.extract import extract_traces
from suite2p.extraction.masks import create_masks
from suite2p.registration.register import register_frames
from TwoP.preprocess_traces import correct_neuropil

@jit(forceobj=True)
def _fill_plane_piezo(stack,piezoNorm,i,spacing=1):
    '''
    

    Parameters
    ----------
    stack : array [z,y,X]
        the registered image stack.
    piezoNorm : array [t]
        the piezo depth normalised to the stack spacing.
    i: int
        the frame to create
    Returns
    -------
    a normalised frame.

    '''
    # Normalised the piezo trace to the current depth
    piezoNorm-=piezoNorm[0]
    piezoNorm+=i
    
    planes = stack.shape[0]
    resolutionx = stack.shape[1]
    resolutiony= stack.shape[2]
    currPixelY = 0
    currDepth = 0
    slantImg =np.zeros(stack.shape[1:])      
    
    pixelsPerMoveY = np.ones(len(piezo))*resolutiony
        
    # Nmber of pixels per piezo step
    numPixelsY = np.round(pixelsPerMoveY/len(piezoNorm)).astype(int)
    
    # correct in case of rounding error        
    Yerr = resolutiony-sum(numPixelsY)          
    numPixelsY[-1]+=Yerr
    
    # The end points of each time bit        
    pixelsY = np.cumsum(numPixelsY).astype(int)
    
    interp = sp.interpolate.RegularGridInterpolator((np.arange(0,planes,spacing),np.arange(0,resolutiony),np.arange(0,resolutionx)),stack,fill_value =None)
    for d in range(len(piezoNorm)):            
        endPointY = pixelsY[d]
        depth = piezoNorm[d]
        
        # If beyond the depth take the final frame        
        if (depth>planes-1):
            depth = planes-1
        for yt in np.arange(currPixelY,endPointY):
            # print(depth,)
            # print (depth,yt)
            line = interp((depth,yt,np.arange(0,resolutionx)))       
            slantImg[yt,0:resolutionx] = line
            
        currPixelY+=numPixelsY[d]       
        currDepth+=depth
        
    return slantImg

            
# TODO. Also: return new planes following a certain angle through the z-stack. New plane should follow movement trace of
# piezo.
def register_zstack(tiff_path, spacing = 1, reference = 'first', piezo=None, save = True):
    """
    Loads tiff file containing imaged z-stack, aligns all frames to each other, averages across repetitions, and (if
    piezo not None) reslices the 3D z-stack so that slant/orientation of the new slices matches the slant of the frames
    imaged during experiments (slant given by piezo trace).

    Parameters
    ----------
    tiff_path : String
        Path to tiff file containing z-stack.
    piezo : np.array [t]
        Movement of piezo across z-axis for one plane. Unit: microns. Raw taken from niDaq
    [Note: need to add more input arguments depending on how registration works. Piezo movement might need to provided
    in units of z-stack slices if tiff header does not contain information about depth in microns]
    spacing: distance between planes (in microns)
    reference : string
    what kind of reference image to use, options:
        - first (default, shown to work best)
        - previous
        - mean
    
    Returns
    -------
    zstack : np.array [x x y x z]
        Registered (and resliced) z-stack.
    """
    
    image= skimage.io.imread(filePath)    
    
    # Registration using the function maria discovered https://pypi.org/project/pystackreg/
    planes = image.shape[0]
    resolutionx = image.shape[2]
    resolutiony= image.shape[3]
    zstack=np.zeros((planes, resolutionx, resolutiony))    
    for i in range(planes):
        sr = StackReg(StackReg.TRANSLATION)        
        # reg_arrays = sr.register_transform_stack(image[i,:,:,:], reference=reference)
        res = register_frames(image[i,0,:,:], image[i,:,:,:].astype(np.int16))    
        # zstack[i,:,:] = np.mean(reg_arrays, axis=0)
        zstack[i,:,:] = np.mean(res[0], axis=0)
    
    
        
    
    if not(piezo is None):
        piezoNorm = piezo/spacing        
        depthDiff = np.diff(piezo)
        totalDepthTravelled = sum(depthDiff)
        proportionTavelled = depthDiff/totalDepthTravelled        
        zstackTmp = np.zeros(zstack.shape)
        for p in range(planes):
           # print(p)
           zstackTmp[p,:,:] =  _fill_plane_piezo(zstack,piezoNorm,p)   
        zstack = zstackTmp
    
    
    if (save):
        savePath = os.path.splitext(tiff_path)[0]+'_angled'
        svTmp = savePath
        i = 0
        while (os.path.exists(savePath+'.tif')):            
            savePath = svTmp+str(i)
            i+=1    
        savePath += '.tif'
        io.imsave(savePath, zstack)
    return zstack


def _moffat(r,B,A,alpha,beta):
    return B+A*(1+ (((r)**2)/alpha**2))**-beta

def _gauss(x, A,mu,sigma):    
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
# TODO

def extract_zprofiles(extraction_path, ROI_masks, neuropil_masks, zstack, target_image, neuropil_correction = None, smootingFactor = 2):
    """
    Extracts fluorescence of ROIs across depth of z-stack.

    Parameters
    ----------
    ROI_masks : np.array [x x y x nROIs]
        (output of suite2p so need to check the format of their ROI masks)
        Pixel masks of ROIs in space (x- and y-axis).
    neuropil_masks : np.array [x x y x nROIs]
        (this assumes that suite2p actually uses masks for neuropil. I'm pretty sure there is this option to use masks
        but we need to use this option instead of using "basis functions".)
        Pixel masks of ROI's neuropil in space (x- and y-axis).
    zstack : np.array [x x y x z]
        Registered z-stack where slices are oriented the same way as imaged planes (output of register_zstack).
    target_image : np.array [x x y]
        Image used by suite2p to align frames to. Is needed to align z-stack to this image and then apply masks at
        correct positions.
    #All of these factors can be replaced by the plane directory - Liad 13/07/2022
    neuropil_correction : np.array [nROIs]
        Correction factors determined by preprocess_traces.correct_neuropil.

    Returns
    -------
    zprofiles : np.array [z x nROIs]
        Depth profiles of ROIs.
    """

    """
    Steps
    1) Register z-stack to target image.
    2) Extract fluorescence within ROI masks across all slices of z-stack.
    3) Extract fluorescence within neuropil masks across all slices of z-stack.
    4) Perform neuropil correction on ROI traces using neuropil traces and correction factors.
    
    Notes (useful functions in suite2p);
    - neuropil masks are created in /suite2p/extraction/masks.create_neuropil_masks called from masks.create_masks
    - ROI and neuropil traces extracted in /suite2p/extraction/extract.extract_traces called from 
      extract.extraction_wrapper
    - to register frames, see line 285 (rigid registration) in /suite2p/registration/register for rigid registration
    """
    
    stat = np.load(os.path.join(extraction_path,'stat.npy'),allow_pickle=True)
    ops = np.load(os.path.join(extraction_path,'ops.npy'),allow_pickle=True).item()
    
    ### Step 1
    refImg = ops['refImg']
    X = ops['Lx']
    Y = ops['Ly']  
    
    # res = register_frames(refImg, zstack.astype(np.int16), rmin=ops['rmin'], rmax=ops['rmax'], bidiphase=ops['bidiphase'], ops=ops, nZ=1)    
    # zstack_reg = res[0]
    zstack_reg = zstack
        
    rois, npils = create_masks(stat, Y, X, ops)
        
    zProfile ,Fneu = extract_traces(zstack_reg, rois, npils,1)
    
    # Perform neuropil correction    
    if (not (neuropil_correction is None)):
        Fbl = np.min(F,0)
        zProfile = zProfile - neuropil_correction.reshape(-1,1) * Fneu
        zProfile = np.fmax(zProfile,np.ones(zProfile.shape)*Fbl)
        zProfile = zProfile.T
    zProfileC = np.zeros(zProfile.shape)
    
    depths = np.arange(-(zstack.shape[0]-1)/2,(zstack.shape[0]-1)/2+1)
    
                                    
    return zProfile.T ,Fneu.T
    
