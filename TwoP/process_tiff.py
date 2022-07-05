"""Pre-process tiff files."""


import numpy as np


# TODO. Also: return new planes following a certain angle through the z-stack. New plane should follow movement trace of
# piezo.
def register_zstack(tiff_path, piezo=None):
    """
    Loads tiff file containing imaged z-stack, aligns all frames to each other, averages across repetitions, and (if
    piezo not None) reslices the 3D z-stack so that slant/orientation of the new slices matches the slant of the frames
    imaged during experiments (slant given by piezo trace).

    Parameters
    ----------
    tiff_path : String
        Path to tiff file containing z-stack.
    piezo : np.array [t]
        Movement of piezo across z-axis for one plane. Unit: microns.
    [Note: need to add more input arguments depending on how registration works. Piezo movement might need to provided
    in units of z-stack slices if tiff header does not contain information about depth in microns]

    Returns
    -------
    zstack : np.array [x x y x z]
        Registered (and resliced) z-stack.
    """

    return zstack


# TODO
def extract_zprofiles(ROI_masks, neuropil_masks, zstack, target_image, neuropil_correction):
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

    return zprofiles