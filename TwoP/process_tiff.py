"""Pre-process tiff files."""

# TODO. Also: return new planes following a certain angle through the z-stack. New plane should follow movement trace of
# piezo.
import numpy as np


def register_zstack():


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
    """

    return zprofiles