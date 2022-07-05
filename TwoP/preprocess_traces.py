"""Pre-process calcium traces extracted from tiff files."""
import numpy as np
from scipy import optimize


def correct_neuropil(F: np.ndarray, N: np.ndarray, numN=20, minNp=10, maxNp=90, prctl_F=5, verbose=True):
    """
    Estimates the correction factor r for neuropil correction, so that:
        C = S - rN
        with C: actual signal from the ROI, S: measured signal, N: neuropil

    Parameters
    ----------
    F : np.ndarray [t x nROIs]
        Calcium traces (measured signal) of ROIs.
    N : np.ndarray [t x nROIs]
        Neuropil traces of ROIs.
    numN : int, optional
        Number of bins used to partition the distribution of neuropil values. Each bin will be associated with
        a mean neuropil value and a mean signal value. The default is 20.
    minNp : int, optional
        Minimum values of neuropil considered, expressed in percentile. 0 < minNp < 100. The default is 10.
    maxNp : int, optional
        Maximum values of neuropil considered, expressed in percentile. 0 < maxNp < 100, minNp < maxNp. The
        default is 90.
    prctl_F : int, optional
        Percentile of the measured signal that will be matched to neuropil. The default is 5.
    verbose : boolean, optional
        Feedback on fitting. The default is True.

    Returns
    -------
    signal : np.ndarray [t x nROIs]
        Neuropil corrected calcium traces.
    regPars : np.ndarray [2 x nROIs], each row: [intercept, slope]
        Intercept and slope of linear fits of neuropil (N) to measured calcium traces (F)
    F_binValues : np.array [numN, nROIs]
        Low percentile (prctl_F) values for each calcium trace bin. These values were used for linear regression.
    N_binValues : np.array [numN, nROIs]
        Values for each neuropil bin. These values were used for linear regression.

    Based on Matlab function estimateNeuropil (in +preproc) written by Mario Dipoppa and Sylvia Schroeder
    """

    [nt, nROIs] = F.shape
    N_binValues = np.ones((numN, nROIs)) * np.nan
    F_binValues = np.ones((numN, nROIs)) * np.nan
    regPars = np.ones((2, nROIs)) * np.nan
    signal = np.ones((nt, nROIs)) * np.nan

    # TODO: set verbose options

    for iROI in range(nROIs):
        # TODO: verbose options

        iN = N[:, iROI]
        iF = F[:, iROI]

        # get low and high percentile of neuropil trace
        N_prct = np.nanpercentile(iN, np.array([minNp, maxNp]), axis=0)
        # divide neuropil values into numN groups
        binSize = (N_prct[1] - N_prct[0]) / numN
        # get neuropil values regularly spaced across range between minNp and maxNp
        N_binValues[:, iROI] = N_prct[0] + (np.arange(stop=numN) + 1) * binSize

        # discretize values of neuropil between minN and maxN, with numN elements
        # N_ind contains values: 0...binSize for N values within minNp and maxNp
        N_ind = np.floor((iN - N_prct[0]) / binSize)

        # for each neuropil bin, find the matching (low percentile) value from F trace
        for iN in range(numN):
            tmp = np.ones_like(iF) * np.nan
            tmp[N_ind == iN] = iF[N_ind == iN]
            F_binValues[iN, iROI] = np.nanpercentile(tmp, prctl_F, 0)

        # perform linear regression between neuropil and signal bins under constraint that 0<slope<2
        res, _ = optimize.curve_fit(_linear, N_binValues[:, iROI], F_binValues[:, iROI],
                                    p0=(np.nanmean(F_binValues[:, iROI]), 0), bounds=([-np.inf, 0], [np.inf, 2]))
        regPars[:, iROI] = res
        # determine neuropil correct signal
        signal[:, iROI] = iF - res[1] * iN

    return signal, regPars, F_binValues, N_binValues


# TODO
def correct_zmotion(F, zprofiles, ztrace):
    """
    Corrects changes in fluorescence due to brain movement along z-axis (depth). Method is based on algorithm
    described in Ryan, ..., Lagnado (J Physiol, 2020)

    Parameters
    ----------
    F : np.array [t x nROIs]
        Calcium traces (measured signal) of ROIs from a single(!) plane. It is assumed that these are neuropil corrected!
    zprofiles : np.array [slices x nROIs]
        Fluorescence profiles of ROIs across depth of z-stack. These profiles are assumed to be neuropil corrected!
    ztrace : np.array [t]
        Depth of each frame of the imaged plane. Indices in this array refer to slices in zprofiles.

    Returns
    -------
    signal : np.array [t x nROIs]
        Z-corrected calcium traces.
    """

    """
    Steps
    1) Smooth z-profile of each ROI using Moffat function.
    2) Create correction vector based on z-profiles and ztrace.
    3) Correct calcium traces using correction vector.
    """
    return signal


# TODO
def register_zaxis():


# TODO
def get_F0():


# TODO
def get_delta_F_over_F():


def _linear(x, a, b):
    return a + b * x
