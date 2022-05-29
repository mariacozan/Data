"""Pre-process calcium traces extracted from tiff files."""
import math
import numpy as np


def correct_neuropil(F: np.ndarray, N: np.ndarray, numN=20, minNp=10, maxNp=90, pCell=5, noNeg=True,
                     constrainedFit=False, window=math.inf, stepSize=None, verbose=True):
    # if stepSize is None:
    #     if window is math.inf:
    #         stepSize = math.inf
    #     else:
    #         stepSize = round(window / 20)

    [nROIs, nt] = F.shape
    # fitNeuro = np.ones((nROIS, numN)) * np.nan
    # lowCell = np.ones((nROIS, numN)) * np.nan
    # corrFactor = np.ones((nROIs, 2)) * np.nan
    #
    # return
