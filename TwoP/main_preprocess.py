import numpy as np
from matplotlib import pyplot as plt
import random
import sklearn
import seaborn as sns
import scipy as sp
from matplotlib import rc
import matplotlib.ticker as mtick
import matplotlib as mpl
import pandas as pd
import os
import glob
import pickle
import Data.TwoP
import Data.Bonsai


# %%
# define directories # Change to a different file
csvDir = "D:\\preprocess.csv"
s2pDir = "D:\\Suite2Pprocessedfiles\\"
zstackDir = "Z:\\RawData\\"
metadatadataDir = "Z:\\RawData\\"

# %%
# read database
database = pd.read_csv(
    csvDir,
    dtype={
        "Name": str,
        "Date": str,
        "Zstack": str,
        "IgnorePlanes": str,
        "SaveDir": str,
        "Process": bool,
    },
)


# %% run over data base
for i in range(len(database)):

    print("reading directories")
    (
        s2pDirectory,
        zstackPath,
        metadataDirectory,
        saveDirectory,
    ) = read_csv_produce_directories(database.loc[i])
    ops = get_ops_file(s2pDirectory)
    print("getting piezo data")
    planePiezo = get_piezo_data(ops)
    print("processing suite2p data")
    fc = process_s2p_directory(
        s2pDirectory,
        planePiezo,
        zstackPath,
        saveDirectory=saveDirectory,
        ignorePlanes=[0],
        debug=True,
    )
    print("reading bonsai data")
    process_metadata_directory(metadataDirectory, ops, saveDirectory)

    # %%
