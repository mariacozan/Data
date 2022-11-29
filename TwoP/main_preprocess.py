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
from Data.TwoP.runners import *
from Data.Bonsai.extract_data import *
from Data.TwoP.folder_defs import *


# %%
csvDir, s2pDir, zstackDir, metadataDir = define_directories()

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
    if database.loc[i]["Process"]:
        print("reading directories")
        (
            s2pDirectory,
            zstackPath,
            metadataDirectory,
            saveDirectory,
        ) = read_csv_produce_directories(
            database.loc[i], s2pDir, zstackDir, metadataDir
        )
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
            debug=False,
        )
        print("reading bonsai data")
        process_metadata_directory(metadataDirectory, ops, saveDirectory)
    else:
        print("skipping " + str(database.loc[i]))

    # %%
