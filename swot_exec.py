import sys

sys.path.append(
    "/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/oceans-research/analysis_scripts/"
)

import collections
import glob
import time
import traceback

import cartopy.crs as ccrs
import flux_sf_figures
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import swot_analysis as swotan
import xarray as xr
from calculate_sfs import StructureFunctions
from calculate_spectral_fluxes import SpectralFlux
from flux_sf_figures import *
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.io import loadmat
from scipy.stats import bootstrap
from swot_analysis import *

import oceans_sf as ocsf

sns.set_style(style="white")
sns.set_context("talk")

plt.rcParams["figure.figsize"] = [9, 6]
# plt.rcParams['figure.dpi'] = 100
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
# %config InlineBackend.figure_format = 'svg'

import os

os.environ["PATH"] = os.environ["PATH"] + ":/Library/TeX/texbin"
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

import warnings

from geopy import distance as gd

warnings.filterwarnings("ignore")

from datetime import datetime

from astropy import constants as c
from astropy import units as u

ds_dict = swotan.load_data(filepath="data/acc_data/*.nc")

processed_data = []
for ds_name in ds_dict:
    ds = ds_dict[ds_name]

    try:
        ds = swotan.preprocess_dataset(ds, latmin=-70, latmax=-50, lonmin=0, lonmax=360)
        processed_data.append(ds)
    except:
        pass

adv_sf_data_list = sf_run(
    ds_dict=ds_dict,
    SFtype="adv",
    figname="swot_figures/acc_adv_%s.png"
    % (datetime.today().strftime("%Y-%m-%d_%H%M%S")),
    latmin=-70,
    latmax=-50,
    lonmin=0,
    lonmax=360,
    ymin=-1e-11,
    ymax=1e-11,
)
