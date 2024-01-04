import sys

sys.path.append(
    "/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/oceans-research/analysis_scripts/"
)

from scipy.io import loadmat
from scipy.stats import bootstrap
import traceback
import pandas as pd
import numpy as np
import h5py
import oceans_sf as ocsf
import collections
from calculate_spectral_fluxes import SpectralFlux
from calculate_sfs import StructureFunctions
from flux_sf_figures import *
import flux_sf_figures
from matplotlib import pyplot as plt
import time
import xarray as xr
import cartopy.crs as ccrs
import glob

import seaborn as sns

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

from geopy import distance as gd

import warnings

warnings.filterwarnings("ignore")

from astropy import units as u
from astropy import constants as c


def geostrophic_velocity(ds, dx, dy):

    da = ds.simulated_true_ssh_karin
    dx = ((dx * u.m).cgs).value
    dy = ((dy * u.m).cgs).value

    omega = 7.2921e-5 / u.s
    detadx, detady = np.gradient(da, dx, dy, axis=(1, 0))
    ug = -(c.g0.cgs / (2 * omega * np.sin(da.latitude * np.pi / 180))) * detady
    vg = (c.g0.cgs / (2 * omega * np.sin(da.latitude * np.pi / 180))) * detadx
    vel_mag = np.sqrt(ug**2 + vg**2)

    ds["vel_mag"] = ug
    ds.vel_mag.data = vel_mag
    ug = ug.rename("u")
    vg = vg.rename("v")
    ds.vel_mag.attrs["long_name"] = "sea surface velocity magnitude"
    ug.attrs["long_name"] = "sea surface u-velocity"
    vg.attrs["long_name"] = "sea surface v-velocity"
    ds.vel_mag.attrs["units"] = "cm/s"
    ug.attrs["units"] = "cm/s"
    vg.attrs["units"] = "cm/s"

    ds["u"] = ug
    ds["v"] = vg
    return ds


def rotational_parameters(ds, dx, dy):

    ug = ds["u"]
    vg = ds["v"]

    dx = ((dx * u.m).cgs).value
    dy = ((dy * u.m).cgs).value

    dudx, dudy = np.gradient(ug, dx, dy, axis=(1, 0))
    dvdx, dvdy = np.gradient(vg, dx, dy, axis=(1, 0))

    w = dvdx - dudy
    div = dudx + dvdy
    adv_e = ug * dudx + vg * dudy
    adv_n = ug * dvdx + vg * dvdy

    ds["w"] = ug
    ds["div"] = ug
    ds["adv_e"] = ug
    ds["adv_n"] = ug

    ds.div.data = div
    ds.div.attrs["long_name"] = "sea surface divergence"
    ds.div.attrs["units"] = "1/s"

    ds.adv_e.data = adv_e
    ds.adv_e.attrs["long_name"] = "sea surface eastward advection"
    ds.adv_e.attrs["units"] = "cm/s$^2$"

    ds.adv_n.data = adv_n
    ds.adv_n.attrs["long_name"] = "sea surface northward advection"
    ds.adv_n.attrs["units"] = "cm/s$^2$"

    ds.w.data = w
    ds.w.attrs["long_name"] = "sea surface vorticity"
    ds.w.attrs["units"] = "1/s"

    return ds


def preprocess_dataset(
    ds, latmin=40, latmax=50, lonmin=170, lonmax=230, dx=2000, dy=2000
):
    ds.simulated_true_ssh_karin.attrs["units"] = "cm"
    ds_cut = ds.where(
        (ds.latitude > latmin)
        & (ds.latitude < latmax)
        & (ds.longitude < lonmax)
        & (ds.longitude > lonmin),
        drop=True,
    )
    ds_vels = geostrophic_velocity(ds_cut, dx=dx, dy=dy)
    ds_rot = rotational_parameters(ds_vels, dx=dx, dy=dy)

    return ds_rot


def postprocess_dataset(sfs, dict=True):
    (
        mean_SF_velocity_meridional,
        mean_SF_velocity_zonal,
        mean_SF_3rd_velocity_meridional,
        mean_SF_3rd_velocity_zonal,
    ) = SFMean(sfs, dict)
    (
        SF_velocity_zonals,
        SF_velocity_meridionals,
        SF_3rd_velocity_zonals,
        SF_3rd_velocity_meridionals,
    ) = ReformatSF(sfs, dict)
    boot_SF_vz, boot_SF_vm, boot_SF_3rd_vz, boot_SF_3rd_vm = BootstrapSF(
        SF_velocity_zonals,
        SF_velocity_meridionals,
        SF_3rd_velocity_zonals,
        SF_3rd_velocity_meridionals,
    )
    return (mean_SF_velocity_zonal, mean_SF_velocity_meridional, boot_SF_vz, boot_SF_vm)


def adv_sf(ds_dict, latmin, latmax, lonmin, lonmax, numl=None):
    processed_data = []
    sfs = []
    for ds_name in ds_dict:

        ds = ds_dict[ds_name]

        try:
            ds = preprocess_dataset(ds, latmin, latmax, lonmin, lonmax)
            processed_data.append(ds)

            SF_adv = ocsf.advection_velocity(
                (ds.u.values * (u.cm / u.s)).si.value,
                (ds.v.values * (u.cm / u.s)).si.value,
                2000 * ds.num_lines.values,
                2000 * ds.num_pixels.values,
                even=False,
                boundary=None,
                grid_type=None,
                nbins=len(ds.num_pixels),
            )
            sfs.append(SF_adv)

        except ValueError:
            pass
    return (sfs, processed_data)


def trad_sf(ds_dict, latmin, latmax, lonmin, lonmax, order=3, numl=None):
    processed_data = []
    sfs = []
    for ds_name in ds_dict:

        ds = ds_dict[ds_name]

        try:
            ds = preprocess_dataset(ds, latmin, latmax, lonmin, lonmax)
            processed_data.append(ds)

            SF_trad = ocsf.traditional_velocity(
                (ds.u.values * (u.cm / u.s)).si.value,
                (ds.v.values * (u.cm / u.s)).si.value,
                2000 * ds.num_lines.values,
                2000 * ds.num_pixels.values,
                even=False,
                boundary=None,
                grid_type=None,
                nbins=len(ds.num_pixels),
                order=order,
            )
            sfs.append(SF_trad)

        except:
            traceback.print_exc()
            pass
    return (sfs, processed_data)


def load_data(filepath):
    filelist = glob.glob(filepath)
    ds_dict = {}

    for f in filelist:
        key = os.path.basename(f)
        ds_dict[key] = xr.open_dataset(f)

    return ds_dict


def sf_run(
    ds_dict,
    SFtype="trad",
    figname="trad.png",
    ymin=-1e-15,
    ymax=1e-15,
    dpi=300,
    latmin=40,
    latmax=50,
    lonmin=170,
    lonmax=230,
    order=3,
    numl=None,
):

    if SFtype == "trad":
        sfs, processed_ds = trad_sf(
            ds_dict, latmin, latmax, lonmin, lonmax, order, numl
        )
        title = "Traditional velocity dissipation rate"
    else:
        sfs, processed_ds = adv_sf(ds_dict, latmin, latmax, lonmin, lonmax, numl)
        title = "Advection velocity dissipation rate"

    SF_z = []
    SF_m = []

    for sf in sfs:
        SF_z.append(sf["SF_zonal_uneven"])
        SF_m.append(sf["SF_meridional"])

    SF_z = np.asarray([a[: len(min(SF_z, key=len))] for a in SF_z])
    SF_m = np.asarray([a[: len(min(SF_m, key=len))] for a in SF_m])

    boot_SF_vz = bootstrap((SF_z,), np.mean, confidence_level=0.9, axis=0)
    boot_SF_vm = bootstrap((SF_m,), np.mean, confidence_level=0.9, axis=0)

    boot_SF_vz_conf = boot_SF_vz.confidence_interval
    boot_SF_vm_conf = boot_SF_vm.confidence_interval

    boot_SF_vz_mean = boot_SF_vz.bootstrap_distribution.mean(axis=1)
    boot_SF_vm_mean = boot_SF_vm.bootstrap_distribution.mean(axis=1)

    xd_uneven = sfs[0]["x-diffs_uneven"][: len(boot_SF_vm_mean)]
    yd_uneven = sfs[0]["y-diffs_uneven"][: len(boot_SF_vz_mean)]

    fig, (ax1) = plt.subplots(figsize=(10, 7))

    if SFtype == "trad":
        flux_sf_figures.SF_bootstrap_plot(
            (2 / 3) * boot_SF_vz_mean / yd_uneven,
            (2 / 3) * boot_SF_vm_mean / xd_uneven,
            yd_uneven,
            xd_uneven,
            bootz0=(2 / 3) * boot_SF_vz_conf[0] / yd_uneven,
            bootz1=(2 / 3) * boot_SF_vz_conf[1] / yd_uneven,
            bootm0=(2 / 3) * boot_SF_vm_conf[0] / xd_uneven,
            bootm1=(2 / 3) * boot_SF_vm_conf[1] / xd_uneven,
            title=title,
            label1="Across-track",
            label2="Along-track",
            ax=ax1,
        )

    else:
        flux_sf_figures.SF_bootstrap_plot(
            boot_SF_vz_mean / 2,
            boot_SF_vm_mean / 2,
            yd_uneven,
            xd_uneven,
            bootz0=boot_SF_vz_conf[0] / 2,
            bootz1=boot_SF_vz_conf[1] / 2,
            bootm0=boot_SF_vm_conf[0] / 2,
            bootm1=boot_SF_vm_conf[1] / 2,
            title=title,
            label1="Across-track",
            label2="Along-track",
            ax=ax1,
        )

    ax1.tick_params(direction="in", which="both")
    ax1.xaxis.get_ticklocs(minor=True)
    ax1.set_ylim(ymin, ymax)
    ax1.minorticks_on()
    ax1.set_title(title)
    fig.savefig(fname=figname, dpi=dpi)

    return (SF_z, SF_m, xd_uneven, yd_uneven, boot_SF_vz, boot_SF_vm)
