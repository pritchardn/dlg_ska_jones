import logging
import pickle
import sys
import time
from typing import NoReturn

import astropy.constants as consts
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
from dlg import droputils

matplotlib.use("Agg")
import numpy as np
import xarray
from astropy.coordinates import SkyCoord
from astropy.time import Time
from dlg.drop import BarrierAppDROP
from dlg.meta import (
    dlg_component,
    dlg_batch_input,
    dlg_streaming_input,
    dlg_batch_output,
    dlg_float_param,
    dlg_int_param,
    dlg_list_param,
)
from jones_solvers.processing_components import solve_jones
from numpy import cos as cos
from numpy import sin as sin
from rascil.data_models import (
    PolarisationFrame,
    BlockVisibility,
    Configuration,
    GainTable,
)
from rascil.processing_components import create_blockvisibility
from rascil.processing_components import create_named_configuration
from rascil.processing_components.calibration.operations import (
    create_gaintable_from_blockvisibility,
)
from rascil.processing_components.util.coordinate_support import lmn_to_skycoord
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

np.set_printoptions(linewidth=-1)

log.info("Init blockvisibility")


def create_phasecentre(ra: float, dec: float) -> SkyCoord:
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs", equinox="J2000")


def generate_frequency_range(nchannels: int, channel_bandwidth: float) -> np.ndarray:
    return np.arange(
        100.0e6, 100.0e6 + nchannels * channel_bandwidth, channel_bandwidth
    )


def filter_visibilities(
        blockvisibilities: BlockVisibility, subarray: list
) -> (BlockVisibility, np.ndarray):
    subarray = (
            np.array(subarray) - 1
    )  # the two most longitudinally separated stations in each cluster

    model_vis = blockvisibilities.where(
        blockvisibilities["antenna1"].isin(subarray)
        * blockvisibilities["antenna2"].isin(subarray),
        drop=True,
    )
    return model_vis, subarray


def setup_blockvisibility(
        lowconfig: Configuration,
        times: np.ndarray,
        frequency: np.ndarray,
        channel_bandwidth: np.ndarray,
        phasecentre: SkyCoord,
        sample_time: float,
) -> BlockVisibility:
    # create empty blockvis with intrumental polarisation (XX, XY, YX, YY)
    model_vis = create_blockvisibility(
        lowconfig,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phasecentre,
        sample_time=sample_time,
        polarisation_frame=PolarisationFrame("linear"),
        weight=1.0,
    )

    return model_vis


def skymodel_single_source(
        nchannels: int,
) -> (int, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    nsrc = 1
    jy = 50 * np.ones((nsrc, nchannels))
    l = np.zeros(nsrc)
    m = np.zeros(nsrc)
    n = np.zeros(nsrc)
    return nsrc, jy, l, m, n


def skymodel_random_sources(
        dist_source_multiplier: float, frequency: np.ndarray, nsrc: int
) -> (int, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    dist_source_max = dist_source_multiplier * np.pi / 180.0
    # sky model: randomise sources across the field
    theta = 2.0 * np.pi * np.random.rand(nsrc)
    phi = dist_source_max * np.sqrt(np.random.rand(nsrc))
    l = sin(theta) * sin(phi)
    m = cos(theta) * sin(phi)
    n = np.sqrt(1 - l * l - m * m) - 1

    spec_index_mult = (frequency[np.newaxis, :] / frequency[0]) ** (-0.8)
    jy = 10 * np.random.rand(nsrc, 1) @ spec_index_mult
    return nsrc, jy, l, m, n


def skymodel_real_sources(
        frequency: np.ndarray,
        ra_hrs: np.ndarray,
        dec_deg: np.ndarray,
        jy_240MHz: np.ndarray,
) -> (int, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    ra0_hrs = 0.0
    dec0_deg = -27.0
    ra = ra_hrs * np.pi / 12.0
    dec = dec_deg * np.pi / 180.0
    ra0 = ra0_hrs * np.pi / 12.0
    dec0 = dec0_deg * np.pi / 180.0
    cdec0 = np.cos(dec0)
    sdec0 = np.sin(dec0)
    cdec = np.cos(dec)
    sdec = np.sin(dec)
    cdra = np.cos((ra - ra0))
    sdra = np.sin((ra - ra0))
    l = cdec * sdra
    m = sdec * cdec0 - cdec * sdec0 * cdra
    n = sdec * sdec0 + cdec * cdec0 * cdra
    nsrc = len(jy_240MHz)

    spec_index_mult = (frequency[np.newaxis, :] / 240e6) ** (-0.8)
    jy = jy_240MHz[:, np.newaxis] * spec_index_mult
    return nsrc, jy, l, m, n


def setup_skymodel(
        sky_mode: int,
        nsrc: int,
        nchannels: int,
        frequency: np.ndarray,
        ra_hrs: np.ndarray = None,
        dec_deg: np.ndarray = None,
        jy_240MHz: np.ndarray = None,
) -> (int, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    if sky_mode == 0:
        return skymodel_single_source(nchannels)
    elif sky_mode == 1:
        return skymodel_random_sources(0.5, frequency, nsrc)
    elif sky_mode == 2:
        return skymodel_random_sources(2.5, frequency, nsrc)
    elif sky_mode == 3:
        return skymodel_real_sources(frequency, ra_hrs, dec_deg, jy_240MHz)
    else:
        raise ValueError("Unknown sky_mode index")


def calculate_flux_densities(
        lon: float,
        lat: float,
        nchannels: int,
        nsrc: int,
        model_vis: xarray.Dataset,
        phasecentre: SkyCoord,
        frequency: np.ndarray,
        jy: np.ndarray,
        l: np.ndarray,
        m: np.ndarray,
        n: np.ndarray,
) -> (xarray.Dataset, xarray.Dataset, np.ndarray):
    # make a full copy of the model for the actual visibilities.
    # The sky and cal models could be different...
    noiselessVis = model_vis.copy(deep=True)

    is_cal = np.zeros(nsrc, "bool")

    for src in range(0, nsrc):

        # analytic response of short dipoles aligned NS & EW to sky xy polarisations
        # with an approx Gaussian taper for a 35m station
        srcdir = lmn_to_skycoord(np.array([l[src], m[src], n[src]]), phasecentre)
        ra = srcdir.ra.value * np.pi / 180.0
        dec = srcdir.dec.value * np.pi / 180.0
        sep = srcdir.separation(phasecentre).value * np.pi / 180.0
        diam = 35.0

        # estimate the apparent flux density
        wl = consts.c.value / np.mean(frequency)
        sigma = wl / diam / 2.355
        gain = np.exp(-sep * sep / (2 * sigma * sigma))
        if gain * jy[src, 0] > 1e-3:
            is_cal[src] = True

        # need to set ha,dec, but need to be in time,freq loop
        for t in range(0, len(model_vis["datetime"])):

            utc_time = model_vis["datetime"].data[t, 0]
            # azel = calculate_azel(location, utc_time, srcdir)
            lst = (
                    Time(utc_time, location=(lon * u.rad, lat * u.rad))
                    .sidereal_time("mean")
                    .value
                    * np.pi
                    / 12.0
            )
            ha = lst - ra

            J00 = cos(lat) * cos(dec) + sin(lat) * sin(dec) * cos(ha)
            J01 = -sin(lat) * sin(ha)
            J10 = sin(dec) * sin(ha)
            J11 = cos(ha)
            J = np.array([[J00, J01], [J10, J11]], "complex")
            # components are unpolarised, so can form power product now
            JJ = J * J.conj().T

            for f in range(0, nchannels):

                wl = consts.c.value / frequency[f]
                sigma = wl / diam / 2.355
                gain = np.exp(-sep * sep / (2 * sigma * sigma))

                srcbeam = JJ * gain

                # vis (time, baselines, frequency, polarisation) complex128

                uvw = model_vis["uvw_lambda"].data[t, :, f, :]
                phaser = (
                        0.5
                        * jy[src, f]
                        * np.exp(
                    2j
                    * np.pi
                    * (uvw[:, 0] * l[src] + uvw[:, 1] * m[src] + uvw[:, 2] * n[src])
                )
                )

                assert all(
                    model_vis["polarisation"].data == ["XX", "XY", "YX", "YY"]
                ), "pol error"

                noiselessVis["vis"].data[t, :, f, 0] += phaser * srcbeam[0, 0]
                noiselessVis["vis"].data[t, :, f, 1] += phaser * srcbeam[0, 1]
                noiselessVis["vis"].data[t, :, f, 2] += phaser * srcbeam[1, 0]
                noiselessVis["vis"].data[t, :, f, 3] += phaser * srcbeam[1, 1]

                if is_cal[src]:
                    model_vis["vis"].data[t, :, f, 0] += phaser * srcbeam[0, 0]
                    model_vis["vis"].data[t, :, f, 1] += phaser * srcbeam[0, 1]
                    model_vis["vis"].data[t, :, f, 2] += phaser * srcbeam[1, 0]
                    model_vis["vis"].data[t, :, f, 3] += phaser * srcbeam[1, 1]
    return model_vis, noiselessVis, is_cal


def solve_jones_trial(
        gt_fit: GainTable,
        model_vis: xarray.Dataset,
        noiseless_vis: xarray.Dataset,
        algorithm: int,
        lin_solver: str = "lsmr",
        lin_solve_normal: bool = None,
        rcond: float = 1e-6,
        tolerance: float = 1e-6,
        niter: int = 50,
) -> (np.ndarray, GainTable):
    gt_fit_copy = gt_fit.copy(deep=True)
    model_vis_copy = model_vis.copy(deep=True)
    observed_vis = noiseless_vis.copy(deep=True)
    t0 = time.time()
    chi_sq = solve_jones(
        observed_vis,
        model_vis_copy,
        gt_fit_copy,
        testvis=noiseless_vis,
        niter=niter,
        algorithm=algorithm,
        lin_solver_normal=lin_solve_normal,
        lin_solver=lin_solver,
        lin_solver_rcond=rcond,
    )
    return chi_sq, gt_fit_copy


def update_visibilities(
        nchannels: int,
        sample_time: float,
        nsamples: int,
        channel_bandwidth: np.ndarray,
        frequency: np.ndarray,
        model_vis: xarray.Dataset,
        noiselessVis: xarray.Dataset,
        subarray: np.ndarray,
) -> (GainTable, GainTable, np.ndarray):
    # Some RASCIL functions to look into using
    # gt_true = simulate_gaintable(modelVis, phase_error=1.0, amplitude_error=0.1, leakage=0.1)
    # gt_fit  = simulate_gaintable(modelVis, phase_error=0.0, amplitude_error=0.0, leakage=0.0)
    # generate a gaintable with a single timeslice
    # (is in sec, so should be > 43200 for a 12 hr observation)
    # could alternatively just use the first time step in the call
    # "ValueError: Unknown Jones type P"
    nsubarray = len(subarray)
    nvis = model_vis["baselines"].shape[0]
    gt_true = create_gaintable_from_blockvisibility(
        model_vis, timeslice=1e6, jones_type="G"
    )
    gt_fit = create_gaintable_from_blockvisibility(
        model_vis, timeslice=1e6, jones_type="G"
    )
    # set up references to the data
    Jt = gt_true["gain"].data
    Jm = gt_fit["gain"].data
    # only both setting gains for stations that are in the subarray
    for idx in range(0, nsubarray):
        stn = subarray[idx]

        # generate the starting model station gain error matrices. The same for all tests
        Jm[0, stn, 0, :, :] = np.eye(2, dtype=complex)

        # generate the true station gain error matrices
        #  - set to model matrices plus some Gaussian offsets
        # Jsigma = 0.1
        # Jt[0,stn,0,:,:] = Jm[0,stn,0,:,:] + Jsigma * ( np.random.randn(2,2) + 1j*np.random.randn(2,2) )
        #  - set to model matrices plus some systematic offsets and Gaussian noise
        Gsigma = 0.1
        gX = np.exp(-0.0j) + Gsigma * (np.random.randn() + 1j * np.random.randn())
        gY = np.exp(+0.1j) + Gsigma * (np.random.randn() + 1j * np.random.randn())
        Dsigma = 0.01
        dXY = +0.05 + Dsigma * (np.random.randn() + 1j * np.random.randn())
        dYX = +0.05 + Dsigma * (np.random.randn() + 1j * np.random.randn())
        Jt[0, stn, 0, :, :] = np.array([[gX, gX * dXY], [-gY * dYX, gY]])
    # Apply calibration factors
    # set up references to the data
    stn1 = model_vis["antenna1"].data
    stn2 = model_vis["antenna2"].data
    for t in range(0, len(model_vis["datetime"])):
        for f in range(0, nchannels):

            # set up references to the data
            modelTmp = model_vis["vis"].data[t, :, f, :]
            noiselessTmp = noiselessVis["vis"].data[t, :, f, :]

            for k in range(0, nvis):
                vis_in = np.reshape(modelTmp[k, :], (2, 2))
                vis_out = Jm[0, stn1[k], 0] @ vis_in @ Jm[0, stn2[k], 0].conj().T
                modelTmp[k, :] = np.reshape(np.array(vis_out), (4))

                vis_in = np.reshape(noiselessTmp[k, :], (2, 2))
                vis_out = Jt[0, stn1[k], 0] @ vis_in @ Jt[0, stn2[k], 0].conj().T
                noiselessTmp[k, :] = np.reshape(np.array(vis_out), (4))
    # Add noise to a visibility
    # RMS of vis noise (Braun, R., 2013, Understanding Synthesis Imaging Dynamic Range. A&A, 551:A91)
    #  - these are pretty close to the numbers from Sokolowski et al. 2022, PASA 39
    # wl = consts.c.value / np.mean(frequency)
    # T_sys = 150. + 60.*wl**2.55
    # A_eff = 2.*256.*wl**(2./3.)
    # #SEFD = 2.*1.38e-23*T_sys/A_eff * 1e26
    # from skalowsensitivitybackup-env.eba-daehsrjt.ap-southeast-2.elasticbeanstalk.com/sensitivity_radec_vs_freq
    #  - AAVS2_sensitivity_ra0.00deg_dec_-27.00deg_0.00hours.txt
    #  - Sokolowski et al. 2022, PASA 39
    sim_freq = (
            np.array(
                [
                    99.8400,
                    108.8000,
                    119.0400,
                    129.2800,
                    139.5200,
                    149.7600,
                    154.8800,
                    160.0000,
                    168.9600,
                    179.2000,
                    185.6000,
                    189.4400,
                    199.6800,
                ]
            )
            * 1e6
    )
    sim_SEFDx = np.array(
        [
            2371.964004,
            2155.226369,
            2025.520665,
            1914.953382,
            1827.958458,
            1818.613829,
            1872.251517,
            1940.699453,
            2012.247193,
            2120.719450,
            2123.762506,
            2092.097418,
            2047.851280,
        ]
    )
    sim_SEFDy = np.array(
        [
            2373.570152,
            2162.044465,
            2100.491293,
            2052.820744,
            1958.236742,
            1998.655397,
            1977.974041,
            2078.091533,
            2257.589304,
            2390.946732,
            2376.884877,
            2347.302348,
            2236.785787,
        ]
    )
    SEFD_fit = interp1d(sim_freq, sim_SEFDx, kind="cubic")
    SEFD = SEFD_fit(frequency)
    sigma_calc = SEFD / np.sqrt(2.0 * channel_bandwidth * sample_time)
    print("Noise estimate:")
    print(" - SEFD range = {:6.1f} - {:6.1f} Jy".format(np.min(SEFD), np.max(SEFD)))
    print(
        " - sigma range = {:4.2f} - {:4.2f} Jy".format(
            np.min(sigma_calc), np.max(sigma_calc)
        )
    )
    sigma = sigma_calc
    # sigma = sigma_calc * 1e-3
    # Some RASCIL functions to look into using
    # calculate_noise_blockvisibility(bandwidth, ...)
    # addnoise_visibility(vis[, t_sys, eta, seed])
    observedVis = noiselessVis.copy(deep=True)
    shape = observedVis["vis"].shape
    assert len(shape) == 4, "require 4 dimensions for blockvisibilty"
    assert shape[0] == nsamples, "unexpected time dimension"
    assert shape[2] == nchannels, "unexpected frequency dimension"
    for f in range(0, nchannels):
        observedVis["vis"].data[:, :, f, :] += sigma[f] * (
                np.random.randn(shape[0], shape[1], shape[3])
                + np.random.randn(shape[0], shape[1], shape[3]) * 1j
        )
        if sigma[f] > 0:
            model_vis["weight"].data[:, :, f, :] *= 1.0 / (sigma[f] * sigma[f])
            observedVis["weight"].data[:, :, f, :] *= 1.0 / (sigma[f] * sigma[f])
    return gt_fit, gt_true, sigma


def plot_visibilities(is_cal: np.ndarray, l: np.ndarray, m: np.ndarray,
                      fileprefix: str) -> NoReturn:
    plt.figure(num=0, figsize=(8, 8), facecolor="w", edgecolor="k")
    plt.subplot(111, aspect="equal")
    plt.plot(
        np.arcsin(l) * 180 / np.pi,
        np.arcsin(m) * 180 / np.pi,
        "c.",
        label="sky model components",
    )
    plt.plot(
        np.arcsin(l[is_cal]) * 180 / np.pi,
        np.arcsin(m[is_cal]) * 180 / np.pi,
        "r*",
        label="cal model components",
    )
    phi = np.arange(0, 2 * np.pi, np.pi / 50.0)
    r = 2.5
    plt.plot(r * cos(phi), r * sin(phi), "r", label=r"$5^\circ$")
    plt.xlabel("sin$^{-1}(l)$ deg", fontsize=14)
    plt.ylabel("sin$^{-1}(m)$ deg", fontsize=14)
    plt.legend(fontsize=12, frameon=False)
    plt.savefig(f"{fileprefix}-blockvisibilities.png")
    plt.clf()


def log_timings(
        chisq1,
        chisq2,
        chisq2a,
        chisq2b,
        chisq2c,
        show1,
        show2,
        show2a,
        show2b,
        show2c,
        t_fillvis,
        t_initvis,
        t_solving1,
        t_solving2,
        t_solving2a,
        t_solving2b,
        t_solving2c,
        t_updatevis,
) -> NoReturn:
    fstr = " - {:<35} {:6.1f} sec"
    log.info("")
    log.info("Timing:")
    log.info(fstr.format("init blockvis", t_initvis))
    log.info(fstr.format("predict blockvis", t_fillvis))
    log.info(fstr.format("apply corruptions", t_updatevis))
    if show1:
        tstr = fstr.format("Alg 1 with defaults", t_solving1)
        if len(chisq1) > 0:
            tstr += " for {} iterations".format(len(chisq1))
        log.info(tstr)
    if show2:
        tstr = fstr.format("Alg 2 with default lsmr", t_solving2)
        if len(chisq2) > 0:
            tstr += " for {} iterations".format(len(chisq2))
        log.info(tstr)
    if show2a:
        tstr = fstr.format("Alg 2 with lsmr & lin_solver_normal", t_solving2a)
        if len(chisq2a) > 0:
            tstr += " for {} iterations".format(len(chisq2a))
        log.info(tstr)
    if show2b:
        tstr = fstr.format("Alg 2 with lstsq, rcond = 1e-6", t_solving2b)
        if len(chisq2b) > 0:
            tstr += " for {} iterations".format(len(chisq2b))
        log.info(tstr)
    if show2c:
        tstr = fstr.format("Alg 2 with lstsq, rcond = 1e-4", t_solving2c)
        if len(chisq2c) > 0:
            tstr += " for {} iterations".format(len(chisq2c))
        log.info(tstr)
    log.info("")


def plot_results(
        J1: np.ndarray,
        J2: np.ndarray,
        J2a: np.ndarray,
        J2b: np.ndarray,
        J2c: np.ndarray,
        Jt: np.ndarray,
        show1: bool,
        show2: bool,
        show2a: bool,
        show2b: bool,
        show2c: bool,
        subarray: np.ndarray,
        fileprefix: str,
) -> NoReturn:
    nsubarray = len(subarray)
    plt.figure(num=1, figsize=(20, 12), facecolor="w", edgecolor="k")
    ax241 = plt.subplot(241)
    ax241.set_title("real(J[0,0])", fontsize=16)
    ax242 = plt.subplot(242)
    ax242.set_title("real(J[0,1])", fontsize=16)
    ax243 = plt.subplot(243)
    ax243.set_title("real(J[1,0])", fontsize=16)
    ax244 = plt.subplot(244)
    ax244.set_title("real(J[1,1])", fontsize=16)
    ax245 = plt.subplot(245)
    ax245.set_title("imag(J[0,0])", fontsize=16)
    ax246 = plt.subplot(246)
    ax246.set_title("imag(J[0,1])", fontsize=16)
    ax247 = plt.subplot(247)
    ax247.set_title("imag(J[1,0])", fontsize=16)
    ax248 = plt.subplot(248)
    ax248.set_title("imag(J[1,1])", fontsize=16)
    ax241.set_xlabel("array index", fontsize=14)
    ax241.grid()
    ax242.set_xlabel("array index", fontsize=14)
    ax242.grid()
    ax243.set_xlabel("array index", fontsize=14)
    ax243.grid()
    ax244.set_xlabel("array index", fontsize=14)
    ax244.grid()
    ax245.set_xlabel("array index", fontsize=14)
    ax245.grid()
    ax246.set_xlabel("array index", fontsize=14)
    ax246.grid()
    ax247.set_xlabel("array index", fontsize=14)
    ax247.grid()
    ax248.set_xlabel("array index", fontsize=14)
    ax248.grid()

    def plot_gain(J, col, label=""):
        log.info(label)
        pref = np.exp(-1j * np.angle(J[0][0, 0]))
        Jref = np.zeros((nsubarray, 2, 2), "complex")
        for stn in range(nsubarray):
            Jref[stn] = J[stn] * pref
        ax241.plot(np.real(Jref[:, 0, 0]), col, label=label)
        ax242.plot(np.real(Jref[:, 0, 1]), col)
        ax243.plot(np.real(Jref[:, 1, 0]), col)
        ax244.plot(np.real(Jref[:, 1, 1]), col)
        ax245.plot(np.imag(Jref[:, 0, 0]), col)
        ax246.plot(np.imag(Jref[:, 0, 1]), col)
        ax247.plot(np.imag(Jref[:, 1, 0]), col)
        ax248.plot(np.imag(Jref[:, 1, 1]), col)

    plot_gain(Jt, "k-", "True gain errors")
    if show1:
        plot_gain(J1, "r-", "Alg 1 with defaults")
    if show2:
        plot_gain(J2, "m-", "Alg 2 with default lsmr")
    if show2a:
        plot_gain(J2a, "g-", "Alg 2 with lsmr & lin_solver_normal")
    if show2b:
        plot_gain(J2b, "b--", "Alg 2 with lstsq, rcond = 1e-6")
    if show2c:
        plot_gain(J2c, "c-", "Alg 2 with lstsq, rcond = 1e-4")
    ax241.legend(fontsize=10)
    plt.savefig(f"{fileprefix}-solver_results.png")
    plt.clf()


def plot_solver_error(
        J1: np.ndarray,
        J2: np.ndarray,
        J2a: np.ndarray,
        J2b: np.ndarray,
        J2c: np.ndarray,
        Jt: np.ndarray,
        show1: bool,
        show2: bool,
        show2a: bool,
        show2b: bool,
        show2c: bool,
        subarray: np.ndarray,
        fileprefix: str,
) -> NoReturn:
    nsubarray = len(subarray)
    plt.figure(num=2, figsize=(20, 12), facecolor="w", edgecolor="k")
    ax241 = plt.subplot(241)
    ax241.set_title("real(U[0,0])", fontsize=16)
    ax242 = plt.subplot(242)
    ax242.set_title("real(U[0,1])", fontsize=16)
    ax243 = plt.subplot(243)
    ax243.set_title("real(U[1,0])", fontsize=16)
    ax244 = plt.subplot(244)
    ax244.set_title("real(U[1,1])", fontsize=16)
    ax245 = plt.subplot(245)
    ax245.set_title("imag(U[0,0])", fontsize=16)
    ax246 = plt.subplot(246)
    ax246.set_title("imag(U[0,1])", fontsize=16)
    ax247 = plt.subplot(247)
    ax247.set_title("imag(U[1,0])", fontsize=16)
    ax248 = plt.subplot(248)
    ax248.set_title("imag(U[1,1])", fontsize=16)
    ax241.set_xlabel("array index", fontsize=14)
    ax241.grid()
    ax242.set_xlabel("array index", fontsize=14)
    ax242.grid()
    ax243.set_xlabel("array index", fontsize=14)
    ax243.grid()
    ax244.set_xlabel("array index", fontsize=14)
    ax244.grid()
    ax245.set_xlabel("array index", fontsize=14)
    ax245.grid()
    ax246.set_xlabel("array index", fontsize=14)
    ax246.grid()
    ax247.set_xlabel("array index", fontsize=14)
    ax247.grid()
    ax248.set_xlabel("array index", fontsize=14)
    ax248.grid()

    def plot_ambiguity(J, Jt, nsubarray, col, label=""):
        log.info(label)
        U0 = J[0] @ np.linalg.inv(Jt[0])
        pref = np.exp(-1j * np.angle(U0[0, 0]))
        U = np.zeros((nsubarray, 2, 2), "complex")
        for stn in range(nsubarray):
            U[stn] = J[stn] @ np.linalg.inv(Jt[stn]) * pref
            if stn < 3:
                fstr = "({0.real:+7.4f}{0.imag:+7.4f}i)"
                log.info(
                    "UrefXX[{}] [[".format(stn)
                    + fstr.format(U[stn, 0, 0])
                    + ","
                    + fstr.format(U[stn, 0, 1])
                    + "],["
                    + fstr.format(U[stn, 1, 0])
                    + ","
                    + fstr.format(U[stn, 1, 1])
                    + "]]"
                )
        ax241.plot(np.real(U[:, 0, 0]), col, label=label)
        ax242.plot(np.real(U[:, 0, 1]), col)
        ax243.plot(np.real(U[:, 1, 0]), col)
        ax244.plot(np.real(U[:, 1, 1]), col)
        ax245.plot(np.imag(U[:, 0, 0]), col)
        ax246.plot(np.imag(U[:, 0, 1]), col)
        ax247.plot(np.imag(U[:, 1, 0]), col)
        ax248.plot(np.imag(U[:, 1, 1]), col)

    ax241.set_ylim((-0.3, +1.3))
    ax242.set_ylim((-0.3, +0.3))
    ax243.set_ylim((-0.3, +0.3))
    ax244.set_ylim((-0.3, +1.3))
    ax245.set_ylim((-0.3, +0.3))
    ax246.set_ylim((-0.3, +0.3))
    ax247.set_ylim((-0.3, +0.3))
    ax248.set_ylim((-0.3, +0.3))
    if show1:
        plot_ambiguity(J1, Jt, nsubarray, "r-", "Alg 1 with defaults")
    if show2:
        plot_ambiguity(J2, Jt, nsubarray, "m-", "Alg 2 with default lsmr")
    if show2a:
        plot_ambiguity(J2a, Jt, nsubarray, "g-", "Alg 2 with lsmr & lin_solver_normal")
    if show2b:
        plot_ambiguity(J2b, Jt, nsubarray, "b--", "Alg 2 with lstsq, rcond = 1e-6")
    if show2c:
        plot_ambiguity(J2c, Jt, nsubarray, "c-", "Alg 2 with lstsq, rcond = 1e-4")
    ax241.legend(fontsize=10)
    plt.savefig(f"{fileprefix}-solver_error.png")
    plt.clf()


def run_trials(
        gt_fit: GainTable,
        gt_true: GainTable,
        model_vis: xarray.Dataset,
        noiselessVis: xarray.Dataset,
        subarray: np.ndarray,
        fileprefix: str,
) -> (
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        bool,
        bool,
        bool,
        bool,
        bool,
        float,
        float,
        float,
        float,
        float,
):
    show1 = True
    show2 = False
    show2a = True
    show2b = True
    show2c = True
    chisq1 = None
    chisq2 = None
    chisq2a = None
    chisq2b = None
    chisq2c = None
    t_solving1 = None
    t_solving2 = None
    t_solving2a = None
    t_solving2b = None
    t_solving2c = None
    if show1:
        log.info("Running algorithm 1 with defaults")
        t0 = time.time()
        chisq1, gt1 = solve_jones_trial(gt_fit, model_vis, noiselessVis, 1)
        t_solving1 = time.time() - t0
    if show2:
        log.info("Running algorithm 2 with defaults")
        t0 = time.time()
        chisq2, gt2 = solve_jones_trial(gt_fit, model_vis, noiselessVis, algorithm=2)
        t_solving2 = time.time() - t0
    if show2a:
        log.info("Running algorithm 2 with lin_solver_normal")
        t0 = time.time()
        chisq2a, gt2a = solve_jones_trial(
            gt_fit,
            model_vis,
            noiselessVis,
            algorithm=2,
            lin_solve_normal=True,
        )
        t_solving2a = time.time() - t0
    if show2b:
        log.info("Running algorithm 2 with lin_solver=lstsq")
        t0 = time.time()
        chisq2b, gt2b = solve_jones_trial(
            gt_fit,
            model_vis,
            noiselessVis,
            algorithm=2,
            lin_solver="lstsq",
        )
        t_solving2b = time.time() - t0
    if show2c:
        log.info("Running algorithm 2 with lin_solver=lstsq & rcond=1e-4")
        t0 = time.time()
        chisq2c, gt2c = solve_jones_trial(
            gt_fit,
            model_vis,
            noiselessVis,
            algorithm=2,
            lin_solver="lstsq",
            rcond=1e-4,
        )
        t_solving2c = time.time() - t0
    # copy gain data for the subarray
    J1 = None
    J2 = None
    J2a = None
    J2b = None
    J2c = None
    Jt = gt_true["gain"].data[0, subarray, 0, :, :]
    if show1:
        J1 = gt1["gain"].data[0, subarray, 0, :, :]
    if show2:
        J2 = gt2["gain"].data[0, subarray, 0, :, :]
    if show2a:
        J2a = gt2a["gain"].data[0, subarray, 0, :, :]
    if show2b:
        J2b = gt2b["gain"].data[0, subarray, 0, :, :]
    if show2c:
        J2c = gt2c["gain"].data[0, subarray, 0, :, :]
    return (
        J1,
        J2,
        J2a,
        J2b,
        J2c,
        Jt,
        chisq1,
        chisq2,
        chisq2a,
        chisq2b,
        chisq2c,
        show1,
        show2,
        show2a,
        show2b,
        show2c,
        t_solving1,
        t_solving2,
        t_solving2a,
        t_solving2b,
        t_solving2c,
    )


def plot_error_compare(
        chisq1: np.ndarray,
        chisq2: np.ndarray,
        chisq2b: np.ndarray,
        model_vis: xarray.Dataset,
        show1: bool,
        show2: bool,
        show2b: bool,
        sigma: np.ndarray,
        subarray: np.ndarray,
        nsamples: int,
        nchannels: int,
        fileprefix: str,
) -> NoReturn:
    plt.figure(num=4, figsize=(14, 8), facecolor="w", edgecolor="k")
    # back of the envelope estimate of the error RMS level.
    # gi_est ~ sum_j((Vij+error)*Mij - Mij*Mij) / sum_j(Mij*Mij)
    # gi_error ~ sum_j(error*Mij) / sum_j(Mij*Mij)
    # g_sigma ~ sqrt( sigma**2 / sum_j(Mij*Mij) )
    # vij ~ (1 + gi_error)*(1 + gj_error)*Mij
    # vij_error ~ Mij*gj_error + Mij*gi_error
    # vij_sigma ~ sqrt(2 * mean(Mij**2)) * g_sigma
    # chisq ~ mean( (vij_error)**2 )
    # really need to do the frequency averaging properly here
    Mij = model_vis.sel({"antenna1": subarray[0]})["vis"].data[0, 1::, 0, 0]
    # could use the mean of all vis and multiply by nstn-1. Careful of autos though
    g_sigma = np.sqrt(
        sigma ** 2 / np.sum(np.abs(Mij) ** 2) / float(nsamples * nchannels)
    )
    v_sigma = 2 * np.sqrt(np.mean(np.abs(Mij) ** 2) * g_sigma ** 2)
    log.info("g_sigma = {}".format(g_sigma[0]))
    ax1 = plt.subplot(111)
    ax1.set_yscale("log")
    if show1:
        plt.plot(chisq1, ".r-", label="Alg 1 with defaults")
    if show2:
        plt.plot(chisq2, ".m-", label="Alg 2 with default lsmr")
    if show2b:
        plt.plot(chisq2b, ".b--", label="Alg 2 with lstsq, rcond = 1e-6")
    if np.any(sigma > 0):
        plt.plot(
            ax1.get_xlim(),
            2 * v_sigma[0] ** 2 * np.ones(2),
            "--",
            label="Error floor estimate",
        )
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel(r"unnormalised $\chi^2$ error", fontsize=14)
    plt.legend(loc=1, fontsize=14)
    plt.grid(True)
    plt.savefig(f"{fileprefix}-alg_error_comparison.png")
    plt.clf()


def plot_performance(
        chisq1: np.ndarray,
        chisq2: np.ndarray,
        chisq2a: np.ndarray,
        chisq2b: np.ndarray,
        chisq2c: np.ndarray,
        model_vis: xarray.Dataset,
        show1: bool,
        show2: bool,
        show2a: bool,
        show2b: bool,
        show2c: bool,
        sigma: np.ndarray,
        subarray: np.ndarray,
        nsamples: int,
        nchannels: int,
        fileprefix: str,
) -> NoReturn:
    plt.figure(num=3, figsize=(20, 8), facecolor="w", edgecolor="k")
    # back of the envelope estimate of the error RMS level.
    # gi_est ~ sum_j((Vij+error)*Mij - Mij*Mij) / sum_j(Mij*Mij)
    # gi_error ~ sum_j(error*Mij) / sum_j(Mij*Mij)
    # g_sigma ~ sqrt( sigma**2 / sum_j(Mij*Mij) )
    # vij ~ (1 + gi_error)*(1 + gj_error)*Mij
    # vij_error ~ Mij*gj_error + Mij*gi_error
    # vij_sigma ~ sqrt(2 * mean(Mij**2)) * g_sigma
    # chisq ~ mean( (vij_error)**2 )
    # really need to do the frequency averaging properly here
    Mij = model_vis.sel({"antenna1": subarray[0]})["vis"].data[0, 1::, 0, 0]
    # could use the mean of all vis and multiply by nstn-1. Careful of autos though
    g_sigma = np.sqrt(
        sigma ** 2 / np.sum(np.abs(Mij) ** 2) / float(nsamples * nchannels)
    )
    v_sigma = 2 * np.sqrt(np.mean(np.abs(Mij) ** 2) * g_sigma ** 2)
    log.info("g_sigma = {}".format(g_sigma[0]))
    ax1 = plt.subplot(131)
    ax1.set_yscale("log")
    if show1:
        plt.plot(chisq1, ".r-", label="Alg 1 with defaults")
    if np.any(sigma > 0):
        plt.plot(
            ax1.get_xlim(),
            2 * v_sigma[0] ** 2 * np.ones(2),
            "--",
            label="Error floor estimate",
        )
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel(r"unnormalised $\chi^2$ error", fontsize=14)
    plt.legend(loc=1, fontsize=12)
    plt.grid(True)
    ax2 = plt.subplot(132)
    ax2.set_yscale("log")
    if show2:
        plt.plot(chisq2, ".m-", label="Alg 2 with default lsmr")
    if show2a:
        plt.plot(chisq2a, ".g-", label="Alg 2 with lsmr & lin_solver_normal")
    # also it is the error, not the chisq, so square it. Also not exactly the right operation
    if np.any(sigma > 0):
        plt.plot(
            ax2.get_xlim(),
            2 * v_sigma[0] ** 2 * np.ones(2),
            "--",
            label="Error floor estimate",
        )
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel(r"unnormalised $\chi^2$ error", fontsize=14)
    plt.legend(loc=1, fontsize=12)
    plt.grid(True)
    ax3 = plt.subplot(133)
    ax3.set_yscale("log")
    if show2b:
        plt.plot(chisq2b, ".b-", label="Alg 2 with lstsq, rcond = 1e-6")
    if show2c:
        plt.plot(chisq2c, ".c-", label="Alg 2 with lstsq, rcond = 1e-4")
    # also it is the error, not the chisq, so square it. Also not exactly the right operation
    if np.any(sigma > 0):
        plt.plot(
            ax3.get_xlim(),
            2 * v_sigma[0] ** 2 * np.ones(2),
            "--",
            label="Error floor estimate",
        )
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel(r"unnormalised $\chi^2$ error", fontsize=14)
    plt.legend(loc=1, fontsize=12)
    plt.grid(True)
    ymin = min([ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0]])
    ymax = max([ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1]])
    ax1.set_ylim((ymin, ymax))
    ax2.set_ylim((ymin, ymax))
    ax3.set_ylim((ymin, ymax))
    plt.savefig(f"{fileprefix}-solver_performance.png")
    plt.clf()


##
# @brief AA05CaliTests
# @details Runs several calibration tests for AA0.5 scale data rates
#
# @par EAGLE_START
# @param category PythonApp
# @param execution_time Execution Time/5/Float/ComponentParameter/readonly//False/False/Estimated execution time
# @param num_cpus No. of CPUs/1/Integer/ComponentParameter/readonly//False/False/Number of cores used
# @param group_start Group start/False/Boolean/ComponentParameter/readwrite//False/False/Is this node the start of a group?
# @param appclass Application Class/dlg_ska_jones.apps.AA05CaliTests/String/ComponentParameter/readonly//False/False/Application class
# @param latitude Latitude/-26.82472208/Float/ApplicationArgument/readwrite//False/False/description
# @param longitude Longitude/116.76444824/Float/ApplicationArgument/readwrite//False/False/description
# @param nsamples Num Samples/3/Integer/ApplicationArgument/readwrite//False/False/description
# @param sample_time Sample Time/10.0/Float/ApplicationArgument/readwrite//False/False/description
# @param nchannels Num Channels/10/Integer/ApplicationArgument/readwrite//False/False/description
# @param channel_bandwidth Channel Bandwidth/1.0e6/Float/ApplicationArgument/readwrite//False/False/description
# @param ra Phase Ctr RA/0.0/Float/ApplicationArgument/readwrite//False/False/description
# @param dec Phase Ctr Dec/-27.0/Float/ApplicationArgument/readwrite//False/False/description
# @param subarray Subarray/[345, 346, 353, 354, 431, 433]/Array/ApplicationArgument/readwrite//False/False/description
# @param sky_model Skymodel/3/Integer/ApplicationArgument/readwrite//False/False/The type of skymodel (1, 2 or 3)
# @param nsrc Num Sources/25/Integer/ApplicationArgument/readwrite//False/False/The number of sources in the sky (if skymodel 1||2)
# @param ra_hrs Ra hrs//Object.Array/InputPort/readwrite//False/False/Right ascension of real sources
# @param dec_deg Dec deg//Object.Array/InputPort/readwrite//False/False/Declination in degrees of real sources
# @param jy_240MHz jy 240MHz//Object.Array/InputPort/readwrite//False/False/Flux density of sources
# @par EAGLE_END
class AA05CaliTests(BarrierAppDROP):
    """
    Runs several calibration tests for AA0.5 scale data rates.
    """

    compontent_meta = dlg_component(
        "AA05CaliTests",
        "AA0.5 Calibration Tests",
        [dlg_batch_input("binary/*", [])],
        [dlg_batch_output("binary/*", [])],
        [dlg_streaming_input("binary/*")],
    )

    latitude = dlg_float_param("latitude", -26.82472208)
    longitude = dlg_float_param("longitude", 116.76444824)
    nsamples = dlg_int_param("nsamples", 3)
    sample_time = dlg_float_param("sample_time", 10.0)
    nchannels = dlg_int_param("nchannels", 10)
    channel_bandwidth = dlg_float_param("channel_bandwidth", 1.0e6)
    ra = dlg_float_param("ra", 0.0)
    dec = dlg_float_param("dec", -27.0)
    subarray = dlg_list_param("subarray", [345, 346, 353, 354, 431, 433])
    sky_model = dlg_int_param("sky_model", 3)
    nsrc = dlg_int_param("nsrc", 25)

    def initialize(self, **kwargs):
        np.random.seed(42)
        super(AA05CaliTests, self).initialize(**kwargs)

    def run(self):
        """
        The run method is mandatory for DALiuGE application components.
        """
        t0 = time.time()
        file_prefix = f"{self.nsamples}-{self.sample_time}-{self.nchannels}-{self.channel_bandwidth}-{self.sky_model}"
        lowconfig = create_named_configuration("LOWBD2")
        stations = lowconfig["stations"]
        nstations = stations.shape[0]

        lon = self.longitude * np.pi / 180.0  # how can I extract these from lowconfig?
        lat = self.latitude * np.pi / 180.0

        times = (np.pi / 43200.0) * np.arange(
            0, self.nsamples * self.sample_time, self.sample_time
        )
        frequency = generate_frequency_range(self.nchannels, self.channel_bandwidth)
        channel_bandwidth = np.array(self.nchannels * [self.channel_bandwidth])

        phasecentre = create_phasecentre(self.ra, self.dec)

        model_vis = setup_blockvisibility(
            lowconfig,
            times,
            frequency,
            channel_bandwidth,
            phasecentre,
            self.sample_time,
        )
        nvis = model_vis["baselines"].shape[0]

        assert (
                model_vis["vis"].shape[0] == self.nsamples
        ), "Shape inconsistent with specified number of times"
        assert (
                model_vis["vis"].shape[2] == self.nchannels
        ), "Shape inconsistent with specified number of channels"
        assert (
                model_vis["vis"].shape[3] == 4
        ), "Shape inconsistent with specified number of polarisations"
        assert (
                model_vis["vis"].shape[0:3] == model_vis["uvw_lambda"].data.shape[0:3]
        ), "vis & uvw_lambda avr inconsistent"
        assert all(
            model_vis["polarisation"].data == ["XX", "XY", "YX", "YY"]
        ), "Polarisations inconsistent with expectations"

        log.info(
            "--------------------------------------------------------------------------"
        )
        log.info("Full array:")
        log.info(" - nstations = {}".format(nstations))
        log.info(" - nproducts = {}".format(int(nstations * (nstations + 1) / 2)))
        log.info(" - nbaseline = {}".format(int(nstations * (nstations - 1) / 2)))
        log.info(" - vis['vis'].shape = {}".format(model_vis["vis"].shape))
        log.info(" - nvis = {}".format(nvis))

        model_vis, subarray = filter_visibilities(model_vis, self.subarray)
        nsubarray = len(subarray)
        nvis = model_vis["baselines"].shape[0]

        log.info(
            "--------------------------------------------------------------------------"
        )
        log.info("Sub-array:")
        log.info(" - nstations = {}".format(nsubarray))
        log.info(" - nproducts = {}".format(int(nsubarray * (nsubarray + 1) / 2)))
        log.info(" - nbaseline = {}".format(int(nsubarray * (nsubarray - 1) / 2)))
        log.info(" - vis['vis'].shape = {}".format(model_vis["vis"].shape))
        log.info(" - nvis = {}".format(nvis))
        log.info("")
        log.info(" - lowconfig shape = {}".format(lowconfig["xyz"].data.shape))
        for stn in subarray:
            log.info(
                "      station {:3d} XY = {:+8.1f}, {:+8.1f}".format(
                    stn, lowconfig["xyz"].data[stn, 0], lowconfig["xyz"].data[stn, 1]
                )
            )
        log.info(
            "--------------------------------------------------------------------------"
        )

        t_initvis = time.time() - t0

        log.info("Predicting blockvisibility")

        t0 = time.time()
        if self.sky_model == 3:
            ra_hrs = pickle.loads(droputils.allDropContents(self.inputs[0]))
            dec_deg = pickle.loads(droputils.allDropContents(self.inputs[1]))
            jy_240MHz = pickle.loads(droputils.allDropContents(self.inputs[2]))
            nsrc, jy, l, m, n = setup_skymodel(
                self.sky_model,
                self.nsrc,
                self.nchannels,
                frequency,
                ra_hrs=ra_hrs,
                dec_deg=dec_deg,
                jy_240MHz=jy_240MHz,
            )
        else:
            nsrc, jy, l, m, n = setup_skymodel(
                self.sky_model, self.nsrc, self.nchannels, frequency
            )

        model_vis, noiseless_vis, is_cal = calculate_flux_densities(
            lon,
            lat,
            self.nchannels,
            nsrc,
            model_vis,
            phasecentre,
            frequency,
            jy,
            l,
            m,
            n,
        )

        t_fillvis = time.time() - t0

        plot_visibilities(is_cal, l, m, file_prefix)

        log.info("Applying calibration factors and noise")

        t0 = time.time()

        gt_fit, gt_true, sigma = update_visibilities(
            self.nchannels,
            self.sample_time,
            self.nsamples,
            channel_bandwidth,
            frequency,
            model_vis,
            noiseless_vis,
            subarray,
        )

        t_updatevis = time.time() - t0

        log.info("Solving calibration")

        # Some RASCIL functions to look into using
        # gtsol=solve_gaintable(cIVis, IVis, phase_only=False, jones_type="B")

        (
            J1,
            J2,
            J2a,
            J2b,
            J2c,
            Jt,
            chisq1,
            chisq2,
            chisq2a,
            chisq2b,
            chisq2c,
            show1,
            show2,
            show2a,
            show2b,
            show2c,
            t_solving1,
            t_solving2,
            t_solving2a,
            t_solving2b,
            t_solving2c,
        ) = run_trials(gt_fit, gt_true, model_vis, noiseless_vis, subarray, file_prefix)

        # --- #
        plot_results(
            J1, J2, J2a, J2b, J2c, Jt, show1, show2, show2a, show2b, show2c, subarray, file_prefix
        )
        # --- #

        plot_solver_error(
            J1, J2, J2a, J2b, J2c, Jt, show1, show2, show2a, show2b, show2c, subarray, file_prefix
        )

        plot_performance(
            chisq1,
            chisq2,
            chisq2a,
            chisq2b,
            chisq2c,
            model_vis,
            show1,
            show2,
            show2a,
            show2b,
            show2c,
            sigma,
            subarray,
            self.nsamples,
            self.nchannels,
            file_prefix,
        )

        plot_error_compare(
            chisq1,
            chisq2,
            chisq2b,
            model_vis,
            show1,
            show2,
            show2b,
            sigma,
            subarray,
            self.nsamples,
            self.nchannels,
            file_prefix,
        )

        log_timings(
            chisq1,
            chisq2,
            chisq2a,
            chisq2b,
            chisq2c,
            show1,
            show2,
            show2a,
            show2b,
            show2c,
            t_fillvis,
            t_initvis,
            t_solving1,
            t_solving2,
            t_solving2a,
            t_solving2b,
            t_solving2c,
            t_updatevis,
        )
        return 0
