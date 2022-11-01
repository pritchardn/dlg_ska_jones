import logging
import sys
import time

import astropy.constants as consts
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
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
from rascil.data_models import PolarisationFrame
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
# @param[in] param/dummy Dummy parameter/ /String/readwrite/
#     \~English Dummy modifyable parameter
# @param[in] port/dummy Dummy in/float/
#     \~English Dummy input port
# @param[out] port/dummy Dummy out/float/
#     \~English Dummy output port
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
        lowconfig = create_named_configuration("LOWBD2")
        lon = self.longitude * np.pi / 180.0  # how can I extract these from lowconfig?
        lat = self.latitude * np.pi / 180.0

        times = (np.pi / 43200.0) * np.arange(
            0, self.nsamples * self.sample_time, self.sample_time
        )
        frequency = np.arange(
            100.0e6,
            100.0e6 + self.nchannels * self.channel_bandwidth,
            self.channel_bandwidth,
        )
        channel_bandwidth = np.array(self.nchannels * [self.channel_bandwidth])

        phasecentre = SkyCoord(
            ra=self.ra * u.deg, dec=self.dec * u.deg, frame="icrs", equinox="J2000"
        )
        # create empty blockvis with intrumental polarisation (XX, XY, YX, YY)
        modelVis = create_blockvisibility(
            lowconfig,
            times,
            frequency,
            channel_bandwidth=channel_bandwidth,
            phasecentre=phasecentre,
            sample_time=self.sample_time,
            polarisation_frame=PolarisationFrame("linear"),
            weight=1.0,
        )

        assert (
                modelVis["vis"].shape[0] == self.nsamples
        ), "Shape inconsistent with specified number of times"
        assert (
                modelVis["vis"].shape[2] == self.nchannels
        ), "Shape inconsistent with specified number of channels"
        assert (
                modelVis["vis"].shape[3] == 4
        ), "Shape inconsistent with specified number of polarisations"
        assert (
                modelVis["vis"].shape[0:3] == modelVis["uvw_lambda"].data.shape[0:3]
        ), "vis & uvw_lambda avr inconsistent"
        assert all(
            modelVis["polarisation"].data == ["XX", "XY", "YX", "YY"]
        ), "Polarisations inconsistent with expectations"

        stations = lowconfig["stations"]
        nstations = stations.shape[0]
        nvis = modelVis["baselines"].shape[0]

        log.info(
            "--------------------------------------------------------------------------"
        )
        log.info("Full array:")
        log.info(" - nstations = {}".format(nstations))
        log.info(" - nproducts = {}".format(int(nstations * (nstations + 1) / 2)))
        log.info(" - nbaseline = {}".format(int(nstations * (nstations - 1) / 2)))
        log.info(" - vis['vis'].shape = {}".format(modelVis["vis"].shape))
        log.info(" - nvis = {}".format(nvis))

        subarray = (
                np.array(self.subarray) - 1
        )  # the two most longitudinally separated stations in each cluster

        nsubarray = len(subarray)
        modelVis = modelVis.where(
            modelVis["antenna1"].isin(subarray) * modelVis["antenna2"].isin(subarray),
            drop=True,
        )
        nvis = modelVis["baselines"].shape[0]

        log.info(
            "--------------------------------------------------------------------------"
        )
        log.info("Sub-array:")
        log.info(" - nstations = {}".format(nsubarray))
        log.info(" - nproducts = {}".format(int(nsubarray * (nsubarray + 1) / 2)))
        log.info(" - nbaseline = {}".format(int(nsubarray * (nsubarray - 1) / 2)))
        log.info(" - vis['vis'].shape = {}".format(modelVis["vis"].shape))
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

        if self.sky_model == 0:

            Nsrc = 1
            dist_source_max = 0.0
            jy = 50 * np.ones((Nsrc, self.nchannels))
            l = np.zeros(Nsrc)
            m = np.zeros(Nsrc)
            n = np.zeros(Nsrc)

        elif self.sky_model == 1 or self.sky_model == 2:

            if self.sky_model == 1:
                dist_source_max = 0.5 * np.pi / 180.0
            elif self.sky_model == 2:
                dist_source_max = 2.5 * np.pi / 180.0

            Nsrc = self.nsrc
            # sky model: randomise sources across the field
            theta = 2.0 * np.pi * np.random.rand(Nsrc)
            phi = dist_source_max * np.sqrt(np.random.rand(Nsrc))
            l = sin(theta) * sin(phi)
            m = cos(theta) * sin(phi)
            n = np.sqrt(1 - l * l - m * m) - 1

            spec_index_mult = (frequency[np.newaxis, :] / frequency[0]) ** (-0.8)
            jy = 10 * np.random.rand(Nsrc, 1) @ spec_index_mult

        elif self.sky_model == 3:

            ra_hrs = np.array(
                [
                    4.30325000e-01,
                    2.39501861e01,
                    4.08366700e-01,
                    2.38472833e01,
                    3.85488600e-01,
                    3.86428500e-01,
                    2.33322528e01,
                    5.61111000e-02,
                    5.85777800e-01,
                    2.36960611e01,
                    4.25237700e-01,
                    2.33505583e01,
                    2.37945133e01,
                    2.35297806e01,
                    2.04988900e-01,
                    7.92540000e-01,
                    2.36869250e01,
                    6.53028000e-02,
                    2.36134139e01,
                    2.33303667e01,
                    2.37626528e01,
                    2.34927056e01,
                    2.35620417e01,
                    2.37233733e01,
                    5.35391000e-02,
                    2.34131444e01,
                    4.37216700e-01,
                    2.35739247e01,
                    2.34765023e01,
                    2.56745300e-01,
                    6.33222000e-02,
                    2.38578388e01,
                    2.39598083e01,
                    2.34800972e01,
                    2.38978806e01,
                    3.53652800e-01,
                    2.39521667e01,
                    2.31687222e01,
                    2.39315167e01,
                    4.60944000e-02,
                    2.33509889e01,
                    2.87417000e-02,
                    6.74212000e-02,
                    5.39288900e-01,
                    2.36807601e01,
                    1.66069400e-01,
                    1.81889000e-02,
                    1.37826700e-01,
                    2.38595278e01,
                    2.39029505e01,
                    1.56786100e-01,
                    7.44852800e-01,
                    2.37488000e01,
                    2.66661100e-01,
                    2.70390000e-01,
                    2.69541700e-01,
                    2.68575600e-01,
                    2.36651444e01,
                    2.35748353e01,
                    2.37367556e01,
                    3.40616700e-01,
                    2.33435259e01,
                    2.39446750e01,
                    2.34754365e01,
                    3.14272200e-01,
                    2.39320121e01,
                    4.58161100e-01,
                    6.18111000e-02,
                    2.32722083e01,
                    2.99505600e-01,
                    2.36175167e01,
                    2.31075111e01,
                    2.34428613e01,
                    1.87030600e-01,
                    1.17247000e-02,
                    1.26750000e-02,
                    2.71982600e-01,
                    2.72426700e-01,
                    6.40802800e-01,
                    2.38140583e01,
                    2.34944194e01,
                    6.22860000e-01,
                    2.06986100e-01,
                    2.36323028e01,
                    2.36833679e01,
                    2.36819885e01,
                    2.35009083e01,
                    2.34221722e01,
                    7.71346400e-01,
                    3.39327800e-01,
                    5.46161100e-01,
                    2.34440389e01,
                    2.33909472e01,
                    2.88338900e-01,
                    2.34217200e-01,
                    2.33471389e01,
                    2.30399861e01,
                    2.35464167e01,
                    3.84028000e-02,
                    2.39243222e01,
                    7.36469400e-01,
                    5.30683300e-01,
                    2.32439338e01,
                    2.76926700e-01,
                    2.76774000e-01,
                    2.38294444e01,
                    5.34275000e-01,
                    9.96333000e-02,
                    6.31222000e-02,
                    1.77300000e-01,
                    2.32196228e01,
                    4.48583300e-01,
                    2.32763417e01,
                    6.00614700e-01,
                    5.98694700e-01,
                    2.95111100e-01,
                    1.77347200e-01,
                    6.65650000e-01,
                    3.66022200e-01,
                    3.87461100e-01,
                    2.38830306e01,
                    2.95025600e-01,
                    2.98812800e-01,
                    7.14139000e-02,
                    2.37492750e01,
                    3.66556000e-02,
                    2.90838900e-01,
                    2.28936100e-01,
                    4.13555600e-01,
                    7.50964300e-01,
                    2.38007278e01,
                    2.37044972e01,
                    2.37271972e01,
                    2.36793306e01,
                    2.34006083e01,
                    2.35220988e01,
                    2.35214193e01,
                    3.35769400e-01,
                    2.36310556e01,
                    2.36536000e01,
                    1.04474670e00,
                    2.43844300e-01,
                    4.74775000e-01,
                    2.35931972e01,
                    5.25202800e-01,
                    4.02738900e-01,
                    5.27688000e-02,
                    3.21264700e-01,
                    3.50058300e-01,
                    5.73386100e-01,
                    9.00083000e-02,
                    1.10069700e-01,
                    2.37321280e01,
                    3.46319300e-01,
                    3.45633000e-01,
                    3.42517900e-01,
                    3.43740000e-01,
                    3.44915600e-01,
                    1.98150000e-01,
                    1.56408300e-01,
                    6.40261100e-01,
                    9.85177800e-01,
                    1.04560380e00,
                    2.35410431e01,
                    2.36056278e01,
                    1.51672200e-01,
                    1.93777800e-01,
                    2.39959556e01,
                    2.37476028e01,
                    2.38269307e01,
                    2.38261536e01,
                    2.39887917e01,
                    2.36140611e01,
                    4.26644400e-01,
                    2.32377917e01,
                    4.55263900e-01,
                    2.38151696e01,
                    6.99972000e-02,
                    5.29258300e-01,
                    2.36256889e01,
                    1.68088900e-01,
                    4.95844400e-01,
                    5.17344400e-01,
                    2.85563900e-01,
                    3.95088900e-01,
                    3.63461100e-01,
                    2.39835250e01,
                    2.30821167e01,
                    6.08177800e-01,
                    1.44027800e-01,
                    3.93889000e-02,
                    2.39502917e01,
                    5.45755600e-01,
                    2.39707708e01,
                    6.86727800e-01,
                    3.31230600e-01,
                    2.39902800e01,
                    2.38604139e01,
                    2.14516700e-01,
                    1.19258300e-01,
                    2.28842387e01,
                    2.37806111e01,
                    2.38331722e01,
                    2.39070815e01,
                    6.85273000e-02,
                    2.36514222e01,
                    2.32633778e01,
                    2.38510167e01,
                    2.37218111e01,
                    2.38653861e01,
                    2.84177800e-01,
                    6.69056000e-02,
                    2.36241556e01,
                    5.56630600e-01,
                    2.28119400e-01,
                    2.38015028e01,
                    3.12177800e-01,
                    2.34978047e01,
                    2.27600000e-01,
                    2.30508267e01,
                    2.34358361e01,
                    2.58602800e-01,
                    1.59933300e-01,
                    2.23851800e-01,
                    2.25496900e-01,
                    2.25911200e-01,
                    2.39073556e01,
                    2.32208806e01,
                    2.38990750e01,
                    9.74372800e-01,
                    2.38214806e01,
                    2.39522972e01,
                    4.34986100e-01,
                    4.64705600e-01,
                    2.37363972e01,
                    7.25083000e-02,
                    3.60394400e-01,
                    2.33315361e01,
                    2.35095056e01,
                    1.66750000e-02,
                    2.37450083e01,
                    5.75667000e-02,
                    2.82139000e-02,
                    5.93686100e-01,
                    2.37445000e01,
                    2.39766778e01,
                    1.13810830e00,
                    2.39406139e01,
                    2.38085313e01,
                    5.92963900e-01,
                    2.36781333e01,
                    2.97292500e-01,
                    2.98568600e-01,
                    2.34312667e01,
                    1.00516700e-01,
                    2.37593791e01,
                    2.37599792e01,
                    3.60000000e-01,
                    2.38386965e01,
                    2.32779694e01,
                    4.31000000e-01,
                    3.81139000e-02,
                    3.99811100e-01,
                    2.07797200e-01,
                    1.56475000e-01,
                    2.35920472e01,
                    2.36314851e01,
                    2.91075000e-01,
                    2.38889000e01,
                    5.31311100e-01,
                    2.36047750e01,
                    9.80528000e-02,
                    2.34344361e01,
                    7.31797200e-01,
                    2.36885694e01,
                    4.07513900e-01,
                    6.14028000e-02,
                    2.35274028e01,
                    2.38198194e01,
                    2.37973528e01,
                    2.34183389e01,
                    2.37282528e01,
                    2.39920806e01,
                    3.29080600e-01,
                    2.37778472e01,
                    2.38424583e01,
                    3.16611000e-02,
                    2.32763444e01,
                    2.39651944e01,
                    3.17513900e-01,
                    2.51780600e-01,
                    2.36439472e01,
                    2.38538167e01,
                    7.35203000e-02,
                    2.34144750e01,
                    2.39417583e01,
                    6.16983300e-01,
                    2.39238472e01,
                    2.37126639e01,
                    2.38222333e01,
                    2.35451306e01,
                    2.36384581e01,
                    3.84813900e-01,
                    8.44167000e-02,
                    2.37521139e01,
                    2.34221750e01,
                    9.08291700e-01,
                    1.84056000e-02,
                    2.35849500e01,
                    1.34855600e-01,
                    2.31182583e01,
                    2.35064861e01,
                    2.38351126e01,
                    2.34889000e-02,
                    3.83811100e-01,
                    5.33175000e-01,
                    2.29197444e01,
                    2.39634056e01,
                ]
            )
            dec_deg = np.array(
                [
                    -26.03689,
                    -34.75881,
                    -29.48025,
                    -24.95,
                    -25.03978,
                    -25.04871,
                    -27.45386,
                    -17.45317,
                    -20.06644,
                    -35.10689,
                    -33.05531,
                    -16.38478,
                    -28.1443,
                    -24.86942,
                    -33.36606,
                    -25.2863,
                    -16.34858,
                    -30.99792,
                    -34.73536,
                    -22.06469,
                    -24.04058,
                    -19.38475,
                    -30.9655,
                    -21.6929,
                    -35.93893,
                    -27.32206,
                    -20.08269,
                    -41.42383,
                    -21.09776,
                    -38.0765,
                    -23.49456,
                    -23.286,
                    -21.22331,
                    -25.13475,
                    -23.19058,
                    -19.17878,
                    -18.29506,
                    -27.96433,
                    -30.01203,
                    -30.47375,
                    -24.17822,
                    -30.95872,
                    -23.11666,
                    -24.08567,
                    -28.81043,
                    -28.49125,
                    -16.99025,
                    -19.6661,
                    -16.12797,
                    -21.94757,
                    -31.771,
                    -35.50911,
                    -30.68383,
                    -19.48933,
                    -31.38664,
                    -31.40909,
                    -31.42519,
                    -33.17097,
                    -33.75069,
                    -24.12831,
                    -20.23833,
                    -33.89219,
                    -31.82311,
                    -14.90443,
                    -12.70931,
                    -18.16879,
                    -27.52003,
                    -17.67411,
                    -27.49858,
                    -22.63444,
                    -32.67978,
                    -25.11514,
                    -40.45431,
                    -31.842,
                    -34.40055,
                    -27.38097,
                    -14.4987,
                    -14.50897,
                    -38.99617,
                    -31.7385,
                    -21.22878,
                    -23.1421,
                    -18.80242,
                    -17.87328,
                    -28.58644,
                    -28.62515,
                    -18.13428,
                    -12.12461,
                    -42.12858,
                    -20.48004,
                    -29.51939,
                    -25.69714,
                    -30.96658,
                    -30.91028,
                    -23.64202,
                    -19.32225,
                    -37.30181,
                    -31.62542,
                    -25.65406,
                    -24.20733,
                    -22.20775,
                    -26.87367,
                    -21.93166,
                    -38.46484,
                    -38.42878,
                    -29.34125,
                    -22.96844,
                    -27.9835,
                    -25.34061,
                    -32.41933,
                    -31.96396,
                    -36.92547,
                    -16.42942,
                    -30.39444,
                    -30.3593,
                    -30.83522,
                    -18.99314,
                    -25.57483,
                    -26.32142,
                    -27.05511,
                    -29.92239,
                    -22.41969,
                    -22.34545,
                    -22.21458,
                    -26.83483,
                    -21.88589,
                    -31.99547,
                    -25.62006,
                    -20.71969,
                    -24.57281,
                    -16.52017,
                    -26.81919,
                    -30.14506,
                    -23.04619,
                    -22.58197,
                    -18.67876,
                    -18.67695,
                    -32.11717,
                    -34.91383,
                    -33.01939,
                    -21.875,
                    -24.48569,
                    -29.57781,
                    -34.5585,
                    -32.99539,
                    -25.92956,
                    -14.67454,
                    -24.65775,
                    -24.84983,
                    -17.99019,
                    -23.02756,
                    -23.13542,
                    -31.65478,
                    -20.35091,
                    -20.35141,
                    -20.27197,
                    -20.29001,
                    -20.31509,
                    -32.24986,
                    -28.36497,
                    -22.88411,
                    -17.01086,
                    -27.5248,
                    -18.85303,
                    -24.65589,
                    -20.94231,
                    -23.98833,
                    -33.06522,
                    -27.68447,
                    -16.58829,
                    -16.59575,
                    -20.79883,
                    -22.58658,
                    -24.99667,
                    -30.62164,
                    -33.7805,
                    -31.28866,
                    -34.52167,
                    -26.42786,
                    -37.47869,
                    -23.28017,
                    -17.67878,
                    -22.11847,
                    -12.94025,
                    -21.07897,
                    -25.88403,
                    -23.27511,
                    -34.52478,
                    -37.46061,
                    -17.1785,
                    -14.11194,
                    -21.63842,
                    -21.73931,
                    -18.02973,
                    -29.13036,
                    -31.34075,
                    -24.4179,
                    -30.13078,
                    -23.45617,
                    -25.54942,
                    -40.96297,
                    -30.27961,
                    -19.20789,
                    -23.98687,
                    -25.06463,
                    -19.40231,
                    -18.01106,
                    -38.65794,
                    -29.72417,
                    -27.97281,
                    -20.987,
                    -26.62183,
                    -26.26236,
                    -21.70033,
                    -30.15736,
                    -31.40839,
                    -24.01244,
                    -21.09864,
                    -32.41275,
                    -18.6886,
                    -28.34294,
                    -30.602,
                    -32.27692,
                    -19.49933,
                    -19.50554,
                    -19.48074,
                    -25.10614,
                    -31.41767,
                    -20.11494,
                    -24.01732,
                    -25.20406,
                    -22.34478,
                    -20.613,
                    -20.12767,
                    -28.85881,
                    -28.67208,
                    -21.77897,
                    -36.94106,
                    -30.79864,
                    -25.08442,
                    -15.06969,
                    -15.78508,
                    -15.67794,
                    -16.58383,
                    -33.56167,
                    -21.98314,
                    -16.07344,
                    -32.83911,
                    -26.17329,
                    -28.41769,
                    -21.7772,
                    -22.37381,
                    -22.34694,
                    -25.76947,
                    -42.57786,
                    -22.79101,
                    -22.78479,
                    -33.57619,
                    -21.97423,
                    -20.89272,
                    -27.28914,
                    -33.52264,
                    -32.96569,
                    -18.59239,
                    -31.37839,
                    -33.09925,
                    -14.58765,
                    -25.07397,
                    -27.22611,
                    -31.24422,
                    -25.93461,
                    -35.36736,
                    -29.18331,
                    -16.08083,
                    -23.1525,
                    -27.99883,
                    -31.06675,
                    -15.94933,
                    -32.90633,
                    -27.64058,
                    -23.40203,
                    -31.03494,
                    -21.43022,
                    -24.87328,
                    -20.50411,
                    -19.63792,
                    -31.42181,
                    -14.15161,
                    -25.24014,
                    -23.351,
                    -33.03206,
                    -26.21642,
                    -19.51403,
                    -23.12605,
                    -23.06861,
                    -32.37286,
                    -23.44578,
                    -33.96589,
                    -29.17831,
                    -29.85561,
                    -27.73736,
                    -33.29003,
                    -18.93183,
                    -19.83222,
                    -30.52022,
                    -24.68378,
                    -23.85867,
                    -17.69081,
                    -31.26825,
                    -28.12547,
                    -25.44611,
                    -35.995,
                    -26.91496,
                    -20.66822,
                    -29.69192,
                    -21.16069,
                    -30.42897,
                    -19.33625,
                ]
            )
            jy_240MHz = np.array(
                [
                    16.25931,
                    13.14117,
                    10.75641,
                    6.50575,
                    5.15164,
                    2.82382,
                    9.32667,
                    7.80235,
                    8.7943,
                    7.27813,
                    5.91139,
                    11.69769,
                    2.65828,
                    3.77433,
                    4.22941,
                    8.74281,
                    6.70284,
                    3.15896,
                    4.84234,
                    4.71073,
                    2.66771,
                    4.87415,
                    3.69794,
                    2.96111,
                    4.24912,
                    3.65458,
                    4.12444,
                    19.3798,
                    3.77629,
                    6.19723,
                    2.16435,
                    2.05592,
                    2.44325,
                    2.61555,
                    2.03427,
                    3.40862,
                    3.07946,
                    4.25758,
                    1.88119,
                    1.96249,
                    3.06423,
                    1.99935,
                    1.90079,
                    2.70411,
                    2.00533,
                    1.7561,
                    3.28716,
                    2.46404,
                    3.70735,
                    1.82422,
                    1.8702,
                    6.63012,
                    1.90321,
                    2.60841,
                    1.53873,
                    0.33037,
                    0.13488,
                    2.47983,
                    2.56449,
                    1.77538,
                    2.26431,
                    3.45556,
                    1.78913,
                    5.84359,
                    8.46879,
                    2.36056,
                    1.83022,
                    2.29289,
                    2.53308,
                    1.59931,
                    2.06722,
                    3.58292,
                    10.04496,
                    1.68406,
                    1.99111,
                    1.24198,
                    2.66086,
                    1.8061,
                    7.37783,
                    1.59014,
                    2.15643,
                    2.34127,
                    2.03425,
                    2.71219,
                    0.91829,
                    0.53026,
                    2.7208,
                    9.78298,
                    21.93739,
                    1.82465,
                    2.01581,
                    1.77069,
                    2.00615,
                    1.39122,
                    1.25452,
                    2.92656,
                    8.55751,
                    1.65386,
                    1.13177,
                    1.10325,
                    2.71315,
                    1.63284,
                    2.74853,
                    1.8509,
                    1.91235,
                    1.15496,
                    1.67426,
                    1.04644,
                    0.98461,
                    1.30131,
                    2.58412,
                    2.91408,
                    4.08207,
                    1.09116,
                    0.56527,
                    1.18034,
                    1.45364,
                    1.7456,
                    1.08752,
                    1.11403,
                    1.03323,
                    0.91178,
                    0.28917,
                    1.00045,
                    1.02884,
                    1.06237,
                    1.22828,
                    1.02701,
                    1.50418,
                    2.12952,
                    2.22938,
                    0.95035,
                    0.94524,
                    1.09341,
                    1.42015,
                    1.14486,
                    0.64661,
                    1.2363,
                    1.73111,
                    1.40234,
                    5.44415,
                    0.88935,
                    1.19064,
                    1.57402,
                    1.714,
                    1.0742,
                    2.55056,
                    0.90615,
                    0.98808,
                    2.34216,
                    0.85384,
                    0.85526,
                    1.01164,
                    0.09336,
                    0.05377,
                    0.64269,
                    0.29951,
                    0.22262,
                    1.10632,
                    0.8407,
                    1.63043,
                    8.45377,
                    4.69758,
                    1.58763,
                    0.90458,
                    1.18307,
                    0.87252,
                    0.96374,
                    0.76391,
                    0.87946,
                    0.67594,
                    1.03714,
                    1.03604,
                    0.92074,
                    1.6597,
                    1.42948,
                    0.93169,
                    1.12693,
                    1.05538,
                    2.02983,
                    0.81218,
                    1.86989,
                    1.27711,
                    3.89158,
                    1.02395,
                    0.84248,
                    0.75921,
                    3.3086,
                    2.81722,
                    1.39497,
                    2.56113,
                    0.86552,
                    1.28813,
                    1.22114,
                    1.41124,
                    0.9596,
                    0.70263,
                    0.77096,
                    0.76058,
                    0.66888,
                    17.9765,
                    0.75406,
                    1.01873,
                    0.66869,
                    0.61299,
                    1.12407,
                    2.52406,
                    2.14214,
                    0.72627,
                    0.688,
                    0.96288,
                    0.62594,
                    0.68792,
                    1.25013,
                    0.75353,
                    0.76085,
                    0.72638,
                    1.01915,
                    0.82519,
                    3.61722,
                    1.007,
                    0.79225,
                    0.88099,
                    0.24071,
                    0.21856,
                    0.06287,
                    0.55392,
                    1.66687,
                    0.86412,
                    2.9007,
                    0.60984,
                    0.69039,
                    1.05822,
                    1.1774,
                    0.68816,
                    0.61856,
                    0.85945,
                    2.48691,
                    0.88349,
                    0.57028,
                    1.76184,
                    1.50381,
                    1.49965,
                    2.30415,
                    0.86556,
                    0.65203,
                    16.3729,
                    0.75779,
                    0.58112,
                    0.99387,
                    0.75737,
                    0.3417,
                    0.41979,
                    0.81026,
                    5.99017,
                    0.28564,
                    0.39871,
                    1.02136,
                    0.75813,
                    1.39549,
                    0.77042,
                    0.79033,
                    0.94435,
                    0.93976,
                    0.67441,
                    0.93822,
                    1.93776,
                    0.63753,
                    0.55544,
                    1.06803,
                    0.63474,
                    0.9803,
                    0.84065,
                    3.11735,
                    0.63822,
                    0.6512,
                    0.64221,
                    1.80424,
                    0.74239,
                    0.53968,
                    0.88458,
                    0.74599,
                    0.62868,
                    0.5996,
                    0.7465,
                    0.76907,
                    0.57329,
                    3.59782,
                    0.55889,
                    0.60854,
                    0.76756,
                    0.57875,
                    0.71878,
                    0.56739,
                    0.88729,
                    0.68274,
                    0.96117,
                    0.76105,
                    0.56075,
                    0.56515,
                    0.62625,
                    0.80777,
                    0.95917,
                    0.7191,
                    0.5817,
                    0.78735,
                    1.90476,
                    0.97482,
                    0.79299,
                    0.51951,
                    1.43709,
                    1.28113,
                    0.48058,
                    0.63444,
                    0.75445,
                    0.92016,
                    2.39996,
                    0.70042,
                ]
            )
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
            Nsrc = len(jy_240MHz)

            spec_index_mult = (frequency[np.newaxis, :] / 240e6) ** (-0.8)
            jy = jy_240MHz[:, np.newaxis] * spec_index_mult

        else:
            raise ValueError("Unknown sky_model index")

        # make a full copy of the model for the actual visibilities.
        # The sky and cal models could be different...
        noiselessVis = modelVis.copy(deep=True)

        is_cal = np.zeros(Nsrc, "bool")

        for src in range(0, Nsrc):

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
            for t in range(0, len(modelVis["datetime"])):

                utc_time = modelVis["datetime"].data[t, 0]
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

                for f in range(0, self.nchannels):

                    wl = consts.c.value / frequency[f]
                    sigma = wl / diam / 2.355
                    gain = np.exp(-sep * sep / (2 * sigma * sigma))

                    srcbeam = JJ * gain

                    # vis (time, baselines, frequency, polarisation) complex128

                    uvw = modelVis["uvw_lambda"].data[t, :, f, :]
                    phaser = (
                            0.5
                            * jy[src, f]
                            * np.exp(
                        2j
                        * np.pi
                        * (
                                uvw[:, 0] * l[src]
                                + uvw[:, 1] * m[src]
                                + uvw[:, 2] * n[src]
                        )
                    )
                    )

                    assert all(
                        modelVis["polarisation"].data == ["XX", "XY", "YX", "YY"]
                    ), "pol error"

                    noiselessVis["vis"].data[t, :, f, 0] += phaser * srcbeam[0, 0]
                    noiselessVis["vis"].data[t, :, f, 1] += phaser * srcbeam[0, 1]
                    noiselessVis["vis"].data[t, :, f, 2] += phaser * srcbeam[1, 0]
                    noiselessVis["vis"].data[t, :, f, 3] += phaser * srcbeam[1, 1]

                    if is_cal[src]:
                        modelVis["vis"].data[t, :, f, 0] += phaser * srcbeam[0, 0]
                        modelVis["vis"].data[t, :, f, 1] += phaser * srcbeam[0, 1]
                        modelVis["vis"].data[t, :, f, 2] += phaser * srcbeam[1, 0]
                        modelVis["vis"].data[t, :, f, 3] += phaser * srcbeam[1, 1]

        t_fillvis = time.time() - t0

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
        plt.savefig("blockvisibilities.png")

        log.info("Applying calibration factors and noise")

        t0 = time.time()

        # Some RASCIL functions to look into using
        # gt_true = simulate_gaintable(modelVis, phase_error=1.0, amplitude_error=0.1, leakage=0.1)
        # gt_fit  = simulate_gaintable(modelVis, phase_error=0.0, amplitude_error=0.0, leakage=0.0)

        # generate a gaintable with a single timeslice
        # (is in sec, so should be > 43200 for a 12 hr observation)
        # could alternatively just use the first time step in the call
        # "ValueError: Unknown Jones type P"
        gt_true = create_gaintable_from_blockvisibility(
            modelVis, timeslice=1e6, jones_type="G"
        )
        gt_fit = create_gaintable_from_blockvisibility(
            modelVis, timeslice=1e6, jones_type="G"
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
        stn1 = modelVis["antenna1"].data
        stn2 = modelVis["antenna2"].data

        for t in range(0, len(modelVis["datetime"])):
            for f in range(0, self.nchannels):

                # set up references to the data
                modelTmp = modelVis["vis"].data[t, :, f, :]
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
        sigma_calc = SEFD / np.sqrt(2.0 * channel_bandwidth * self.sample_time)
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
        assert shape[0] == self.nsamples, "unexpected time dimension"
        assert shape[2] == self.nchannels, "unexpected frequency dimension"

        for f in range(0, self.nchannels):
            observedVis["vis"].data[:, :, f, :] += sigma[f] * (
                    np.random.randn(shape[0], shape[1], shape[3])
                    + np.random.randn(shape[0], shape[1], shape[3]) * 1j
            )
            if sigma[f] > 0:
                modelVis["weight"].data[:, :, f, :] *= 1.0 / (sigma[f] * sigma[f])
                observedVis["weight"].data[:, :, f, :] *= 1.0 / (sigma[f] * sigma[f])

        t_updatevis = time.time() - t0

        log.info("Solving calibration")

        # Some RASCIL functions to look into using
        # gtsol=solve_gaintable(cIVis, IVis, phase_only=False, jones_type="B")

        show1 = True
        if show1:
            log.info("Running algorithm 1 with defaults")
            gt1 = gt_fit.copy(deep=True)
            modelVis1 = modelVis.copy(deep=True)
            t0 = time.time()
            chisq1 = solve_jones(
                observedVis,
                modelVis1,
                gt1,
                testvis=noiselessVis,
                algorithm=1,
                niter=50,
                tol=1e-6,
            )
            t_solving1 = time.time() - t0

        show2 = False
        if show2:
            log.info("Running algorithm 2 with defaults")
            gt2 = gt_fit.copy(deep=True)
            modelVis2 = modelVis.copy(deep=True)
            t0 = time.time()
            chisq2 = solve_jones(
                observedVis, modelVis2, gt2, testvis=noiselessVis, algorithm=2
            )
            t_solving2 = time.time() - t0

        show2a = True
        if show2a:
            log.info("Running algorithm 2 with lin_solver_normal")
            gt2a = gt_fit.copy(deep=True)
            modelVis2a = modelVis.copy(deep=True)
            t0 = time.time()
            chisq2a = solve_jones(
                observedVis,
                modelVis2a,
                gt2a,
                testvis=noiselessVis,
                algorithm=2,
                lin_solver_normal=True,
            )
            t_solving2a = time.time() - t0

        show2b = True
        if show2b:
            log.info("Running algorithm 2 with lin_solver=lstsq")
            gt2b = gt_fit.copy(deep=True)
            modelVis2b = modelVis.copy(deep=True)
            t0 = time.time()
            chisq2b = solve_jones(
                observedVis,
                modelVis2b,
                gt2b,
                testvis=noiselessVis,
                algorithm=2,
                lin_solver="lstsq",
            )
            t_solving2b = time.time() - t0

        show2c = True
        if show2c:
            log.info("Running algorithm 2 with lin_solver=lstsq & rcond=1e-4")
            gt2c = gt_fit.copy(deep=True)
            modelVis2c = modelVis.copy(deep=True)
            t0 = time.time()
            chisq2c = solve_jones(
                observedVis,
                modelVis2c,
                gt2c,
                testvis=noiselessVis,
                algorithm=2,
                lin_solver="lstsq",
                lin_solver_rcond=1e-4,
            )
            t_solving2c = time.time() - t0

            # copy gain data for the subarray
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

            # --- #

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
            plt.savefig("solver_results.png")
            # --- #

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

            def plot_ambiguity(J, col, label=""):
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
                plot_ambiguity(J1, "r-", "Alg 1 with defaults")
            if show2:
                plot_ambiguity(J2, "m-", "Alg 2 with default lsmr")
            if show2a:
                plot_ambiguity(J2a, "g-", "Alg 2 with lsmr & lin_solver_normal")
            if show2b:
                plot_ambiguity(J2b, "b--", "Alg 2 with lstsq, rcond = 1e-6")
            if show2c:
                plot_ambiguity(J2c, "c-", "Alg 2 with lstsq, rcond = 1e-4")
            ax241.legend(fontsize=10)
        plt.savefig("solver_error.png")
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
        Mij = modelVis.sel({"antenna1": subarray[0]})["vis"].data[0, 1::, 0, 0]
        # could use the mean of all vis and multiply by nstn-1. Careful of autos though
        g_sigma = np.sqrt(
            sigma ** 2
            / np.sum(np.abs(Mij) ** 2)
            / float(self.nsamples * self.nchannels)
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
        plt.grid()

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
        plt.grid()

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
        plt.grid()

        ymin = min([ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0]])
        ymax = max([ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1]])
        ax1.set_ylim((ymin, ymax))
        ax2.set_ylim((ymin, ymax))
        ax3.set_ylim((ymin, ymax))
        plt.savefig("solver_performance.png")

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
        Mij = modelVis.sel({"antenna1": subarray[0]})["vis"].data[0, 1::, 0, 0]
        # could use the mean of all vis and multiply by nstn-1. Careful of autos though
        g_sigma = np.sqrt(
            sigma ** 2
            / np.sum(np.abs(Mij) ** 2)
            / float(self.nsamples * self.nchannels)
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
        plt.grid()
        plt.savefig("alg_error_comparison.png")

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

        return 0
