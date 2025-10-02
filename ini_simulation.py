import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import time
from tensorflow import keras
from keras import Model
import sionna as sn
import numpy as np
import pickle
import scipy.io as sio
from sionna.channel.tr38901 import Antenna, AntennaArray, CDL
from sionna.channel import OFDMChannel
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, \
    ResourceGridDemapper
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, log10, expand_to_rank
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber
from sionna.mimo import StreamManagement
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers
from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, \
    time_lag_discrete_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.utils import BinarySource, ebnodb2no, sim_ber
from sionna.utils.metrics import compute_ber

# We need to enable sionna.config.xla_compat before we can use
# tf.function with jit_compile=True.
sn.config.xla_compat = True


num_ut = 1
num_bs = 1
num_ut_ant = 1
num_bs_ant = 2

num_streams_per_tx = num_ut_ant
rx_tx_association = np.array([[1]])

num_tx = 1
num_streams_per_tx = 1
num_ofdm_symbols = 14
num_effective_subcarriers = 64
mask = np.zeros([num_tx,
                 num_streams_per_tx,
                 num_ofdm_symbols,
                 num_effective_subcarriers])
mask[0, :, [2], :] = 1
num_pilot_symbols = int(np.sum(mask[0, 0]))
pilots = np.zeros([num_tx, num_streams_per_tx, num_pilot_symbols], np.complex64)
pilots[0, 0, 0:num_pilot_symbols:2] = (1+1j)/np.sqrt(2)
pp = sn.ofdm.PilotPattern(mask, pilots)   # pilot parttern


class Model(tf.keras.Model):
    """This Keras model simulates OFDM MIMO transmissions over the CDL model.

    Simulates point-to-point transmissions between a UT and a BS.
    Uplink and downlink transmissions can be realized with either perfect CSI
    or channel estimation. ZF Precoding for downlink transmissions is assumed.
    The receiver (in both uplink and downlink) applies LS channel estimation
    and LMMSE MIMO equalization. A 5G LDPC code as well as QAM modulation are
    used.

    Parameters
    ----------
    domain : One of ["time", "freq"], str
        Determines if the channel is modeled in the time or frequency domain.
        Time-domain simulations are generally slower and consume more memory.
        They allow modeling of inter-symbol interference and channel changes
        during the duration of an OFDM symbol.

    direction : One of ["uplink", "downlink"], str
        For "uplink", the UT transmits. For "downlink" the BS transmits.

    cdl_model : One of ["A", "B", "C", "D", "E"], str
        The CDL model to use. Note that "D" and "E" are LOS models that are
        not well suited for the transmissions of multiple streams.

    delay_spread : float
        The nominal delay spread [s].

    perfect_csi : bool
        Indicates if perfect CSI at the receiver should be assumed. For downlink
        transmissions, the transmitter is always assumed to have perfect CSI.

    speed : float
        The UT speed [m/s].

    cyclic_prefix_length : int
        The length of the cyclic prefix in number of samples.

    pilot_ofdm_symbol_indices : list, int
        List of integers defining the OFDM symbol indices that are reserved
        for pilots.

    subcarrier_spacing : float
        The subcarrier spacing [Hz]. Defaults to 15e3.

    Input
    -----
    batch_size : int
        The batch size, i.e., the number of independent Mote Carlo simulations
        to be performed at once. The larger this number, the larger the memory
        requiremens.

    ebno_db : float
        The Eb/No [dB]. This value is converted to an equivalent noise power
        by taking the modulation order, coderate, pilot and OFDM-related
        overheads into account.

    Output
    ------
    b : [batch_size, 1, num_streams, k], tf.float32
        The tensor of transmitted information bits for each stream.

    b_hat : [batch_size, 1, num_streams, k], tf.float32
        The tensor of received information bits for each stream.
    """

    def __init__(self,
                 domain,
                 direction,
                 cdl_model,
                 delay_spread,
                 perfect_csi,
                 speed,
                 cyclic_prefix_length,
                 pilot_ofdm_symbol_indices,
                 subcarrier_spacing=15e3
                 ):
        super().__init__()

        # Provided parameters
        self._domain = domain
        self._direction = direction
        self._cdl_model = cdl_model
        self._delay_spread = delay_spread
        self._perfect_csi = perfect_csi
        self._speed = speed
        self._cyclic_prefix_length = cyclic_prefix_length
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices

        # System parameters
        self._carrier_frequency = 2.6e9
        self._subcarrier_spacing = subcarrier_spacing
        self._fft_size = 76
        self._num_ofdm_symbols = 14
        self._num_ut_ant = 1  # Must be a multiple of two as dual-polarized antennas are used
        self._num_bs_ant = 2  # Must be a multiple of two as dual-polarized antennas are used
        self._num_streams_per_tx = self._num_ut_ant
        self._dc_null = True
        self._num_guard_carriers = [5, 6]
        self._pilot_pattern = pp #"kronecker"
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._num_bits_per_symbol = 4
        self._coderate = 0.5

        # Required system components
        self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)

        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing=self._subcarrier_spacing,
                                num_tx=1,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        self._k = int(self._n * self._coderate)

        self._ut_array = Antenna(polarization="single",
                                 polarization_type="V",
                                 antenna_pattern="38.901",
                                 carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=1,
                                      polarization="dual",
                                      polarization_type="VH",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        # self._ut_array = AntennaArray(num_rows=1,
        #                               num_cols=int(self._num_ut_ant/2),
        #                               polarization="dual",
        #                               polarization_type="cross",
        #                               antenna_pattern="38.901",
        #                               carrier_frequency=self._carrier_frequency)
        #
        # self._bs_array = AntennaArray(num_rows=1,
        #                               num_cols=int(self._num_bs_ant/2),
        #                               polarization="dual",
        #                               polarization_type="cross",
        #                               antenna_pattern="38.901",
        #                               carrier_frequency=self._carrier_frequency)

        self._cdl = CDL(model=self._cdl_model,
                        delay_spread=self._delay_spread,
                        carrier_frequency=self._carrier_frequency,
                        ut_array=self._ut_array,
                        bs_array=self._bs_array,
                        direction=self._direction,
                        min_speed=self._speed)

        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)

        if self._domain == "freq":
            self._channel_freq = ApplyOFDMChannel(add_awgn=True)

        elif self._domain == "time":
            self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
            self._l_tot = self._l_max - self._l_min + 1
            self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                                  l_tot=self._l_tot,
                                                  add_awgn=True)
            self._modulator = OFDMModulator(self._cyclic_prefix_length)
            self._demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        if self._direction == "downlink":
            self._zf_precoder = ZFPrecoder(self._rg, self._sm, return_effective_channel=True)

        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)

    @tf.function(jit_compile=True)  #
    def call(self, batch_size, ebno_db):  # ebno_db
        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)  # ebno_db 0
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        if self._domain == "time":
            # Time-domain simulations

            a, tau = self._cdl(batch_size, self._rg.num_time_samples + self._l_tot - 1, self._rg.bandwidth)
            h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                         l_min=self._l_min, l_max=self._l_max, normalize=True)

            # As precoding is done in the frequency domain, we need to downsample
            # the path gains `a` to the OFDM symbol rate prior to converting the CIR
            # to the channel frequency response.
            a_freq = a[..., self._rg.cyclic_prefix_length:-1:(self._rg.fft_size + self._rg.cyclic_prefix_length)]
            a_freq = a_freq[..., :self._rg.num_ofdm_symbols]
            h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder([x_rg, h_freq])

            x_time = self._modulator(x_rg)
            y_time = self._channel_time([x_time, h_time, no])

            y = self._demodulator(y_time)

        elif self._domain == "freq":
            # Frequency-domain simulations

            cir = self._cdl(batch_size, self._rg.num_ofdm_symbols, 1 / self._rg.ofdm_symbol_duration)
            h_freq = cir_to_ofdm_channel(self._frequencies, *cir, normalize=True)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder([x_rg, h_freq])

            y = self._channel_freq([x_rg, h_freq, no])

        if self._perfect_csi:
            if self._direction == "uplink":
                h_hat = self._remove_nulled_scs(h_freq)
            elif self._direction == "downlink":
                h_hat = g
            err_var = 0.0
        else:
            h_hat, err_var = self._ls_est([y, no])

        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
        llr = self._demapper([x_hat, no_eff])
        b_hat = self._decoder(llr)

        return b, y_time  # y_time b_hat


# # difference channel model
# UL_SIMS = {
#     "ebno_db": list(np.arange(-10, 32, 2.0)),
#     "cdl_model": ["A", "B", "C", "D", "E"], #
#     "delay_spread": 100e-9,
#     "domain": "time",
#     "direction": "uplink",
#     "perfect_csi": False,
#     "speed": 0.0,
#     "cyclic_prefix_length": 6,
#     "pilot_ofdm_symbol_indices": [0],
#     "ber": [],
#     "bler": [],
#     "duration": None
# }
#
# start = time.time()
#
# for cdl_model in UL_SIMS["cdl_model"]:
#
#     model = Model(domain=UL_SIMS["domain"],
#                   direction=UL_SIMS["direction"],
#                   cdl_model=cdl_model,
#                   delay_spread=UL_SIMS["delay_spread"],
#                   perfect_csi=UL_SIMS["perfect_csi"],
#                   speed=UL_SIMS["speed"],
#                   cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
#                   pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"])
#
#     ber, bler = sim_ber(model,
#                         UL_SIMS["ebno_db"],
#                         batch_size=256,
#                         max_mc_iter=100,
#                         num_target_block_errors=1000)
#
#     UL_SIMS["ber"].append(list(ber.numpy()))
#     UL_SIMS["bler"].append(list(bler.numpy()))
#
# UL_SIMS["duration"] = time.time() - start

#####################################################################################################
# speed influence
MOBILITY_SIMS = {
    # "ebno_db": list(np.arange(0, 32, 2.0)),
    "cdl_model": "D",
    "delay_spread": 100e-9,
    "domain": "time",
    "direction": "uplink",
    "perfect_csi": False,
    "speed": 20.0,
    "cyclic_prefix_length": 6,
    "pilot_ofdm_symbol_indices": [2],
    "ber": [],
    "bler": [],
    "duration": None
}
##################################################################################################
# # # 9canshu
# start = time.time()
model = Model(domain=MOBILITY_SIMS["domain"],
              direction=MOBILITY_SIMS["direction"],
              cdl_model=MOBILITY_SIMS["cdl_model"],
              delay_spread=MOBILITY_SIMS["delay_spread"],
              perfect_csi=MOBILITY_SIMS["perfect_csi"],
              speed=MOBILITY_SIMS["speed"],
              cyclic_prefix_length=MOBILITY_SIMS["cyclic_prefix_length"],
              pilot_ofdm_symbol_indices=MOBILITY_SIMS["pilot_ofdm_symbol_indices"])

# # 遍历不同的 ebno_db,修改dB并往里叠加即可
# # for ebno_db in range(23, 28, 1):
# for ebno_db in range(0, 5, 1):     # 1-15 30000 16-30 30000 20-30 22000  23-28 12500 20-265 3000
#     b, ber = model(ebno_db=ebno_db, batch_size=500)
#     b = b.numpy()
#     ber = ber.numpy()
#     # 保存结果到.mat文件
#     if ebno_db == 0:
#         # 如果是第一次保存，直接创建文件
#         sio.savemat(f'../../5G_OFDM_Detection/Dataset_0709/train_dataset_{MOBILITY_SIMS["cdl_model"]}_X.mat', {'train_data': ber, 'train_label': b})
#     else:
#         # 如果文件已存在，直接追加数据
#         existing_data = sio.loadmat('train_dataset_20.mat')
#         existing_data['train_data'] = np.concatenate([existing_data['train_data'], ber])
#         existing_data['train_label'] = np.concatenate([existing_data['train_label'], b])
#         sio.savemat('train_dataset_20.mat', existing_data)

# # 测试数据
# for ebno_db in range(23, 28, 1):  # 1-15 12000 16-30 12000 20-31 8800
#     b, ber = model(ebno_db=ebno_db, batch_size=200)
#     b = b.numpy()
#     ber = ber.numpy()
#     # 保存结果到.mat文件
#     if ebno_db == 0:
#         # 如果是第一次保存，直接创建文件
#         sio.savemat('test_dataset.mat', {'test_data': ber, 'test_label': b})
#     else:
#         # 如果文件已存在，直接追加数据
#         existing_data = sio.loadmat('test_dataset.mat')
#         existing_data['test_data'] = np.concatenate([existing_data['test_data'], ber])
#         existing_data['test_label'] = np.concatenate([existing_data['test_label'], b])
#         sio.savemat('test_dataset.mat', existing_data)
#####################################################################################################
b, ber = model(ebno_db=20, batch_size=400)
b = b.numpy()
ber = ber.numpy()
filepath = "train_dataset_20_mobile_20.mat"
if not os.path.exists(filepath):
    # 如果是第一次保存，直接创建文件
    sio.savemat('train_dataset_20_mobile_20.mat', {'train_data': ber, 'train_label': b})
else:
    # 如果文件已存在，直接追加数据
    existing_data = sio.loadmat('train_dataset_20_mobile_20.mat')
    existing_data['train_data'] = np.concatenate([existing_data['train_data'], ber])
    existing_data['train_label'] = np.concatenate([existing_data['train_label'], b])
    sio.savemat('train_dataset_20_mobile_20.mat', existing_data)

# b, ber = model(ebno_db=0, batch_size=2000)
# b = b.numpy()
# ber = ber.numpy()
# filepath = "test_dataset_0_mobile_20.mat"
# if not os.path.exists(filepath):
#     # 如果是第一次保存，直接创建文件
#     sio.savemat('test_dataset_0_mobile_20.mat', {'test_data': ber, 'test_label': b})
# else:
#     # 如果文件已存在，直接追加数据
#     existing_data = sio.loadmat('test_dataset_0_mobile_20.mat')
#     existing_data['test_data'] = np.concatenate([existing_data['test_data'], ber])
#     existing_data['test_label'] = np.concatenate([existing_data['test_label'], b])
#     sio.savemat('test_dataset_0_mobile_20.mat', existing_data)

# # shiyanyong
# b, ber = model(batch_size=2000)
# b = b.numpy()
# ber = ber.numpy()
# filepath = "test_dataset_20_mobile_20.mat"
# if not os.path.exists(filepath):
#     # 如果是第一次保存，直接创建文件
#     sio.savemat('test_dataset_20_mobile_20.mat', {'test_data': ber, 'test_label': b})
# else:
#     # 如果文件已存在，直接追加数据
#     existing_data = sio.loadmat('test_dataset_20_mobile_20.mat')
#     existing_data['test_data'] = np.concatenate([existing_data['test_data'], ber])
#     existing_data['test_label'] = np.concatenate([existing_data['test_label'], b])
#     sio.savemat('test_dataset_20_mobile_20.mat', existing_data)
######################################################################################################
# speed influence
# MOBILITY_SIMS = {
#     "ebno_db": list(np.arange(0, 32, 2.0)),
#     "cdl_model": "D",
#     "delay_spread": 100e-9,
#     "domain": "time",
#     "direction": "uplink",
#     "perfect_csi": [True, False],
#     "speed": [0, 20.0],
#     "cyclic_prefix_length": 6,
#     "pilot_ofdm_symbol_indices": [2],
#     "ber": [],
#     "bler": [],
#     "duration": None
# }
#
# # 9canshu
# start = time.time()
#
# for perfect_csi in MOBILITY_SIMS["perfect_csi"]:
#     for speed in MOBILITY_SIMS["speed"]:
#         model = Model(domain=MOBILITY_SIMS["domain"],
#                       direction=MOBILITY_SIMS["direction"],
#                       cdl_model=MOBILITY_SIMS["cdl_model"],
#                       delay_spread=MOBILITY_SIMS["delay_spread"],
#                       perfect_csi=perfect_csi,
#                       speed=speed,
#                       cyclic_prefix_length=MOBILITY_SIMS["cyclic_prefix_length"],
#                       pilot_ofdm_symbol_indices=MOBILITY_SIMS["pilot_ofdm_symbol_indices"])
#
#         ber, bler = sim_ber(model,
#                             MOBILITY_SIMS["ebno_db"],
#                             batch_size=2000,  # 256
#                             max_mc_iter=100,
#                             num_target_block_errors=1000)
#
#         MOBILITY_SIMS["ber"].append(list(ber.numpy()))
#         MOBILITY_SIMS["bler"].append(list(bler.numpy()))
#
# MOBILITY_SIMS["duration"] = time.time() - start
############################################################################################

# CP length
# CP_SIMS = {
#     "ebno_db": list(np.arange(0, 16, 1.0)),
#     "cdl_model": "C",
#     "delay_spread": 100e-9,
#     "subcarrier_spacing": 15e3,
#     "domain": ["freq", "time"],
#     "direction": "uplink",
#     "perfect_csi": False,
#     "speed": 10.0,
#     "cyclic_prefix_length": [20, 4],
#     "pilot_ofdm_symbol_indices": [2],
#     "ber": [],
#     "bler": [],
#     "duration": None
# }
#
# start = time.time()
#
# for cyclic_prefix_length in CP_SIMS["cyclic_prefix_length"]:
#     for domain in CP_SIMS["domain"]:
#         model = Model(domain=domain,
#                   direction=CP_SIMS["direction"],
#                   cdl_model=CP_SIMS["cdl_model"],
#                   delay_spread=CP_SIMS["delay_spread"],
#                   perfect_csi=CP_SIMS["perfect_csi"],
#                   speed=CP_SIMS["speed"],
#                   cyclic_prefix_length=cyclic_prefix_length,
#                   pilot_ofdm_symbol_indices=CP_SIMS["pilot_ofdm_symbol_indices"],
#                   subcarrier_spacing=CP_SIMS["subcarrier_spacing"])
#
#         ber, bler = sim_ber(model,
#                         CP_SIMS["ebno_db"],
#                         batch_size=256,
#                         max_mc_iter=100,
#                         num_target_block_errors=1000)
#
#         CP_SIMS["ber"].append(list(ber.numpy()))
#         CP_SIMS["bler"].append(list(bler.numpy()))
#
# CP_SIMS["duration"] = time.time() - start






















