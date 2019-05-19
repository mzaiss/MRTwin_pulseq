import numpy as np
from utils import *
from vbvd_common import *


def SCAN_BLOCK(num_channels, samples_in_scan):
    return \
    np.dtype(
        [
            ('channel_block', CHANNEL_BLOCK(samples_in_scan), num_channels)
        ]
    )

@assert_size_in_bytes(128)
def SCAN_HEADER():
    return \
    np.dtype(
        [
            ('dma_length',              'u4',           1),
            ('meas_uid',                'u4',           1),
            ('scan_counter',            'u4',           1),
            ('timestamp',               'u4',           1),
            ('pmu_timestamp',           'u4',           1),           
            ('eval_info_mask',          'u8',           1),
            ('samples_in_scan',         'u2',           1),
            ('used_channel',            'u2',           1),
            ('line',                    'u2',           1),
            ('acq',                     'u2',           1),
            ('slc',                     'u2',           1),
            ('par',                     'u2',           1),
            ('eco',                     'u2',           1),
            ('phs',                     'u2',           1),
            ('rep',                     'u2',           1),
            ('set',                     'u2',           1),
            ('seg',                     'u2',           1),
            ('ida',                     'u2',           1),
            ('idb',                     'u2',           1),
            ('idc',                     'u2',           1),
            ('idd',                     'u2',           1),
            ('ide',                     'u2',           1),
            ('pre',                     'u2',           1),
            ('post',                    'u2',           1),
            ('k_space_center_column',   'u2',           1),
            ('coil_select',             'u2',           1),
            ('readout_offcenter',       'u4',           1),
            ('time_since_last_rf',      'u4',           1),
            ('k_space_center_line_num', 'u2',           1),
            ('k_space_center_par_num',  'u2',           1),
            ('ice_program_param',       'u1',           8),
            ('free_param',              'u1',           8),
            ('slice_data',              SLICE_DATA(),   1),
            ('channel_id',              'u2',           1),
            ('ptab_pos_neg',            'u2',           1)
        ]
    )

def CHANNEL_BLOCK(samples_in_scan):
    return \
    np.dtype(
        [
            ('scan_header', SCAN_HEADER(),    1),
            ('pixel_data',     np.complex64,        samples_in_scan)
        ]
    )



