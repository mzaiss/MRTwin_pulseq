import numpy as np
from utils import *
from vbvd_common import *

@assert_size_in_bytes(9736)
def MULTI_RAID_FILE_HEADER(): 
    return \
    np.dtype(
        [
            ('mr_parc_raid_file_header', MR_PARC_RAID_FILE_HEADER(), 1),
            ('mr_parc_raid_file_entry',  MR_PARC_RAID_FILE_ENTRY(),  64)
        ]
    )

@assert_size_in_bytes(8)
def MR_PARC_RAID_FILE_HEADER(): 
    return \
    np.dtype(
        [
            ('header_size', 'u4', 1),
            ('num_meas',    'u4', 1)
        ]
    )

@assert_size_in_bytes(152)
def MR_PARC_RAID_FILE_ENTRY():
    return \
    np.dtype(
        [
            # fieldName       dtype         count
            # =========       =====         =====
            ('meas_id',       'u4',          1),
            ('file_id',       'u4',          1),
            ('offset',        'u8',          1),
            ('length',        'u8',          1),
            ('patient_name',   np.bytes_,    64),
            ('protocol_name',  np.bytes_,    64)
        ]
    )

def SCAN_BLOCK(num_channels, samples_in_scan):
    return \
    np.dtype(
        [
            ('scan_header',   SCAN_HEADER(),   1),
            ('channel_block', CHANNEL_BLOCK(samples_in_scan), num_channels)
        ]
    )

@assert_size_in_bytes(192)
def SCAN_HEADER():
    return \
    np.dtype(
        [
            ('dma_length',              'u4',           1),
            ('meas_uid',                'u4',           1),
            ('scan_counter',            'u4',           1),
            ('timestamp',               'u4',           1),
            ('pmu_timestamp',           'u4',           1),
            ('system_type',             'u2',           1),
            ('ptab_pos_delay',          'u2',           1),
            ('ptab_pos_x',              'u4',           1),
            ('ptab_pos_y',              'u4',           1),
            ('ptab_pos_z',              'u4',           1),
            ('reserved_1',              'u4',           1),
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
            ('slice_data',              SLICE_DATA(),   1),
            ('ice_program_param',       'u1',          48),
            ('reserved_param',          'u1',           8),
            ('app_counter',             'u2',           1),
            ('app_mask',                'u2',           1),
            ('crc',                     'u4',           1),
        ]
    )


def CHANNEL_BLOCK(samples_in_scan):
    return \
    np.dtype(
        [
            ('channel_header', CHANNEL_HEADER(),    1),
            ('pixel_data',     np.complex64,        samples_in_scan)
        ]
    )

@assert_size_in_bytes(32)
def CHANNEL_HEADER():
    return \
    np.dtype(
        [
            ('type_and_channel_length', 'u4', 1),
            ('meas_uid',                'u4', 1),
            ('scan_counter',            'u4', 1),
            ('reserved1',               'u1', 4),
            ('sequence_time',           'u4', 1),
            ('unused_2',                'u1', 4),
            ('channel_id',              'u2', 1),
            ('unused_3',                'u1', 2),
            ('crc',                     'u4', 1)
        ]
    )




