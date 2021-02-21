import numpy as np
from utils import *

assert_size_in_bytes(28)
def SLICE_DATA():
    return \
    np.dtype(
        [
            ('sag',        'f4', 1),
            ('cor',        'f4', 1),
            ('tra',        'f4', 1),
            ('quaternion', 'f4', 4)
        ]
    )

EVAL_INFO_BITFIELD = generate_bitfield(64)

EVAL_INFO_FLAGS = \
(
    np.array([
        'acq_end',
        'rt_feedback',
        'hp_feedback',
        'online',
        'offline',
        'sync_data',
        'unassigned_06',
        'unassigned_07',
        'last_scan_in_concat',
        'unassigned_09',
        'raw_data_correction',
        'last_scan_in_meas',
        'scan_scale_factor',
        'second_had_a_mar_pulse',
        'ref_phase_stab_scan',
        'phase_stab_scan',
        'd3_fft',
        'sign_rev',
        'phase_fft',
        'swapped',
        'post_shared_line',
        'phase_corr',
        'pat_ref_scan',
        'pat_ref_and_im_scan',
        'reflect',
        'noise_adj_scan',
        'share_now',
        'last_measured_line',
        'first_scan_in_slice',
        'last_scan_in_slice',
        'tr_effective_begin',
        'tr_effective_end',
        'mds_ref_position',
        'slc_averaged',
        'tag_flag_1',
        'ct_normalize',
        'scan_first',
        'scan_last',
        'unassigned_38',
        'unassigned_39',
        'first_scan_in_blade',
        'last_scan_in_blade',
        'last_blade_in_tr',
        'unassigned_43',
        'pace',
        'retro_last_phase',
        'retro_end_of_meas',
        'repeat_this_hearbeat',
        'repeat_prev_heartbeat',
        'abort_scan_now',
        'retro_last_heartbeat',
        'retro_dummy_scan',
        'retro_arr_det_disabled',
        'b1_control_loop',
        'skip_online_phase_corr',
        'skip_regridding',
        'unassigned_56',
        'unassigned_57',
        'unassigned_58',
        'unassigned_59',
        'unassigned_60',
        'unassigned_61',
        'unassigned_62',
        'unassigned_63'
    ])
)

def get_flags(num):
    mask  =  (num & EVAL_INFO_BITFIELD) > 0
    flags =  EVAL_INFO_FLAGS[mask]
    return flags

def get_bitfield(flags):
    idx = np.zeros(len(flags), dtype=int)
    for i,flg in enumerate(flags):
        idx[i] = list(EVAL_INFO_FLAGS).index(flg)

    val = np.bitwise_or.reduce(EVAL_INFO_BITFIELD[idx])

    return val


def check_flag(arr, flag_name):
     idx = np.where(EVAL_INFO_FLAGS == flag_name)[0][0]
     tf = ( (1 << idx) & arr ) > 0
     return tf


def filter_flags(arr, good_flags = [], bad_flags = []):

    # Filter bitfield arr so that the subarray returned has all of the
    # good flags and none of the bad flags

    total_mask = np.full(arr.shape, True)
    
    for flag in good_flags:
        flag_mask   = check_flag(arr, flag) 
        total_mask  = np.logical_and(total_mask, flag_mask)
    
    for flag in bad_flags:
        flag_mask  = np.logical_not(check_flag(arr, flag))
        total_mask = np.logical_and(total_mask, flag_mask)

    return arr[total_mask]