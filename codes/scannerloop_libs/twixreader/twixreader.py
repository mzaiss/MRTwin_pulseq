import numpy as np
import io
from numpy.lib.recfunctions import *
from utils import *
import vdspec as vd
import vbspec as vb
import vbvd_common as vbvd
import os
from header_parser import header_parser as hp
from json_html_viewer import json_html_viewer as jview

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def detect_vb_or_vd(datpath):
    
    header_size = np.fromfile(datpath, 'u4', 1)[0]

    filecode = \
    {   #       (ver.,   filetype,                    comment)
        0   :   ('VD',   'MR_PARC_RAID_ALLDATA',      'normal VD11A file'),
        1   :   ('VD',   'MR_PARC_RAID_MDHONLY',      'compact file (meas data removed)'),
        2   :   ('VD',   'MR_PARC_RAID_HDONLY',       'file only with Multi-RAID and buffer header'),
        32  :   ('VA',   'MR_PARC_RAID_LEGACY_THR',   'pre VD11A file without buffer header (no RAID)')
    }

    if header_size in filecode:
        val = filecode[header_size]
    else:
        val =('VB', '', 'pre VD11A file with buffer header')

    print('{:<10}: {}'.format('VERSION',    val[0]))
    print('{:<10}: {}'.format('FILE TYPE',  val[1]))
    print('{:<10}: {}'.format('COMMENT',    val[2]))

    return val

def read_twix(datpath):
    ver, filetype, comment = detect_vb_or_vd(datpath)
    twix_reader_select = dict(VB = TwixReaderVB, VD = TwixReaderVD)
    return twix_reader_select[ver](datpath)
    
class TwixReader:

    def __init__(self, datpath):
        self.datpath = os.path.abspath(datpath)
        self.twix_map = np.memmap(self.datpath, mode = 'r')
        
class TwixReaderVB(TwixReader):

    def __init__(self, datpath):
        super().__init__(datpath)
        self.num_meas = 1

    def read_measurement(self, header_only=False, parse_buffers = True):
        meas = MeasurementVB(self.twix_map, header_only=header_only, parse_buffers = parse_buffers)
        return meas

    def vers(self):
        return 'VB'
        
        return meas

class TwixReaderVD(TwixReader):

    def __init__(self,datpath):
        super().__init__(datpath)
        dtype = vd.MULTI_RAID_FILE_HEADER()
        multi_raid_file_header = self.twix_map[0:dtype.itemsize].view(dtype).item()        
        self.num_meas = multi_raid_file_header[0][1]
        self._mr_parc_raid_file_entry = np.rec.array(multi_raid_file_header[1][0:self.num_meas])

    def vers(self):
        return 'VD'

    @property
    def meas_names(self):
        names = [s.decode('UTF-8') for s in self._mr_parc_raid_file_entry.protocol_name]
        return names

    def _read_all_measurements(self, header_only=False):
        val = [self.read_measurement(i, header_only=header_only) for i in range(self.num_meas)]
        return val

    def read_measurement(self, meas_num=None, header_only = False, parse_buffers = True):
        
        if meas_num is None:
            return self._read_all_measurements(header_only=header_only)

        file_entry = self._mr_parc_raid_file_entry[meas_num]
        meas_offset = file_entry['offset']
        meas_length = file_entry['length']
        meas_id     = file_entry['meas_id']
        meas_map = self.twix_map[meas_offset:(meas_offset+meas_length)]

        meas = MeasurementVD(meas_map, header_only = header_only, parse_buffers = parse_buffers)
        meas.mid = meas_id
        
        return meas


class Measurement:

    dtype_scan_header = None
    
    def __init__(self, meas_map, header_only = False, parse_buffers = True):
        self.meas_map = meas_map
        self.header_size, self.num_buffers = read_from_bytearr(self.meas_map, 
                                                               dtype='u4', count = 2)
        buffer_offset = 8
        meas_hdr_map = self.meas_map[buffer_offset:self.header_size]
        self.hdr = _MeasurementHeader(meas_hdr_map, self.num_buffers,
                                      parse_buffers = parse_buffers)
        self.hdr.set_parent(self)
        
        if header_only:
            return

        self._bytearr = self.meas_map[self.header_size::]

        self._all_mdh = self._read_all_mdh()
        self._all_mdh = self.remove_non_image_scans(self._all_mdh)
        
        #print('Skipping to first imaging measurement...')
        #self._all_mdh = self.skip_to_first_meas(self._all_mdh)

        self.mid = int( np.median(self._all_mdh['meas_uid']) )
        self.split_by()
        
    def _read_all_mdh(self):

        bytearr = self._bytearr

        pos = 0
        counter = 1

        dma_length_arr = []

        for (counter, dma_length) in self._read_dma_length(bytearr):
            print('Reading MDH {0}'.format(counter),end='\r')
            dma_length_arr.append(dma_length)
        
        print('\n')
        
        mempos = np.zeros(len(dma_length_arr), dtype=int)
        mempos[1::] += np.cumsum(dma_length_arr[0:-1]).astype(int)

        dtype      = self.dtype_scan_header

        mdh_arr = split_bytearr(bytearr, mempos, dtype.itemsize).view(dtype)
        
        mdh_arr['dma_length'] = dma_length_arr
        mdh_arr = append_fields(mdh_arr, 'mempos', mempos, usemask=False)

        return mdh_arr

    def select_by(self, mdh_flags = {}, 
                        contains_eval_info_flags = (),
                        excludes_eval_info_flags = ('sync_data',
                                                    'acq_end',
                                                    'ct_normalize')
                 ):
        """
        select a measurement buffer that satisfies certain flag criteria
        ----------
        mdh_flags : dict
            Contains key/value pairs describing the values that mdh flags should 
            have in order to be selected
        contains_eval_info_flags : tuple,list
            list of flags that should be included
        excludes_eval_info_flags : tuple,list
            list of flags that should not be present

        Returns
        -------
        out : MeasurementBuffer object
            Returns a measurement buffer containing only the measurements that
            match the selection criteria

        Examples
        --------
        meas_object.select_by({'samples_in_scan': 512,
                               'used_channel': 12},      # MDH flags must have these values
                               ( ),                      # No required eval_info_mask fields
                               ('acq_end', 'sync_data')) # eval_info_mask should NOT contain these flags
        """

        mdh = self._all_mdh

        # Keep MDH's that match these fields
        for key,val in mdh_flags.items():
            idx = (mdh[key] == val)
            mdh = mdh[idx]

        # Keep MDH's that have these eval_info_mask properties
        for flag_name in contains_eval_info_flags:
            idx = vbvd.check_flag(mdh['eval_info_mask'], flag_name)
            mdh = mdh[idx]

        # Keep MDH's that do NOT have these eval_info_mask properties
        for flag_name in excludes_eval_info_flags:
            idx = ~vbvd.check_flag(mdh['eval_info_mask'], flag_name)
            mdh = mdh[idx]

        buf = self._create_measurement_buffer(mdh)
        
        return buf


    @staticmethod
    def _read_dma_length(bytearr, pos=0):
        
        counter = 0
        LEAST_SIGNIFICANT_25_BITS = (1 << 25) - 1

        while pos < (len(bytearr)-192) :
            dma_length = read_from_bytearr(bytearr, dtype='u4', offset=pos)[0] 
            dma_length = dma_length & LEAST_SIGNIFICANT_25_BITS
            pos += int(dma_length)
            counter += 1

            yield counter, dma_length
                    
    def group_info(self):
        print(self._group_info)

    def split_by(self, field_names=['dma_length', 'used_channel', 'samples_in_scan','eco']):

        mdh_arr = self._all_mdh
        vals = mdh_arr[field_names]

        unique_vals = np.unique(vals)

        mdh_groups = []
        ind_groups = []

        for uval in unique_vals:
            idx = np.where(vals==uval)
            g = mdh_arr[idx]
            # Skip scans with 0 samples
            mdh_groups.append(g)
        
        self._group_info = self._print_group_info(mdh_groups, field_names)
        self._mdh_groups = mdh_groups

    @staticmethod
    def _print_group_info(mdh_groups, field_names):

        table_headers = ['Group','# Scans'] + field_names
        table_rows = []
        for i,g in enumerate(mdh_groups):
            table_rows.append([i,len(g)]+[g[0][fld] for fld in field_names])
        
        val = print_table(table_rows, table_headers)
        return val

    def split_by_eval_info_mask(self):

        mdh_arr = self._all_mdh
        vals = mdh_arr['eval_info_mask']

        unique_vals = np.unique(vals)

        mdh_groups = []
        ind_groups = []

        counter = 0
        for uval in unique_vals:
            idx = np.where(vals==uval)
            g = mdh_arr[idx]
            # Skip scans with 0 samples
            mdh_groups.append(g)
            flags = vbvd.get_flags(uval)
            print('Group {0}: '.format(counter) + ', '.join(map(str, flags)))
            counter += 1      

        self._mdh_groups = mdh_groups

    def get_meas_buffer(self, group_num):
        
        try:
            mdh_group = self._mdh_groups[group_num]
        except:
            mdh_group = np.concatenate([self._mdh_groups[i] for i in group_num])

        return self._create_measurement_buffer(mdh_group)
               

    def filter_flags(self, mdh_arr, bad_flags=['ct_normalize']):

        for flg in bad_flags:
            ind = np.logical_not(vbvd.check_flag(mdh_arr['eval_info_mask'], flg))
            mdh_arr = mdh_arr[ind]

        return mdh_arr

    def skip_to_first_meas(self, mdh_arr):

        hasflag = lambda flag_name: vbvd.check_flag(mdh_arr['eval_info_mask'], flag_name)
        ind = np.min(np.where(hasflag('first_scan_in_slice'))[0])
        mdh_arr = mdh_arr[ind::]

        return mdh_arr


    def remove_non_image_scans(self, mdh_arr):

        hasflag = lambda flag_name: vbvd.check_flag(mdh_arr['eval_info_mask'], flag_name)
        
        mask = (        hasflag('acq_end') 
                    |   hasflag('rt_feedback') 
                    |   hasflag('hp_feedback') 
                    |   hasflag('phase_corr') 
                    |   hasflag('noise_adj_scan') 
                    |   hasflag('phase_stab_scan')
                    |   hasflag('ref_phase_stab_scan')
                    |   hasflag('sync_data')
                    |   ( hasflag('pat_ref_scan') & ~hasflag('pat_ref_and_im_scan') )
                )

        return mdh_arr[~mask]


class MeasurementVD(Measurement):
    
    dtype_scan_header = vd.SCAN_HEADER()

    def _create_measurement_buffer(self, mdh_group):
        return _MeasurementBufferVD(mdh_group, self._bytearr)
    

class MeasurementVB(Measurement):
    dtype_scan_header = vb.SCAN_HEADER()

    def _create_measurement_buffer(self, mdh_group):
        return _MeasurementBufferVB(mdh_group, self._bytearr)

class _MeasurementBuffer:

    SCAN_HEADER_LENGTH = None
    mdh_dim_order = ('ide', 'idd', 'idc', 'idb', 'ida', 
                     'seg', 'set', 'rep', 'eco', 'phs', 
                     'acq', 'slc', 'par', 'line')

    dim_order = mdh_dim_order + ('cha', 'col') # opposite of MATLAB order (C order)
    
    _ndim = len(dim_order)

    def __init__(self, mdh_arr, bytearr):
        self.mdh     = mdh_arr 
        self._num_mdh = len(self.mdh)
        self.bytearr = bytearr
        self.num_pixels = self.mdh[0]['samples_in_scan']
        self.num_channels = self.mdh[0]['used_channel']
        self.squeeze_dims = True
        self.sort_data    = True
        self.reshape_data = True
        self.mdh_use_dims = _MeasurementBuffer.mdh_dim_order[6::]
               
    def __len__(self):
        return self._num_mdh
     
    @property
    def shape(self):
        dim_sizes = np.zeros(self._ndim, dtype=int)

        dim_sizes[-1] = self.mdh[0]['samples_in_scan']
        dim_sizes[-2] = self.mdh[0]['used_channel']

        for i,d in enumerate(self.mdh_use_dims):
            dim_sizes[i] = len(np.unique(self.mdh[d]))
        
        if self.squeeze_dims == True:
            dim_sizes = dim_sizes[dim_sizes>1]

        return tuple(dim_sizes)

    @ property
    def mdh_shape(self):
        return self.shape[0:-2]
        
    @property
    def ndim(self):
        return len(self.shape)

    def get_sort_idx(self, mdh_arr):
        keys = np.vstack(mdh_arr[k] for k in self.mdh_use_dims)
        idx = np.lexsort(keys)
        return idx

    def _getitem_helper(self, key):

        mdh_nd = self.mdh.reshape(self.mdh_shape)
        mdh_subarr_nd = mdh_nd.__getitem__(key)
        final_shape = mdh_subarr_nd.shape + (self.num_channels, self.num_pixels)
        
        mdh_subarr = mdh_subarr_nd.ravel()
        mdh_subarr = np.atleast_1d(mdh_subarr)
        if self.sort_data:
            idx = self.get_sort_idx(mdh_subarr)
            mdh_subarr = mdh_subarr[idx]
        
        num_requested   = len(mdh_subarr)

        scan_data = split_bytearr(self.bytearr, 
                                  offset = mdh_subarr['mempos'], 
                                  size = mdh_subarr['dma_length']).view(self.scan_dtype)

        scan_data = scan_data['channel_block']['pixel_data'].reshape(final_shape)

        return scan_data
    
    def _unsorted(self, key):

        mdh_subarr = self.mdh.__getitem__(key)
        mdh_subarr = np.atleast_1d(mdh_subarr)

        num_requested   = len(mdh_subarr)
        final_shape = (num_requested, self.num_channels, self.num_pixels)

        scan_data = split_bytearr(self.bytearr, 
                                  offset = mdh_subarr['mempos'], 
                                  size = mdh_subarr['dma_length']).view(self.scan_dtype)

        scan_data = scan_data['channel_block']['pixel_data'].reshape(final_shape)

        return scan_data

    def __getitem__(self,idx):
        
        if self.reshape_data == False:
            return self._unsorted(idx)

        if has_len(idx) and (len(idx) > (self.ndim-2)):
            data = self._getitem_helper(idx[0:-2])
            data = np.array(data, ndmin = self.ndim)
            data = data.__getitem__(idx)
        else:
            data = self._getitem_helper(idx)

        return data
    
class _MeasurementBufferVD(_MeasurementBuffer):

    SCAN_HEADER_LENGTH = 192

    def __init__(self, mdh_arr, bytearr):
        super().__init__(mdh_arr, bytearr)
        self.scan_dtype = vd.SCAN_BLOCK(self.num_channels, self.num_pixels)

class _MeasurementBufferVB(_MeasurementBuffer):

    SCAN_HEADER_LENGTH = 128

    def __init__(self, mdh_arr, bytearr):
        super().__init__(mdh_arr, bytearr)
        self.scan_dtype = vb.SCAN_BLOCK(self.num_channels, self.num_pixels)

class _MeasurementHeader:

    def __init__(self, meas_hdr_map, num_buffers, parse_buffers=True):
        self._buffers = dict()
        self._raw_buffers = dict()
        self.meas_hdr_map = meas_hdr_map
        self.num_buffers = num_buffers    
                
        fileobj = io.BytesIO(self.meas_hdr_map.tobytes())
                
        for i in range(self.num_buffers):
            buffer_name   = read_null_terminated_string(fileobj).decode('UTF-8')
            buffer_length, = np.frombuffer(fileobj.read(4), 'u4', 1)
            buffer_text = fileobj.read(int(buffer_length))
            self._raw_buffers[buffer_name] = buffer_text.decode('UTF-8')

        if parse_buffers:
            self._parse_raw_buffers()


    def _parse_raw_buffers(self):
        for key,val in self._raw_buffers.items():
            print('Parsing {}'.format(key))
            self._buffers[key],_ = hp.parse_string(val)

    def __getitem__(self, key):
        return self._buffers.__getitem__(key)

    def keys(self):
        return self._buffers.keys()

    def __repr__(self):
        s = '\n'.join(['<MeasurementBuffer.' + key + '>' for key in self.keys()])
        return s

    def set_parent(self, parent):
        self._parent = parent

    def view(self):
        if jview:
            jview.view(self._buffers)
        else:
            print(self._buffers)

    def dump(self, filepath=None, file_ext='json'):

        if filepath is None:
            file_namer = lambda key: 'MID{0:05}_{1}.{2}'.format(self._parent.mid, key, file_ext)
        else:
            filepath = os.path.splitext(filepath)[0]
            file_namer = lambda key: '{0}_{1}.{2}'.format(filepath, key, file_ext)

        savefun_and_buffer_select = dict(yaml = (hp.dict2yaml, self._buffers), 
                                         txt  = (hp.dict2txt, self._raw_buffers),
                                         json = (hp.dict2json, self._buffers),
                                         html = (jview.write,  self._buffers)
                                    )
        if file_ext not in savefun_and_buffer_select:
            file_ext = 'json'
        
        savefun, buffer_dict = savefun_and_buffer_select[file_ext]
        
        savefun(buffer_dict, file_namer('header'))


if __name__ == '__main__':
    
    pass


    
