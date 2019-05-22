from . import twixreader 
from .json_html_viewer import json_html_viewer as jview
import os

def twix2html(twixpath, save = False):
    tr = twixreader.read_twix(twixpath)

    html_code = []

    for meas_num in range(tr.num_meas):

        if tr.vers() == 'VD':
            meas = tr.read_measurement(meas_num, header_only=True)
        else:
            meas = tr.read_measurement(header_only=True)

        d = meas.hdr._buffers
        
        html_writer = jview.HTML_Writer()
        html_code_meas = html_writer.get_html(d)

        if save:
            with open(os.path.splitext(twixpath)[0] + '_{:02}.html'.format(meas_num), 'w+') as f:
                f.write('{}'.format(html_code_meas))
            return

        html_code.append(html_code_meas)

    return html_code

def get_raw_buffers(twixpath, meas_num=0):
    tr = twixreader.read_twix(twixpath)

    if tr.vers() == 'VD':
        meas = tr.read_measurement(meas_num, header_only=True, parse_buffers=False)
    else:
        meas = tr.read_measurement(header_only=True, parse_buffers=False)

    return meas.hdr._raw_buffers