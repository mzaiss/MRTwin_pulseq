#!/usr/bin/env python3

import os
import struct
import re
import time
import errno
import shutil
import threading
import tempfile
from sys import platform
from socketserver import TCPServer, BaseRequestHandler


class DataCatcher(BaseRequestHandler):

    ADDRESS = '', 6666
    BUFFER_SIZE = 1024 * 32

    def setup(self):
        """Parse Protocol and Prepare a file for writing"""
        # Get Protocol Length and Protocol String (Raw & Converted -> Sending & Parsing)
        protocol_length_raw = self.recv_all(4)
        protocol_length = struct.unpack('<l', bytes(protocol_length_raw))[0] - 4
        protocol_raw = self.recv_all(protocol_length)
        protocol = str(protocol_raw)

        # Get Measurement ID from Protocol
        self.id = int(self.protfind(protocol, '<ParamLong."MeasUID">'))

        # Get Patient/Protocol Name from Protocol -> Create Directory & File Path
        patient_name = self.protfind(protocol, '<ParamString."tPatientName">')
        protocol_name = self.protfind(protocol, '<ParamString."tProtocolName">')
        date = time.strftime('%Y%m%d')

	# get twix save path from control file
	if platform == 'linux':
	    #basepath = '/media/upload3t/CEST_seq/pulseq_zero/sequences'
	    basepath = '/is/ei/aloktyus/Desktop/pulseq_mat_py'
	else:
	    basepath = '???'

	today_datestr = time.strftime('%y%m%d')
	with open(os.path.join(basepath,"position.txt"),"r") as f:
	    position = int(f.read())
	    
	with open(os.path.join(basepath,"control.txt"),"r") as f:
	    control_lines = f.readlines()
	    
	control_lines = [l.strip() for l in control_lines]
	if position >= len(control_lines) or len(control_lines) == 0 or control_lines.count('wait') > 1 or control_lines.count('quit') > 1:
	    self.print("ERROR: control file is corrupt")
	    raise
	    
	if control_lines[position] == 'wait' or control_lines[position] == 'quit':
	    self.print("ERROR: control file is corrupt")
            raise

	# get current twix target path at position
	twix_path = control_lines[position]
	twix_dir = os.path.dirname(twix_path)
	fn = os.path.basename(twix_path)
	self.twix_fn = os.path.join(twix_dir,"data",fn[:-4]+".dat")
	self.twix_temp = os.path.join(twix_dir,"data","temp.dat")

        self.file = open(self.twix_temp, mode='wb')

        # Write Protocol to File
        self.file.write(protocol_length_raw)
        self.file.write(protocol_raw)

        self.print("File: {}".format(self.twix_temp))

    def handle(self):
        """Write Incoming Stream to File"""

        # Statistics
        REFRESH_RATE = 2
        time_start = time.time()
        bytes_delta = 0

        while True:
            # Receive Data from Scanner and Write Data to File
            data = self.request.recv(self.BUFFER_SIZE)
            if not data:
                break
            else:
                self.file.write(data)

            # Statistics
            time_delta = time.time() - time_start
            bytes_delta += len(data)

            if time_delta > REFRESH_RATE:
                self.print("Streaming: {:3.1f} MB/s".format((bytes_delta /
                                                             time_delta) / (1024**2)), update=True)
                time_start = time.time()
                bytes_delta = 0

    def finish(self):
        self.print("Done Streaming")
        threading.Thread(target=self.copy).start()

    def copy(self):
        self.print("Moving to {}".format(self.twix_fn))
        try:
            shutil.move(self.twix_temp, self.twix_fn)
            self.print("Moved to to {}".format(self.twix_fn))
        except:
            self.print("ERROR: Could not Move to {}".format(self.twix_fn))
        self.file.close()

    def print(self, message: str, update=False):
        """Print Formatted Message to User"""
        if update:
            print("\r[{}][MID {}][DataCatcher][{}]".format(
                time.strftime('%H:%M:%S'), self.id, message), end="")
        else:
            print("\r[{}][MID {}][DataCatcher][{}]".format(
                time.strftime('%H:%M:%S'), self.id, message))

    def protfind(self, protocol: str, query: str):
        """
        Find Field in Protocol
        :param protocol: protocol to search in
        :param query: query to find
        :return: string value corresponding to query
        """
        p_attribute = protocol.find(query)
        p_bracket_left = protocol.find('{', p_attribute) + 1
        p_bracket_right = protocol.find('}', p_bracket_left)
        return protocol[p_bracket_left: p_bracket_right].replace('\"', '').strip()

    def recv_all(self, n_bytes: int):
        """
        Receive Exactly n_bytes
        :param n_bytes: number of bytes to receive
        :return: byte[n_bytes]
        """
        total_data = bytearray()

        while True:
            data = self.request.recv(n_bytes - len(total_data))

            if not data:
                return total_data
            else:
                total_data.extend(data)

            if len(total_data) == n_bytes:
                return total_data

    def create_folder(self, path: str):
        """Create a folder and mute 'EEXIST' errors"""
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise


if __name__ == "__main__":
    print("DataCatcher online")
    print(" listen to {}".format(DataCatcher.ADDRESS))
    print("-" * 40)
    TCPServer(DataCatcher.ADDRESS, DataCatcher).serve_forever()

