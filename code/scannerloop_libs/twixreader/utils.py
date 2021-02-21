import numpy as np
from io import StringIO

try :
    from colorama import Fore
except:
    class ForeDummy:
        def __getattribute__(self, attr):
            return ''

    Fore = ForeDummy()

NULL_BYTE = b'\x00'

def read_null_terminated_string(f, max_len=128):
    string = b''
    next_char = b''
    while next_char != NULL_BYTE:
        string = string + next_char
        next_char = f.read(1) 
        assert len(string) < max_len
    
    return string

def generate_bitfield(n):
    bf = 1 << np.arange(n)
    return bf

def eval_bitfield(x, bitfield):
    return (x & bitfield) > 0



class assert_size_in_bytes:

    def __init__(self, byte_size):
        self.byte_size = byte_size
    
    def __call__(self, f):
        def wrapped_f(*args,**kwargs):
            val  = f(*args, **kwargs)
            assert  type(val) is np.dtype, \
            'expected numpy.dtype not {0}'.format(type(val))

            assert  val.itemsize == self.byte_size,\
            'expected {0} bytes, found {1} bytes'.format(self.byte_size, 
                                                    val.itemsize)
        

            return val

        return wrapped_f

def arr2str(x):
    outstring = ''
 
    for name in x.dtype.names:
        val = x[name]
        if type(val) is np.void:
            val = bytes(val)
        if type(val) is bytes:
            valstring = '{0}B bytestring'.format(len(val))
        else:
            valstring = str(val)

        outstring = outstring + Fore.LIGHTBLUE_EX+name+': ' + Fore.RESET + valstring + '\n'
    
    return outstring

def print_arr(x):

    print(arr2str(x))

def prettyvars(obj, skip_private = True):
    d = vars(obj)
    for key,val in d.items():
        if key[0] == '_':
            continue
        print(Fore.LIGHTBLUE_EX + key + ': ' + Fore.RESET + str(val))

def print_table(data, headers=None, col_width=16):
    io = StringIO()
    num_cols = len(data[0])
    fmt = '{:>' + str(col_width) + '}'
    if headers:
        print(Fore.LIGHTBLUE_EX + \
              (fmt*num_cols).format(*headers, col_width=col_width) + \
              Fore.RESET, \
              file=io
        )
        

    for row in data:
        val_strings = tuple(map(str, row))
        print((fmt*num_cols).format(*val_strings, col_width=col_width), file=io)

    io.seek(0)
    s = io.read()
    print(s)
    return s

def ensure_not_none(x, default_val):
    if x is None:
        return default_val
    return x

def has_len(x):
    try:
        len(x)
        return True
    except:
        return False

def read_from_bytearr(bytearr, dtype = 'uint8', offset = 0, count = 1):

    dtype     = np.dtype(dtype)
    num_bytes = dtype.itemsize*count
    val = bytearr[offset:(offset+num_bytes)].view(dtype)

    return val

def split_bytearr(bytearr, offset, size):
    """
    e.g. offset = [0,  1000, 2000]
         size   = 192

    would read bytearr[0:192], bytearr[1000:1192], bytearr[2000:2192]
    and concatenate into one bytearr

    size can also be an array, e.g. size = [192, 200, 300]

    """

    split_inds       = np.empty(2*len(offset), dtype='u8')
    split_inds[0::2] = offset
    split_inds[1::2] = offset + size

    byte_groups = np.split(bytearr, split_inds)
    val = np.concatenate(byte_groups[1::2])

    return val





