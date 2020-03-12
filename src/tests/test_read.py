import numpy as np
import sys
import struct

def read_doses(filename):
    f = open(filename, 'rb')
    n = struct.unpack_from('i', f.read(4), 0)[0]

    v = np.fromfile(f, np.int32, n)
    b = np.fromfile(f, np.int32, n)
    d = np.fromfile(f, np.float32, n)

    return v,b,d

v,b,d = read_doses("aa.cache")

print(v)
print(b)
print(d)

