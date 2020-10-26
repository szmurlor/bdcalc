import numpy as np
import struct
import os

def save_ndarray(fname, data):
    """
    :param data: must be np.array
    :return:

    Struktura pliku binarnego to: 
    n - liczba wymiarów
    1 ... n - rozmiary w kolejnych wymiarach
    dane....
    """
    print(f"Saving ndarray data with dimensions {data.shape} to file: {fname}")
    fout = open(fname, "wb")

    bdims = struct.pack("i", len(data.shape))
    fout.write(bdims)

    for s in data.shape:
        fout.write( struct.pack("i",s) )

    data.tofile(fout)
    fout.close()


# noinspection PyUnresolvedReferences
def read_ndarray(fname, dtype=np.float32):
    data = None
    print("Reading ndarray data from file: %s" % fname)
    if os.path.isfile(fname):
        fin = open(fname, "rb")
        bdim = fin.read(4)
        ndim = struct.unpack("i", bdim)[0]

        shape = []
        for i in range(ndim):
            bsize = fin.read(4)
            nsize = struct.unpack("i", bsize)[0]
            shape.append( nsize )
        
        print(f"ndarray data file shape: {shape}")

        data = np.fromfile(fin, dtype, np.prod(shape))
        data = np.reshape(data, shape)

        fin.close()

    return data