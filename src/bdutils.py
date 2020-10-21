import numpy as np 
import struct
import os
from common import log

def save_beam_doses(fname, beam_doses):
    """
    :param beam_doses: must be np.array
    :return:
    """
    log.info("Saving beam doses to cache file: %s" % fname)
    bnint = struct.pack("i", np.prod(beam_doses.shape))

    fout = open(fname, "wb")
    fout.write(bnint)
    beam_doses.tofile(fout)
    fout.close()


# noinspection PyUnresolvedReferences
def read_beam_doses(fname, shape):
    beam_doses = None
    log.info("Reading beam doses from cache file: %s" % fname)
    if os.path.isfile(fname):
        fin = open(fname, "rb")
        bnint = fin.read(4)

        nint = struct.unpack("i", bnint)[0]

        if nint == np.prod(shape):
            beam_doses = np.fromfile(fin, np.float32, nint)
            beam_doses = np.reshape(beam_doses, shape)
        else:
            print("ERROR! Size of beam doses cache (%d) is not equal size of shape (%d)" % (nint, np.prod(shape)[0]))

        fin.close()

    return beam_doses



def save_beamlets_doses_map(fname, beamlets_doses):
    """
    :param treatment_name:
    :param beam_idx: integer - beam index
    :param beam_doses: must be np.array
    :return:
    """
    log.info("Saving beamlets doses to cache file: %s" % fname)
    nkeysint = struct.pack("i", len(beamlets_doses.keys()))

    fout = open(fname, "wb")
    fout.write(nkeysint)

    for btidx in sorted(beamlets_doses.keys()):
            fout.write(struct.pack("i", btidx))
            nrows = beamlets_doses[btidx].shape[0]
            fout.write(struct.pack("i", nrows))
            ncols = beamlets_doses[btidx].shape[1]
            fout.write(struct.pack("i", ncols))

            beamlets_doses[btidx].tofile(fout)

    fout.close()


# noinspection PyUnresolvedReferences
def read_beamlets_doses_map(fname):
    beamlets_doses_map = None
    log.info("Reading beamlets doses from cache file: %s" % fname)
    if os.path.isfile(fname):
        beamlets_doses_map = {}
        fin = open(fname, "rb")
        nkeys = struct.unpack("i", fin.read(4))[0]

        for i in range(nkeys):
            btidx = struct.unpack("i", fin.read(4))[0]
            nrows = struct.unpack("i", fin.read(4))[0]
            ncols = struct.unpack("i", fin.read(4))[0]

            beamlet_doses = np.fromfile(fin, np.float32, nrows*ncols)
            beamlet_doses = np.reshape(beamlet_doses, (nrows, ncols))

            beamlets_doses_map[btidx] = beamlet_doses

        fin.close()

    return beamlets_doses_map