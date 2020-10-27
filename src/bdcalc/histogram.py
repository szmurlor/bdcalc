# -*- coding: utf-8 -*-
from sys import argv
import numpy as np
import re
import os

import time


def histogram(doses, markerArray, sid, fname, scale, npts, dmax):
    start = time.time()
    d = doses[(markerArray & sid) == sid]
    if len(d) < 1:
        return (0, 0, 0)

    vol = len(d)
    hist = [(0., 100.)]
    for s in range(npts):
        treshold = dmax * (s + 1) / npts
        i = np.sum(d < treshold)
        v = 1. - (i - 1.) / vol
        hist.append((scale * treshold, v * 100.))

    f = open(fname, 'w')
    for p in hist:
        f.write('%f %f\n' % p)
    f.close()

    end = time.time()
    print("Histogram generated in %f seconds to the file %s." % (end - start, fname))

    return np.min(d), np.average(d), np.max(d)


class DosesMain:
    def __init__(self, fname):
        self.override_fluences_filename = None
        self.override_output_folder = None
        self.preview_fluence = None
        self.save_png_preview_fluence = None
        self.path = os.path.dirname(fname)
        fin = open(fname, mode='r')
        self.treatment_name = fin.readline().rstrip()
        self.bno = self.read_1_int(fin)
        self.bnos = np.array(range(self.bno), dtype=np.int32)
        self.bsizes = np.array(range(self.bno), dtype=np.int32)
        for i in range(self.bno):
            self.bnos[i], self.bsizes[i] = self.read_2_int(fin)

        self.vno = self.read_1_int(fin)

        self.dosegridscaling = self.read_1_float(fin)
        self.roino = self.read_1_int(fin)
        self.roinames = []
        self.roiids = []
        for i in range(self.roino):
            cols = re.split(r'\s+', fin.readline(), 1)
            self.roiids.append(int(cols[0]))
            self.roinames.append(cols[1].rstrip())

        self.btno = np.sum(self.bsizes)

        self.voxels = None # self.read_voxels()
        self.x = None # self.read_fluences()
        self.xcoords = None
        self.D = None # self.read_doses()

        self.d = None # self.D.dot(self.x)

    def get_fluences_filename(self, i):
        if self.override_fluences_filename is not None:
            # if overriden name starts with / or . the pareto data path will not be attached
            if (self.override_fluences_filename[0] in '/.'):
                fname = '%s%d.txt' % (self.override_fluences_filename, i)
            else:
                fname = '%s/%s%d.txt' % (self.path, self.override_fluences_filename, i)
        else:
            fname = '%s/x_%s_%d.txt' % (self.path, self.treatment_name, i)

        return fname

    def read_fluences(self):
        res = np.zeros((self.btno,), dtype=np.float32)
        j = 0
        start_counting_from = 0
        fn = self.get_fluences_filename(start_counting_from)
        if (not os.path.isfile(fn)):
            start_counting_from += 1
            fn = self.get_fluences_filename(start_counting_from)

        print(f"Starting counting fluence files from: {start_counting_from}")

        for i in range(self.bno):
            nbeam = self.bnos[i]
            fn = self.get_fluences_filename(start_counting_from + nbeam - 1)
            print(f"Reading fluences for beam {nbeam} from file: {fn}")
            fbeam = open(fn)

            for k in range(self.bsizes[i]):
                s = fbeam.readline()
                cols = s.split(' ')
                if len(cols) == 1: # sometimes fluence map files have additional ordinal column 
                    res[j] = float(cols[0])
                else:
                    res[j] = float(cols[1])
                j += 1
        return res

    def read_xcoords(self):
        res = {}
        for i in range(self.bno):
            nbeam = self.bnos[i]
            res[nbeam] = {}

            f = open('%s/xcoords_%s_%d.txt' % (self.path, self.treatment_name, nbeam))
            f.readline() # skip
            res[nbeam]["sizex"] = int(self.read_1_float(f))
            f.readline() # skip
            res[nbeam]["sizey"] = int(self.read_1_float(f))
            f.readline() # skip
            res[nbeam]["spacingx"] = self.read_1_float(f)
            f.readline() # skip
            res[nbeam]["spacingy"] = self.read_1_float(f)
            f.readline() # skip
            res[nbeam]["originx"] = self.read_1_float(f)
            f.readline() # skip
            res[nbeam]["originy"] = self.read_1_float(f)
            for k in range(self.bsizes[i]):
                x, y = self.read_2_int(f)
                res[nbeam]["%dx%d" % (x, y)] = 1

        return res

    def read_voxels(self):
        res = np.array(range(self.vno), dtype=np.int32)
        f = file('%s/v_%s.txt' % (self.path, self.treatment_name))
        for k in range(self.vno):
            line = f.readline()
            sr, sx, sy, sz = line.split()
            #sr, = line.split()

            res[k] = int(sr)
        return res

    def read_doses(self):
        from scipy.sparse import dok_matrix
        res = dok_matrix((self.vno, self.btno),dtype=np.float32)
        #res = np.zeros((self.vno, self.btno), dtype=np.float32)
        start_col = 0
        for i in range(self.bno):
            nbeam = self.bnos[i]
            bsize = self.bsizes[i]
            f = file('%s/d_%s_%d.txt' % (self.path, self.treatment_name, nbeam))
            count = self.read_1_int(f)
            print("Reading doses for beam no %d (size: %d)" % (nbeam, count))
            for k in range(count):
            #for k in range(200000):
                v, b, d = self.read_3_int(f)
                res[v, start_col + b] = float(d)
            start_col += bsize
        return res

    def histogram(self):
        if self.voxels is None:
            self.voxels = self.read_voxels()

        if self.x is None:
            self.x = self.read_fluences()

        if self.D is None:
            self.D = self.read_doses()

        self.d = self.D.dot(self.x)

        dmax = np.max(self.d)
        HIST_PTS = 50

        if self.override_output_folder is not None and not os.path.isdir(self.override_output_folder):
            os.makedirs(self.override_output_folder)

        if (self.override_output_folder is not None):
            f = open('%s/histograms.gpt' % self.override_output_folder, 'w')
        else:
            f = open('%s/histograms.gpt' % self.path, 'w')

        f.write('set grid\nset style data lp\nset xlabel \'Dose [cGy]\'\n'
                'set ylabel \'% of volume\'\nset yrange [0:110]\nplot ')
        for r in range(self.roino):
            sid = self.roiids[r]
            name = self.roinames[r]
            print("+----- %s" % (name))
            minD, avgD, maxD = histogram(self.d, self.voxels, sid, "%s/%s.hist" % (self.path, name), 100. * self.dosegridscaling, HIST_PTS, dmax)
            print('Voxel doses in %20s: min=%12g avg=%12g max=%12g [cGy]' % (
                name, 100. * minD * self.dosegridscaling, 100. * avgD * self.dosegridscaling, 100. * maxD * self.dosegridscaling))
            if maxD > 0:
                f.write('\'' + name + '.hist\', ')
        f.write('\npause 120\n')
        f.close()

    def fluences(self):
        if self.xcoords is None:
            self.xcoords = self.read_xcoords()

        if self.x is None:
            self.x = self.read_fluences()

        if self.override_output_folder is not None and not os.path.isdir(self.override_output_folder):
            os.makedirs(self.override_output_folder)

        t = 0
        for i in range(self.bno):

            nbeam = self.bnos[i]

            rows = self.xcoords[nbeam]["sizey"]
            cols = self.xcoords[nbeam]["sizex"]

            if self.preview_fluence or self.save_png_preview_fluence:
                fmap = np.zeros((rows, cols))

            if self.override_output_folder is not None:
                f = open('%s/Field %d_%s.fluence' % (self.override_output_folder, nbeam, self.treatment_name), 'w')
            else:
                f = open('%s/Field %d_%s.fluence' % (self.path, nbeam, self.treatment_name), 'w')

            f.write('# Pareto optimal fluence for %s field %d\n' % (self.treatment_name, nbeam))
            f.write('optimalfluence\n')
            f.write('sizex %d\n' % self.xcoords[nbeam]["sizex"])
            f.write('sizey %d\n' % self.xcoords[nbeam]["sizey"])
            f.write('spacingx  %g\n' % self.xcoords[nbeam]["spacingx"])
            f.write('spacingy  %g\n' % self.xcoords[nbeam]["spacingy"])
            f.write('originx %g\n' % self.xcoords[nbeam]["originx"])
            f.write('originy %g\n' % self.xcoords[nbeam]["originy"])
            f.write('values\n')
            for j in range(0, rows):
                for i in range(0, cols):
                    key = "%dx%d" % (j, i)
                    if key in self.xcoords[nbeam]:
                        f.write('%g\t' % self.x[t])
                    else:
                        f.write('%g\t' % 0.0)

                    if self.preview_fluence or self.save_png_preview_fluence:
                        if key in self.xcoords[nbeam]:
                            fmap[j, i] = self.x[t]

                    if key in self.xcoords[nbeam]:
                        t += 1

                f.write('\n')
            f.close()

            if self.preview_fluence:
                print("Showing plot")
                import matplotlib.pyplot as plt
                plt.imshow(fmap)
                plt.show()

            if self.save_png_preview_fluence:
                if self.override_output_folder is not None:
                    fname = '%s/Preview Field %d_%s.png' % (self.override_output_folder, nbeam, self.treatment_name)
                else:
                    fname = '%s/Preview Field %d_%s.png' % (self.path, nbeam, self.treatment_name)

                print("Saving plot to %s" % fname)
                import matplotlib.pyplot as plt
                plt.imshow(fmap)
                plt.savefig(fname)

    @staticmethod
    def read_2_int(fin):
        cols = fin.readline().split()
        return int(cols[0]), int(cols[1])

    @staticmethod
    def read_3_int(fin):
        cols = fin.readline().split()
        return int(cols[0]), int(cols[1]), int(cols[2])

    @staticmethod
    def read_1_int(fin):
        cols = fin.readline().split()
        return int(cols[0])

    @staticmethod
    def read_1_float(fin):
        cols = fin.readline().split()
        return float(cols[0])


if __name__ == '__main__':
    if len(argv) < 2:
        print('Usage: %s <mainfile> [histogram|fluences] [-of override_fluences_filename] [-od output_folder] --preview_fluence --save_png_fluence' % argv[0])
        exit()

    path = os.path.dirname(argv[1])
    main = DosesMain(argv[1])

    if "-of" in argv:
        idx = argv.index("-of")
        main.override_fluences_filename = argv[idx+1]

    if "-od" in argv:
        idx = argv.index("-od")
        main.override_output_folder = argv[idx+1]

    if "--preview_fluence" in argv:
        print("Previewing fluences")
        main.preview_fluence = True

    if "--save_png_fluence" in argv:
        print("Saving fluences maps to png files")
        main.save_png_preview_fluence = True

    what = "."
    if len(argv) > 2 and argv[2] == "histogram":
        main.histogram()
        what = "generating histogram."

    if len(argv) > 2 and argv[2] == "fluences":
        main.fluences()
        what = "generating fluence maps."


    print("Finished %s" % what)