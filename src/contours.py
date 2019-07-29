#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import dicomutils
from surfacedata import SurfaceData
from matplotlib.pyplot import figure, show
from mpl_toolkits.mplot3d import Axes3D
from rtstruct import RTStructureVolumeDataReader


class NoRSFileException(Exception):
    def __init__(self, directory):
        Exception.__init__(self, "No RS.* or rtss.* file in %s" % directory)

def asHex(color):
	r, g, b = int(255 * color[0]), int(255 * color[1]), int(255 * color[2])
	return '#%02X%02X%02X' % (r, g, b)

if __name__ == '__main__':
	if len(sys.argv) > 1:
		filename_or_directory = sys.argv[1]
		directory, filename = dicomutils.findRS(filename_or_directory)
		if filename is None:
			raise NoRSFileException(directory)
		rt = RTStructureVolumeDataReader(filename)

		if len(sys.argv) > 2:
			plots = []
			labels = []
			fig = figure()
			ax = fig.gca(projection='3d')
			ax.set_aspect('equal')
			for name in sys.argv[2:]:
				p = None
				if name.endswith('.vtp'):
					surface, color = SurfaceData().read(name), (1.0, 0.0, 0.5)
					surface.rotate(ax=90)
					contours = rt.findIntersectionContours(surface)
				else:
					number, color = rt.find(name)
					contours = rt.findContours(number)
				for c in contours:
					if len(c) > 1:
						p, = ax.plot(c[:, 0], c[:, 1], c[:, 2], 'o-', color=asHex(color))
				if p is not None:
					plots.append(p)
					labels.append(name)
			ax.legend(plots, labels)
			show()
		else:
			for roi in RTStructureVolumeDataReader(filename).list():
				print(roi)
