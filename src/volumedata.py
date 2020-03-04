#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import path
from vtk import vtkXMLImageDataReader, vtkXMLImageDataWriter
from vtk import vtkImageData, vtkShortArray, vtkIntArray, vtkFloatArray

from common import log


class VolumeData(object):
	def __init__(self, data=None, thickness=None):
		self.data = data
		if data != None:
			sx, sy, sz = data.GetSpacing()
			# print "spacing (%g,%g,%g), thickness=%g" % ( sx, sy, sz, thickness )
			if thickness != None and thickness != sz:
				log.warning("Thickness corrected: %g -> %g" % (sz, thickness))
				data.SetSpacing([sx, sy, thickness])

	def createGrid(self, spacing, dimensions, origin):
		grid = vtkImageData()

		sx, sy, sz = spacing
		dx, dy, dz = dimensions
		ox, oy, oz = origin

		grid.SetSpacing(sx, sy, sz)
		grid.SetDimensions(dx, dy, dz)
		grid.SetOrigin(ox, oy, oz)

		return grid

	def createArray(self, grid):
		n = grid.GetNumberOfPoints()

		array = vtkShortArray()
		array.SetNumberOfComponents(1)
		array.SetNumberOfTuples(n)

		log.debug("Created an array with %d points" % n)
		return array

	def createIntegerArray(self, grid):
		n = grid.GetNumberOfPoints()

		array = vtkIntArray()
		array.SetNumberOfComponents(1)
		array.SetNumberOfTuples(n)

		log.debug("Created an array with %d points" % n)
		return array

	def createFloatArray(self, grid):
		n = grid.GetNumberOfPoints()

		array = vtkFloatArray()
		array.SetNumberOfComponents(1)
		array.SetNumberOfTuples(n)

		return array

	def read(self, name):
		name, _ = path.splitext(name)
		reader = vtkXMLImageDataReader()
		reader.SetFileName(name + '.vti')
		reader.Update()

		self.data = reader.GetOutput()
		return self

	def save(self, name):
		name, _ = path.splitext(name)
		writer = vtkXMLImageDataWriter()
		writer.SetCompressorTypeToNone()
		writer.SetEncodeAppendedData(0)
		if not name.endswith('.vti'):
			name = "%s.vti" % name
		writer.SetFileName(name)
		try:
			writer.SetInput(self.data)
		except:
			writer.SetInputData(self.data)
		writer.Update()
