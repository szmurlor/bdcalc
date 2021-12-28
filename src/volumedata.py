#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import path
from vtk import vtkXMLImageDataReader, vtkXMLImageDataWriter
from vtk import vtkImageData, vtkShortArray, vtkIntArray, vtkLongArray, vtkFloatArray
from common import log
import numpy as np



class VolumeData(object):
	def __init__(self, data=None, thickness=None):
		self.data = data
		if data != None:
			sx, sy, sz = data.GetSpacing()
			if thickness != None and thickness != sz:
				log.warning("Thickness corrected: %g -> %g" % (sz, thickness))
				data.SetSpacing([sx, sy, thickness])

	def saveVolumeGridToFile(spacing, dims, origin, ngrid, filename):
		grid = VolumeData.createGrid(spacing, dims, origin)
		array = VolumeData.createFloatArray(grid)
		array.SetVoidArray(ngrid, np.prod(ngrid.shape), 1)
		grid.GetPointData().SetScalars(array)
		volume = VolumeData(grid)
		volume.save(filename)

	def saveVolumeGridToFileAsLong(spacing, dims, origin, ngrid, filename):
		grid = VolumeData.createGrid(spacing, dims, origin)
		array = VolumeData.createLongArray(grid)
		array.SetVoidArray(ngrid, np.prod(ngrid.shape), 1)
		grid.GetPointData().SetScalars(array)
		volume = VolumeData(grid)
		volume.save(filename)

	def createGrid(spacing, dimensions, origin):
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

	def createLongArray(grid):
		n = grid.GetNumberOfPoints()

		array = vtkLongArray()
		array.SetNumberOfComponents(1)
		array.SetNumberOfTuples(n)

		log.debug("Created a long array with %d points" % n)
		return array

	def createIntegerArray(grid):
		n = grid.GetNumberOfPoints()

		array = vtkIntArray()
		array.SetNumberOfComponents(1)
		array.SetNumberOfTuples(n)

		log.debug("Created an array with %d points" % n)
		return array

	def createFloatArray(grid):
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
