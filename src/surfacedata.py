#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dicomutils
from numpy import array
from os import unlink, path
from subprocess import call
from vtk import vtkXMLPolyDataWriter, vtkXMLPolyDataReader
from vtk import vtkTransform, vtkTransformPolyDataFilter, vtkSmoothPolyDataFilter, vtkDecimatePro, vtkSurfaceReconstructionFilter
from vtk import vtkPlane, vtkCutter

class SurfaceData(object):
	def __init__(self, data=None):
		self.data = data

	def read(self, name):
		name, _ = path.splitext(name)
		reader = vtkXMLPolyDataReader()
		reader.SetFileName(name + '.vtp')
		reader.Update()

		self.data = reader.GetOutput()
		print("Surface from %s read" % reader.GetFileName())
		return self

	def save(self, name):
		name, _ = path.splitext(name)
		writer = vtkXMLPolyDataWriter()
		writer.SetFileName(name + '.vtp')
		writer.SetInput(self.data)
		writer.Update()
		print("Saved ROI Volume Data as '%s'" % writer.GetFileName())

	def transform(self, transformation):
		transform = vtkTransformPolyDataFilter()
		transform.SetInput(self.data)
		transform.SetTransform(transformation)
		transform.Update()
		self.data = transform.GetOutput()
		self.data.ComputeBounds()
		return self

	def translate(self, dx=0.0, dy=0.0, dz=0.0):
		transformation = vtkTransform()
		transformation.Translate([dx, dy, dz])
		self.transform(transformation)
		return self

	def rotate(self, ax=None, ay=None, az=None, origin=(0.0, 0.0, 0.0)):
		ox, oy, oz = origin

		transformation = vtkTransform()
		transformation.PostMultiply()
		transformation.Translate([-ox, -oy, -oz])
		if ax is not None:
			transformation.RotateX(ax)
		if ay is not None:
			transformation.RotateY(ays)
		if az is not None:
			transformation.RotateZ(az)
		transformation.Translate([+ox, +oy, +oz])

		self.transform(transformation)
		return self

	def slice(self, z=0.0):
		plane = vtkPlane()
		cutter = vtkCutter()
		cutter.SetInput(self.data)
		plane.SetOrigin(0, 0, z)
		plane.SetNormal(0, 0, 1)
		cutter.SetCutFunction(plane)
		cutter.Update()
		cut = cutter.GetOutput()
		if cut.GetNumberOfPoints() > 0:
			contours = dicomutils.convertCutToContours(cut)
			return contours
		else:
			# print "No points after cut at z = %f" % z
			return []

	def applyPoissonReconstruction(self, depth=3):
		inputdata = '/tmp/for-reconstruction'
		outputdata = '/tmp/after-reconstruction'
		program = './poisson/build/bin/PoissonReconstruction'
		if not path.exists(program):
			print("%s is missing." % program)
			print("It seems like you didn't compile PoissonReconstruction program.")
			print("Consult the README to see how to do this.")
			return None

		self.save(inputdata)

		call([program, inputdata + '.vtp', str(depth), outputdata + '.vtp'])

		reconstructed = SurfaceData().read(outputdata)
		unlink(inputdata + '.vtp')
		unlink(outputdata + '.vtp')

		return reconstructed

	def applyPointReconstruction(self):
		reconstruction = vtkSurfaceReconstructionFilter()
		reconstruction.SetInput(self.data)
		reconstruction.Update()

		reconstructed = SurfaceData(reconstruction.GetOutput())

		return reconstructed

	def applySmoothing(self, iterations=100):
		decimating = vtkDecimatePro()
		decimating.SetInput(self.data)
		decimating.SetTargetReduction(0.75)
		decimating.SetPreserveTopology(1)
		decimating.Update()

		smoothing = vtkSmoothPolyDataFilter()
		smoothing.SetInput(decimating.GetOutput())
		smoothing.SetNumberOfIterations(iterations)
		smoothing.FeatureEdgeSmoothingOff()
		smoothing.Update()

		smoothed = smoothing.GetOutput()
		return SurfaceData(smoothed)
