# coding=utf-8
from sys import argv
from math import sin, cos, pi
import pydicom as dicom
import numpy as np
import vtk
from vtk import vtkPoints, vtkXMLPolyDataWriter, vtkPolyData, vtkCellArray, vtkFloatArray

SOURCE_TO_ISOCENTER_DISTANCE = 1000
RAY_LENGTH = 2.0


class Beamlets:
	def __init__(self, beam):
		self.beamlets_doses = None
		self.beam_number = beam.BeamNumber
		angle = float(beam.ControlPointSequence[0].GantryAngle) - 90.0
		angle = pi * angle / 180.0
		# [ dx, dy, 0 ] - is isocenter-to-source vector
		dx = SOURCE_TO_ISOCENTER_DISTANCE * cos(angle)
		dy = SOURCE_TO_ISOCENTER_DISTANCE * sin(angle)
		self.isocenter = np.array(list(map(float, beam.ControlPointSequence[0].IsocenterPosition)))
		self.source = np.array([self.isocenter[0] + dx, self.isocenter[1] + dy, self.isocenter[2]])
		# print("Isocenter: %f, %f, %f" % (self.isocenter[0], self.isocenter[1], self.isocenter[2]))

		compensator = beam.CompensatorSequence[0]
		self.rows = compensator.CompensatorRows
		self.columns = compensator.CompensatorColumns
		self.hx = float(compensator.CompensatorPixelSpacing[0])
		self.hy = float(compensator.CompensatorPixelSpacing[1])
		xOffset = abs(float(compensator.CompensatorPosition[0])) + self.hx / 2
		yOffset = abs(float(compensator.CompensatorPosition[1])) + self.hy / 2
		self.originx = float(compensator.CompensatorPosition[0])
		self.originy = float(compensator.CompensatorPosition[1])
		xSize = self.columns * self.hx
		ySize = self.rows * self.hy

		v = np.array([dy, -dx, 0])
		vn = v / np.linalg.norm(v)
		# vn - normal vector tangential to the fluence plane, only x and y components
		v = vn * xOffset
		# fluence plane offset (top-rightmost corner)
		origin = np.array([self.isocenter[0] + v[0],
						   self.isocenter[1] + v[1],
						   self.isocenter[2] + yOffset])


		# tau - vector tangential to the fluence plane, half of the grid length )
		self.tau = [v[0] * self.hx / 2, v[1] * self.hx / 2, self.hy / 2]
		self.points = []
		self.beamlet_edges_v1 = []
		self.beamlet_edges_v2 = []
		self.beamlet_edges_v3 = []
		self.beamlet_edges_long_v1 = []
		self.beamlet_edges_long_v2 = []
		self.beamlet_edges_long_v3 = []
		self.beamlet_edges_long_v4 = []

		for j in range(self.rows):
			z = origin[2] - (j * self.hy + self.hy / 2)
			ze = origin[2] - j * self.hy
			for i in range(self.columns):
				x = origin[0] - (i * self.hx + self.hx / 2) * vn[0]
				y = origin[1] - (i * self.hx + self.hx / 2) * vn[1]
				xe = origin[0] - (i * self.hx) * vn[0]
				ye = origin[1] - (i * self.hx) * vn[1]

				# Najpierw wyznaczam współrzędne do liczenia w Monte Carlo (krótsze, bo musi się wczesniej zaczynać)
				v1 = np.array([self.source[0] + 0.35 * (xe - self.hx * vn[0] - self.source[0]),
					  self.source[1] + 0.35 * (ye - self.hx * vn[1] - self.source[1]),
					  self.source[2] + 0.35 * (ze - self.source[2])])
				v2 = np.array([self.source[0] + 0.35 * (xe - self.source[0]),
					  self.source[1] + 0.35 * (ye - self.source[1]),
					  self.source[2] + 0.35 * (ze - self.source[2])])
				v3 = np.array([self.source[0] + 0.35 * (xe - self.source[0]),
					  self.source[1] + 0.35 * (ye - self.source[1]),
					  self.source[2] + 0.35 * (ze - self.hy - self.source[2])])

				self.beamlet_edges_v1.append(v1)
				self.beamlet_edges_v2.append(v2)
				self.beamlet_edges_v3.append(v3)

				# Teraz wyzanczam współrzędne krawędzi tylko do wizualizacji (dlatego aż cztery)
				v1 = np.array([self.source[0] + RAY_LENGTH * (xe - self.hx * vn[0] - self.source[0]),
					  self.source[1] + RAY_LENGTH * (ye - self.hx * vn[1] - self.source[1]),
					  self.source[2] + RAY_LENGTH * (ze - self.source[2])])
				v2 = np.array([self.source[0] + RAY_LENGTH * (xe - self.source[0]),
					  self.source[1] + RAY_LENGTH * (ye - self.source[1]),
					  self.source[2] + RAY_LENGTH * (ze - self.source[2])])
				v3 = np.array([self.source[0] + RAY_LENGTH * (xe - self.source[0]),
					  self.source[1] + RAY_LENGTH * (ye - self.source[1]),
					  self.source[2] + RAY_LENGTH * (ze - self.hy - self.source[2])])
				v4 = np.array([self.source[0] + RAY_LENGTH * (xe - self.hx * vn[0] - self.source[0]),
					  self.source[1] + RAY_LENGTH * (ye - self.hx * vn[1] - self.source[1]),
					  self.source[2] + RAY_LENGTH * (ze - self.hy - self.source[2])])

				self.beamlet_edges_long_v1.append(v1)
				self.beamlet_edges_long_v2.append(v2)
				self.beamlet_edges_long_v3.append(v3)
				self.beamlet_edges_long_v4.append(v4)


				# Teraz wyznaczam współrzędne środka beamleta (może służyć do wyznaczenia PDD)
				self.points.append(np.array((self.source[0] + RAY_LENGTH * (x - self.source[0]),
											 self.source[1] + RAY_LENGTH * (y - self.source[1]),
											 self.source[2] + RAY_LENGTH * (z - self.source[2]))))
		self.size = self.rows * self.columns
		self.fluence = compensator.CompensatorTransmissionData

		self.active_number = []
		self.active_row = []
		self.active_col = []
		self.active = []
		ab = 0
		for i in range(len(self.fluence)):
			if self.fluence[i] > 0:
				self.active.append(ab)
				self.active_number.append(i)
				self.active_row.append(i / self.rows)
				self.active_col.append(i % self.columns)
				ab += 1
			else:
				self.active.append(-1)
		self.active_size = ab

	def set_flu(self, idx, flu):
		self.fluence[idx] = flu

	def write_vtp(self, filename):
		pts = vtkPoints()
		pts.InsertNextPoint(self.source)
		colors = vtkFloatArray()
		for i in range(self.size):
			pts.InsertNextPoint(self.points[i])
			colors.InsertValue(i, self.fluence[i])

		polydata = vtkPolyData()
		polydata.SetPoints(pts)
		lines = vtkCellArray()
		for i in range(1, pts.GetNumberOfPoints()):
			l = vtk.vtkLine()
			l.GetPointIds().SetId(0, 0)
			l.GetPointIds().SetId(1, i)
			lines.InsertNextCell(l)
		polydata.SetLines(lines)
		polydata.GetCellData().SetScalars(colors)

		writer = vtkXMLPolyDataWriter()
		writer.SetFileName(filename)
		try:
			writer.SetInput(polydata)
		except:
			writer.SetInputData(polydata)
		writer.Update()

	def write_active_vtp(self, filename, scale=1):
		pts = vtkPoints()
		pts.InsertNextPoint(self.source * scale)
		colors = vtkFloatArray()
		n = 0
		for i in range(self.size):
			if self.active[i] >= 0:
				pts.InsertNextPoint(self.points[i] * scale)
				colors.InsertValue(n, self.fluence[i])
				n += 1

		polydata = vtkPolyData()
		polydata.SetPoints(pts)
		lines = vtkCellArray()
		for i in range(1, pts.GetNumberOfPoints()):
			l = vtk.vtkLine()
			l.GetPointIds().SetId(0, 0)
			l.GetPointIds().SetId(1, i)
			lines.InsertNextCell(l)
		polydata.SetLines(lines)
		polydata.GetCellData().SetScalars(colors)

		writer = vtkXMLPolyDataWriter()
		writer.SetFileName(filename)
		try:
			writer.SetInput(polydata)
		except:
			writer.SetInputData(polydata)
		writer.Update()

	def write_active_edges_vtp(self, filename, scale=1):
		pts = vtkPoints()
		pts.InsertNextPoint(self.source * scale)
		colors = vtkFloatArray()
		n = 0
		for i in range(self.size):
			if self.active[i] >= 0:
				pts.InsertNextPoint(self.beamlet_edges_long_v1[i] * scale)
				colors.InsertValue(n, self.fluence[i])
				n += 1
				pts.InsertNextPoint(self.beamlet_edges_long_v2[i] * scale)
				colors.InsertValue(n, self.fluence[i])
				n += 1
				pts.InsertNextPoint(self.beamlet_edges_long_v3[i] * scale)
				colors.InsertValue(n, self.fluence[i])
				n += 1
				pts.InsertNextPoint(self.beamlet_edges_long_v4[i] * scale)
				colors.InsertValue(n, self.fluence[i])
				n += 1

		polydata = vtkPolyData()
		polydata.SetPoints(pts)
		lines = vtkCellArray()
		for i in range(1, pts.GetNumberOfPoints()):
			l = vtk.vtkLine()
			l.GetPointIds().SetId(0, 0)
			l.GetPointIds().SetId(1, i)
			lines.InsertNextCell(l)
		polydata.SetLines(lines)
		polydata.GetCellData().SetScalars(colors)

		writer = vtkXMLPolyDataWriter()
		writer.SetFileName(filename)
		try:
			writer.SetInput(polydata)
		except:
			writer.SetInputData(polydata)
		writer.Update()


if __name__ == '__main__':
	plan = dicom.read_file(argv[1])

	if len(argv) > 2 and argv[2] == 'txt':
		txt = True
	else:
		txt = False

	b = 0
	for beam in plan.Beams:
		beamlets = Beamlets(beam)
		beamlets.write_vtp('bundle_%d.vtp' % b)
		beamlets.write_active_vtp('bundle_%d_active.vtp' % b)
		if txt:
			for i in range(beamlets.size):
				if beamlets.active[i] >= 0:
					print("\"%g,%g,%g\" \"%g,%g,%g\"\n" % (tuple(beamlets.source) + beamlets.points[i]))
		b += 1
