#!/usr/bin/env python
# -*- coding: utf-8 -*-
from array import array
import numpy as np
import os
import pydicom

import dicomutils
from dicomutils import DICOMSorter
from volumedata import VolumeData
from dicomreader import DICOMImageReader

import logging
log = logging.getLogger("ct")


# Custom exceptions
class NoCTFilesException(Exception):
	def __init__(self):
		Exception.__init__(self, "No files to read CT")


class MultipleStudiesException(Exception):
	def __init__(self, directory, studies):
		Exception.__init__(self, "More than one study found in %s: %s" % (directory, studies))
		self.studies = studies
		self.directory = directory


class StudyMismatchException(Exception):
	def __init__(self, directory, found_study, expected_study):
		Exception.__init__(self, "Mismatched study found in %s. Expected %s, found %s" % (
			directory, expected_study, found_study))
		self.directory = directory
		self.found_study = found_study
		self.expected_study = expected_study


# Regular classes
class CTVolumeData(VolumeData):
	def __init__(self, data, port=None, thickness=None):
		VolumeData.__init__(self, data, thickness)

		self.port = port
		self.spacing = data.GetSpacing()
		self.dimensions = data.GetDimensions()
		self.origin = data.GetOrigin()
		log.info("CT Spacing %s, Dimensions %s, Origin %s" % (self.spacing, self.dimensions, self.origin))

	def getIndexCoordinatesOfPoint(self, x, y, z, ctScale=1.0):
		ix = int((x / ctScale - self.origin[0]) / self.spacing[0])
		iy = int((y / ctScale - self.origin[1]) / self.spacing[1])
		iz = int((z / ctScale - self.origin[2]) / self.spacing[2])

		if ix > self.dimensions[0]:
			log.error("Warning! ix = %d > size x = %d" % (ix, self.dimensions[0]))
		if iy > self.dimensions[1]:
			log.error("Warning! iy = %d > size y = %d" % (ix, self.dimensions[1]))
		if iz > self.dimensions[2]:
			log.error("Warning! iz = %d > size z = %d" % (ix, self.dimensions[2]))

		return ix, iy, iz

	def getIndex(self, ix, iy, iz):
		return ix + iy * self.dimensions[0] + iz * self.dimensions[0] * self.dimensions[1]

	def getCTDataAsNumpyArray(self):
		vtkArray = self.data.GetPointData().GetScalars()
		# 'h' - dane wczytane przez VTK to short
		r = array('h', [0] * self.dimensions[0] * self.dimensions[1] * self.dimensions[2])
		vtkArray.ExportToVoidPointer(r)
		return np.array(r, dtype=np.float32)

	def min(self, dim):
		return self.origin[dim]

	def max(self, dim):
		return self.origin[dim] + self.dimensions[dim] * self.spacing[dim]


class CTVolumeDataReader(object):
	def __init__(self, directory, study=None, ctfiles=None):
		if ctfiles is None:
			log.info("Looking for valid files in %s" % directory)
			files = dicomutils.listDirectory(directory)
			files = self.filter(files)
		else:
			files = ctfiles

		studies = self.find(files)

		if len(studies) is 1:
			if study is None or study in studies:
				study = list(studies.keys())[0]
				log.info("Study ID: %s" % study)
			else:
				raise StudyMismatchException(directory, study, studies.keys())
		elif len(studies) is 0:
			raise NoCTFilesException()
		else:
			if study is not None:
				log.info("Study ID: %s" % study)
				files = studies[study]
			else:
				raise MultipleStudiesException(directory, studies.keys())

		files = self.sort(files)
		if len(files) is 0:
			raise NoCTFilesException()
		else:
			log.info("Reading %s CT files" % len(files))

		self.thickness = 0
		if len(files) > 1:
			f0 = pydicom.read_file(files[0])
			f1 = pydicom.read_file(files[1])
			st = float(f0.SliceThickness) if f0.SliceThickness != '' else 0
			dst = list(map(float, f1.ImagePositionPatient))[2] - list(map(float, f0.ImagePositionPatient))[2]
			log.debug("slice thickness = %g, distance = %g" % (st, dst))
			self.thickness = max([st, dst])

		self.files = files
		self.study = study

	def find(self, files):
		studies = {}
		for file in files:
			dataset = pydicom.dcmread(file)
			study = dataset.StudyID
			if study not in studies:
				studies[study] = []
			studies[study].append(file)
		return studies

	def read(self):
		reader = DICOMImageReader()
		reader.SetFileNames(self.files)
		reader.Update()

		return CTVolumeData(reader.GetOutput(), port=reader.GetOutputPort(), thickness=self.thickness)

	def sort(self, files):
		sorter = DICOMSorter()
		return sorter.sort(files)

	def filter(self, files):
		validFiles = []
		for each in files:
			filepath, extension = os.path.splitext(each)
			filename = os.path.basename(filepath)
			if filename.startswith("CT") and extension == '.dcm':
				validFiles.append(each)
			if filename.startswith("ct") and extension == '.dcm':
				validFiles.append(each)

		return validFiles

	def topSlice(self):
		ctslice = pydicom.read_file(self.files[-1])
		return ctslice

	def bottomSlice(self):
		ctslice = pydicom.read_file(self.files[0])
		return ctslice
