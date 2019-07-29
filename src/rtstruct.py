#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import itertools as it
import dicom as pydicom
import dicomutils
from ct import CTVolumeDataReader
from dicomutils import asASCII
from volumedata import VolumeData
from matplotlib.path import Path
from numpy import arange

class RTStructureVolumeData(VolumeData):
    def __init__(self, data, name, number, color=None):
        VolumeData.__init__(self, data)
        
        self.name = name
        self.number = number
        self.color = color


class RTStructureVolumeDataReader(object):
    def __init__(self, filename):
        dataset = pydicom.read_file(filename)
        spacing, dimensions, origin = self.inspect(dataset)

        print("Spacing %s, Dimensions %s, Origin %s" % (spacing, dimensions, origin))
        print("Study %s" % (dataset.StudyID))

        self.dataset = dataset
        self.spacing = spacing
        self.dimensions = dimensions
        self.origin = origin

    def fillArray(self, array, contours):
        ox, oy, oz = self.origin
        sx, sy, sz = self.spacing
        dx, dy, dz = self.dimensions
        xs, ys, zs = arange(ox, ox + dx*sx, sx), arange(oy, oy + dy*sy, sy), arange(oz, oz + dz*sz, sz)
        my = oy + 0.5*dy*sy

        points = [(point[1],point[0]) for point in it.product(ys, xs)] # x ma się zmieniac najpierw, potem y
        #print points
        nx, ny, nz = len(xs), len(ys), len(zs)
        print("%d x %d x %d = %d" % (nx, ny, nz, nx * ny * nz))

        for coords in contours:
            xy, z = coords[:, 0:2], coords[0, 2]
            #xy[:,1] = 2*my - xy[:,1]
            path = Path(xy)
            zi = zs.searchsorted(z)
            if zi < nz:
                i = zi * nx * ny + 1
                for each in path.contains_points(points):
                    value = array.GetValue(i)
                    if each:
                        value = value ^ 255  # invert the value
                    array.SetValue(i, value)
                    i += 1
        print("done.")
        return array

    def read(self, name):
        volume = VolumeData()
        number, color = self.find(name)
        contours = self.findContours(number)
        grid  = volume.createGrid(self.spacing, self.dimensions, self.origin)
        # sprawdzamy, jak sa ponumerowane wezly - wychodzi, ze najpierw x, potem y, na koncu z
        #for i in range(0,self.dimensions[0]*self.dimensions[1]*5,self.dimensions[0]*self.dimensions[1]):
        #    print i
        #    print grid.GetPoint(i)
        array = volume.createArray(grid)
        array = self.fillArray(array, contours)
        grid.GetPointData().SetScalars(array)
        return RTStructureVolumeData(grid, name, number, color)

    def findFirstClosedPlanarROI(self):
        for roi in self.dataset.StructureSetROIs:
            return roi
        return None

    def findFirstClosedPlanarContour(self):
        for contour in self.dataset.ROIContours:
            return contour
        return None

    def findLastObservation(self):
        last = None
        for observation in self.dataset.RTROIObservations:
            if last is None or observation.ObservationNumber > last.ObservationNumber:
                last = observation
        return last

    def add(self, contours, roiname, roinumber, color):

        def newROI():
            old = self.findFirstClosedPlanarROI()
            new = copy.deepcopy(old)
            new.ROINumber = roinumber
            new.ROIGenerationAlgorithm = 'PW'
            new.ROIName = asASCII(roiname)
            return new

        self.dataset.StructureSetROIs.append(newROI())

        def newContours():
            old = self.findFirstClosedPlanarContour()
            new = copy.deepcopy(old)
            new.RefdROINumber = roinumber
            new.ReferencedROINumber = roinumber
            new.ROIDisplayColor = color
            template  = copy.deepcopy(new.Contours[0])
            del new.Contours[:]
            for c in contours:
                c = c.flatten()
                numberOfPoints = int(len(c) / 3)
                if numberOfPoints > 1:
                    nc = copy.deepcopy(template)
                    nc.NumberofContourPoints = numberOfPoints
                    nc.ContourData = ['%g' % x for x in c]
                    nc.ContourGeometricType = 'CLOSED_PLANAR'
                    new.Contours.append(nc)
            return new

        self.dataset.ROIContours.append(newContours())

        def newObservation():
            old = self.findLastObservation()
            new = copy.deepcopy(old)
            new.ObservationNumber = old.ObservationNumber + 1
            new.ROIObservationLabel = asASCII(roiname)
            #new.ROIInterpreter = roiinterpreter
            #new.RTROIInterpretedType = roitype
            new.ReferencedROINumber = roinumber
            new.RefdROINumber = roinumber
            return new

        self.dataset.RTROIObservations.append(newObservation())
        # overwrite the dataset
        pydicom.write_file(self.dataset.filename, self.dataset)

    def intersect(self, surface, roiname, color=None):
        contours = self.findIntersectionContours(surface)
        roinumber = self.findNextROINumber()
        color = color or self.randomColor()
        self.add(contours, roiname, roinumber, color)
        print("Added the intersection as a new ROI: %s" % roiname)
        return self

    def findNextROINumber(self):
        nextNumber = 0
        for (name, number) in self.list():
            if number > nextNumber:
                nextNumber = number
        return nextNumber + 1

    def randomColor(self):
        return [43, 204, 234]

    def findIntersectionContours(self, surface):
        ox, oy, oz = self.origin
        sx, sy, sz = self.spacing
        dx, dy, dz = self.dimensions

        contours = []
        for z in arange(oz, oz + dz*sz, sz):
            for contour in surface.slice(z):
                # print "Cutting along Z = %d (%d points)" % (z, len(contour))
                contours.append(contour)

        return contours

    def findContours(self, number):
        contours = dicomutils.findContours(self.dataset, number)
        return contours

    def findPoints(self, name):
        points = dicomutils.findPoints(self.dataset)
        if name in points:
            return points[name]
        return points

    def list(self):
        if hasattr(self.dataset, 'StructureSetROIs'):
            for roi in self.dataset.StructureSetROIs:
                yield asASCII(roi.ROIName), roi.ROINumber

    def find(self, roiname):
        roinumber = None
        for (name, number) in self.list():
            if asASCII(name) == asASCII(roiname):
                roinumber = number
                break

        if roinumber is not None:
            if hasattr(self.dataset, 'ROIContours'):
                for roi in self.dataset.ROIContours:
                    if roi.ReferencedROINumber == roinumber:
                        r, g, b = roi.ROIDisplayColor
                        r, g, b = float(r), float(g), float(b)
                        return  roinumber, (r/255, g/255, b/255)
        else:
            raise Exception("No ROI named %s" % roiname)

    def getReferencedSeries(self, dataset):
        if hasattr(dataset, 'ReferencedFrameofReferences'):
            frame = dataset.ReferencedFrameofReferences[0]
            if hasattr(frame, 'RTReferencedStudies'):
                study = frame.RTReferencedStudies[0]
                if hasattr(study, 'RTReferencedSeries'):
                    series = study.RTReferencedSeries[0]
                    return series
        return None

    def getReferencedImage(self, dataset, contour):
        prefix = 'CT'
        directory, _ = os.path.split(dataset.filename)
        filename = os.path.join(directory, prefix + '.' + contour.ReferencedSOPInstanceUID + '.dcm')
        if not os.path.exists(filename):
            files = dicomutils.listDirectory(directory)
            filename = None
            for one in files:
                _, name = os.path.split(one)
                if name.upper().startswith("CT"):
                    filename = one
            if filename is None:
                raise Exception("No CT images in %s" % directory)
        image = pydicom.read_file(filename)
        return image

    def inspect(self, dataset):
        series = self.getReferencedSeries(dataset)
        contour = series.ContourImages[0]
        dz = len(series.ContourImages)

        image = self.getReferencedImage(dataset, contour)
        sx, sy = image.PixelSpacing
        sz = image.SliceThickness
        sx, sy, sz = float(sx), float(sy), float(sz)
        dx, dy = image.Columns, image.Rows
        dx, dy = int(dx), int(dy)
        ox, oy, oz = image.ImagePositionPatient
        ox, oy, oz = float(ox), float(oy), float(oz)

        directory, _ = os.path.split(dataset.filename)
        ct = CTVolumeDataReader(directory, study=image.StudyID)
        bottomSlice = ct.bottomSlice()
        oz = float(bottomSlice.SliceLocation)

        spacing = sx, sy, sz
        dimensions = dx, dy, dz
        origin = ox, oy, oz

        return spacing, dimensions, origin
