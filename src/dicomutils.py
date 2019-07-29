#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pydicom
import numpy as np
from datetime import datetime

ERROR_LEVEL = 0
WARNING_LEVEL = 1
INFO_LEVEL = 2
DEBUG_LEVEL_LEVEL = 3
DEBUG_LEVEL = 3
TRACE_LEVEL = 4


def debug(msg):
    if (DEBUG_LEVEL >= DEBUG_LEVEL_LEVEL):
        st = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
        print("([%d] %s): %s" % (DEBUG_LEVEL_LEVEL, st, msg))


def warning(msg):
    if (DEBUG_LEVEL >= WARNING_LEVEL):
        st = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
        print('([%d] %s): %s' % (WARNING_LEVEL, st, msg))


def info(msg):
    if (DEBUG_LEVEL >= INFO_LEVEL):
        st = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
        print("([%d] %s): %s" % (INFO_LEVEL, st, msg))

def trace(msg):
    if (DEBUG_LEVEL >= TRACE_LEVEL):
        st = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
        print("([%d] %s): %s" % (INFO_LEVEL, st, msg))


def error(msg):
    st = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
    print("([%d] %s): %s" % (ERROR_LEVEL, st, msg))


def listDirectory(directory):
    files = []

    for root, _, filenames in os.walk(directory):
        for each in filenames:
            filename, extension = os.path.splitext(each)
            filepath = os.path.join(root, each)
            files.append(filepath)

    return files


def findContours(dataset, number):
    contours = []
    if hasattr(dataset, 'ROIContourSequence'):
        for roi in dataset.ROIContourSequence:
            if roi.ReferencedROINumber == number:
                for contour in roi.ContourSequence:
                    if contour.ContourGeometricType != 'CLOSED_PLANAR':
                        continue
                    data = np.array(contour.ContourData)
                    n = data.shape[0]
                    coords = np.reshape(data, (n//3, 3) )
                    #coords[:, 1] *= -1  # for some reason Y axis is reversed
                    #coords = coords[:, np.array([1,0,2])]
                    coords = coords.astype(np.float)
                    contours.append(coords)
    return contours


def findPoints(dataset):
    points = {}
    if hasattr(dataset, 'ROIContours'):
        for roi in dataset.ROIContours:
            if roi.ReferencedROINumber == number:
                for contour in roi.Contours:
                    if contour.ContourGeometricType != 'POINT':
                        continue
                    name = 'unnamed' + number
                    if hasattr(dataset, 'StructureSetROIs'):
                        name = dataset.StructureSetROIs[number].ROIName
                    data = contour.ContourData
                    points[name] = data
    return points


def asASCII(text):
    try:
        result = text.decode('utf-8').encode('ascii', errors='replace')
    except Exception as e:
        result = text.decode('latin2').encode('ascii', errors='replace')
    return result


def applyGrahamScan(points):
    """ from http://tomswitzer.net/2010/03/graham-scan """
    def turn(p, q, r):
        return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)
    def keep_left(hull, r):
        while len(hull) > 1 and turn(hull[-2], hull[-1], r) != +1:
            hull.pop()
        if not hull or hull[-1] != r:
            hull.append(r)
        return hull
    points = sorted(points)
    l = reduce(keep_left, points, [])
    u = reduce(keep_left, reversed(points), [])
    l.extend(u[i] for i in xrange(1, len(u) - 1))
    return l

def getPointsFromCut(cut):
    from vtk import vtkIdList
    lines = {}
    idlist = vtkIdList()

    cut.GetLines().InitTraversal()
    while cut.GetLines().GetNextCell(idlist):
        assert idlist.GetNumberOfIds() == 2
        lines[idlist.GetId(0)] = idlist.GetId(1)

    ids = lines.keys()
    while len(ids) > 0:
        points = []
        start = ids[0]
        while start in ids:
            ids.remove(start)
            end = lines[start]

            coords = [0.0, 0.0, 0.0]
            cut.GetPoint(start, coords)
            points.append(coords)

            start = end
        yield points

def convertCutToContours(cut):
    contours = []
    for points in getPointsFromCut(cut):
        contour = np.array(points)
        contour[:, 1] *= -1  # for some reason Y axis is reversed
        contour = contour[:, np.array([0, 1, 2])]
        contours.append(contour)
    return contours

def findRS(filename_or_directory):
    if os.path.isdir(filename_or_directory):
        directory = filename_or_directory
        files = listDirectory(directory)
        filename = None
        for each in files:
            filepath, extension = os.path.splitext(each)
            if os.path.basename(filepath).startswith("rtss") and extension == '.dcm':
                filename = filepath + extension
                break
            if os.path.basename(filepath).startswith("RS") and extension == '.dcm':
                filename = filepath + extension
                break
    else:
        directory = "."
        filename = filename_or_directory
    return directory, filename

def findRSandRP(filename_or_directory):
    if os.path.isdir(filename_or_directory):
        directory = filename_or_directory
        files = listDirectory(directory)
        filename = None
        for each in files:
            filepath, extension = os.path.splitext(each)
            if os.path.basename(filepath).startswith("rtss") and extension == '.dcm':
                filename = filepath + extension
                break
            if os.path.basename(filepath).startswith("RS") and extension == '.dcm':
                filename = filepath + extension
                break
        rs = filename
        filename = None
        for each in files:
            filepath, extension = os.path.splitext(each)
            if os.path.basename(filepath).startswith("rtplan") and extension == '.dcm':
                filename = filepath + extension
                break
            if os.path.basename(filepath).startswith("RP") and extension == '.dcm':
                filename = filepath + extension
                break
        rp = filename
        directory = os.path.dirname(rp)
    else:
        directory = rs = rp = None
    return directory, rs, rp

class DICOMSorter(object):
    def __init__(self):
        pass

    def sort(self, files):
        result = {}
        for file in files:
            dataset = pydicom.read_file(file)
            z = dataset.SliceLocation if hasattr(dataset, 'SliceLocation') else dataset.ImagePositionPatient[2]
            z = float(z)
            result[file] = z
        return sorted(result, key=lambda key: result[key])
