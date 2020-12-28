#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pydicom
import numpy as np
from datetime import datetime
import logging as log

""" http://www.dicomlibrary.com/dicom/sop/ """
CT_SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
RTSSS_SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
RT_DOSE_SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2'
RT_PLAN_SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'

def listDirectory(directory):
    files = []

    for root, _, filenames in os.walk(directory):
        for each in filenames:
            filename, extension = os.path.splitext(each)
            if not filename.startswith("."):
                filepath = os.path.join(root, each)
                files.append(filepath)

    return files


def find_ct_rs_rp_dicom(directory_name):
    ct = []
    doses = []
    rs = None
    rp = None
    if os.path.isdir(directory_name):
        files = listDirectory(directory_name)

        for file in files:
            d = pydicom.read_file(file)
            print(f"{file} - {d.SOPClassUID}")
            if CT_SOPClassUID in d.SOPClassUID:
                ct.append(file)

            if RTSSS_SOPClassUID in d.SOPClassUID:
                rs = file

            if RT_PLAN_SOPClassUID in d.SOPClassUID:
                rp = file

            if RT_DOSE_SOPClassUID in d.SOPClassUID:
                doses.append(file)

    return rs, rp, ct, doses

def listDirectory(directory):
    files = []

    for root, _, filenames in os.walk(directory):
        for each in filenames:
            filename, extension = os.path.splitext(each)
            filepath = os.path.join(root, each)
            files.append(filepath)

    return files


def findContours(dataset, number):
    """
    input:
        dataset - dictionary representing dicom structure file 
        number - filter out by the ROI number

    Returns:
        A list of contours (unordered - same order as in a dicom file)
        Each contour is a np.float NumPy array.
        
    """
    contours = []
    if hasattr(dataset, 'ROIContourSequence'):
        for roi in dataset.ROIContourSequence:
            if roi.ReferencedROINumber == number:
                if hasattr(roi, 'ContourSequence'):
                    for contour in roi.ContourSequence:
                        if contour.ContourGeometricType != 'CLOSED_PLANAR':
                            continue
                        data = np.array(contour.ContourData)
                        n = data.shape[0]
                        coords = np.reshape(data, (n//3, 3) )
                        coords = coords.astype(np.float)
                        contours.append(coords)
                else:
                    log.warn(f"ROI: {number} doesn't have attribute ContourSequence.")
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
