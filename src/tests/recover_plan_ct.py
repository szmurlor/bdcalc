#!python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

from vtk import vtkMarchingCubes, vtkPolyDataNormals, vtkXMLImageDataReader, vtkXMLImageDataWriter
from vtk import vtkImageData, vtkShortArray, vtkIntArray, vtkFloatArray

def saveToVTI(filename, data, s, d, o):
    """ Zapisywanie danych do formatu vti
    data - to numpy array
    s - spacing (sx,sy,sz)
    d - dimensions (dx,dy,dz)
    o - origin (ox,oy,oz)
    """
    
    grid = vtkImageData()
    grid.SetSpacing(s[0], s[1], s[2])
    grid.SetDimensions(d[0], d[1], d[2])
    grid.SetOrigin(o[0], o[1], o[2])

    n = grid.GetNumberOfPoints()
    array = vtkFloatArray()
    array.SetNumberOfComponents(1)
    array.SetNumberOfTuples(n)
    array.SetVoidArray(data, np.prod(d), 1)
    grid.GetPointData().SetScalars(array)

    writer = vtkXMLImageDataWriter()
    writer.SetCompressorTypeToNone()
    writer.SetEncodeAppendedData(0)
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Update()


def read_v(fname, dimensions):
    """ Wczytuje z pliku tekstowego dane o znacznikach roi oraz współczynniki CT
    """    
    xmax, ymax, zmax = dimensions
    ct = np.zeros( (zmax, ymax, xmax), dtype=np.float32 )
    with open(fname) as fin:
        for line in fin:
            (r, z, y, x, c) = line.split()
            z = int(z)
            y = int(y)
            x = int(x)
            r = float(r)
            c = float(c)
            ct[z][y][x] = r
    return ct

def read_m(fname):
    m = {}
    with open(fname) as fin:
        m['casename'] = fin.readline().strip()
        m['nbeams'] = int(fin.readline().split()[0])
        m['beams'] = []
        for i in range(m['nbeams']):
            m['beams'].append( [int(v) for v in fin.readline().split()] )
        m['nvo'] = int(fin.readline().split()[0])
        m['dosegridscaling'] = float(fin.readline().split()[0])
        m['nroi'] = int(fin.readline().split()[0])
        m['rois'] = []
        for i in range(m['nroi']):
            r,n = fin.readline().split(" ", 1)
            m['rois'].append( [int(r), n] )
        ctg = fin.readline().split()
        plang = fin.readline().split()
        m['ct_dim'] = [int(ctg[0]), int(ctg[1]), int(ctg[2])]
        m['ct_origin'] = [float(ctg[3]), float(ctg[4]), float(ctg[5])]
        m['ct_spacing'] = [float(ctg[6]), float(ctg[7]), float(ctg[8])]

        m['pl_dim'] = [int(plang[0]), int(plang[1]), int(plang[2])]
        m['pl_origin'] = [float(plang[3]), float(plang[4]), float(plang[5])]
        m['pl_spacing'] = [float(plang[6]), float(plang[7]), float(plang[8])]

    return m

def print_usage():
    print(f"""
Użycie: 
    {__file__} M_FILE V_FILE VTI_FILE

Opcje:
    M_FILE - plik z metdanaymi o planie radioterapii,
    V_FILE - plik z danymi przynależności vokseli do roiów,
    VTI_FILE - plik gdzie ma zostać zapisany wynik

Przykład:
      CP=/doses-nfs/sim/pacjent_5/output/PARETO_5; python3 {__file__} $CP/m_PARETO_5.txt $CP/v_PARETO_5.txt ala.vti
""")

if __name__=="__main__":
    if len(sys.argv) < 4:
        print_usage()
        exit(1)

    print(f"Reading main file meta data information from {sys.argv[1]}...", end='')
    m = read_m(sys.argv[1])
    print("Done.")

    print(f"Reading voxel data file meta data information from {sys.argv[2]}...", end='')
    ct = read_v(sys.argv[2], m['pl_dim'])
    print("Done.")

    # print(m)

    print(f"Saving roi data to vtk VTI format: {sys.argv[3]}...", end='')
    saveToVTI(sys.argv[3], ct, m['pl_spacing'], m['pl_dim'], m['pl_origin'])
    print("Done.")

    print(f"VTI data saved to {sys.argv[3]} file. You can use ParaView (https://www.paraview.org) to view it.")
