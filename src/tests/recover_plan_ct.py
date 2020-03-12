#!python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import os.path as path
import struct
import pandas

from vtk import vtkMarchingCubes, vtkPolyDataNormals, vtkXMLImageDataReader, vtkXMLImageDataWriter
from vtk import vtkImageData, vtkShortArray, vtkIntArray, vtkFloatArray

def printp(s):
    print(s, end='', flush=True)

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


def read_x(path, casename, nbeams):
    """ Wczytuje z plików tekstowych dane o znacznikach roi oraz współczynniki CT
    """    
    res = []
    for beam in range(nbeams):
        fname = f"{path}/x_{casename}_{beam+1}.txt"
        printp(f'Reading fluences from {fname}...')

        with open(fname) as fin:
            f = []
            for line in fin:
                f.append(float(line))
            res.append(np.array(f))

        printp(f'Done.\n')

    return res

def _read_doses_cache(filename):
    """ Wczytuje plik z binarnego cache'a utworzonego za pomoca porgamu w języku C convert_doses.c
    """
    f = open(filename, 'rb')
    n = struct.unpack_from('i', f.read(4), 0)[0]

    v = np.fromfile(f, np.int32, n)
    b = np.fromfile(f, np.int32, n)
    d = np.fromfile(f, np.float32, n)

    return v,b,d

def read_d(fpath, casename, nbeams, coords, dimensions, fluences):
    """ Wczytuje z plików tekstowych dane o znacznikach roi oraz współczynniki CT
    """    
    xmax, ymax, zmax = dimensions
    totaldoses = np.zeros( (zmax, ymax, xmax), dtype=np.float32 )

    for beam in range(nbeams):
        f = fluences[beam]
        
        fname_cache = f"{fpath}/d_{casename}_{beam+1}.npbin"
        if path.exists(fname_cache):
            # jeżeli istnieje wersja binarna pliku, utworzona za pomocą narzędzia convert_doses 
            # - mega szybsze wczytywanie z użyciem Pandas

            printp(f'\rReading doses from binary file: {fname}... ')

            v,b,d = _read_doses_cache(fname_cache)       

            # konwertuję na Pandas, ponieważ ma funkcję groupby
            df = pandas.DataFrame({'v':v, 'b': b, 'd':d})
            df["d"] = df["d"] * f[b]
            gb = df.groupby('v').sum()            

            # dodaję do końcowego wyniku zgrupowane po wokselach i przeskalowane fluencjami dawki
            x, y, z = coords[gb.index, 0], coords[gb.index, 1], coords[gb.index, 2]
            totaldoses[z,y,x] += gb['d']

            print(f'Done.')

        else:
            # naiwne wczytywanie pliku tekstowego jeżeli nie istnieje plik binarny 
            # (uwaga działa grubo ponad 100 razy dłużej)
            fname = f"{fpath}/d_{casename}_{beam+1}.txt"
            i = 0
            with open(fname) as fin:
                icount = int(fin.readline())
                for line in fin:
                    (v, bt, d) = (int(v) for v in line.split())                
                    x, y, z = coords[v]
                    bt = int(bt)
                    totaldoses[z][y][x] += d * f[bt]

                    if i % 100000:
                        printp(f'\rReading doses from {fname}... {round((i / icount) * 100)}% done. ')
                    i += 1;
            print(f'Done.')

    return totaldoses


def read_v(fname, vno):
    """ Wczytuje z pliku tekstowego dane o znacznikach roi oraz współczynniki CT
    """    
    rois = np.zeros( (vno,), dtype=np.int32 )
    coords = np.zeros( (vno,3), dtype=np.int32 )
    ct = np.zeros( (vno,), dtype=np.float32 )
    i = 0
    with open(fname) as fin:
        for line in fin:
            (r, z, y, x, c) = line.split()
            coords[i,:] = (int(x), int(y), int(z))
            rois[i] = float(r)
            ct[i] = float(c)
            i += 1
    return rois, coords, ct


def read_rois(fname, dimensions):
    """ Wczytuje z pliku tekstowego dane o znacznikach roi oraz współczynniki CT
    """    
    xmax, ymax, zmax = dimensions
    rois = np.zeros( (zmax, ymax, xmax), dtype=np.float32 )
    with open(fname) as fin:
        for line in fin:
            (r, z, y, x, c) = line.split()
            z = int(z)
            y = int(y)
            x = int(x)
            r = float(r)
            c = float(c)
            rois[z][y][x] = r
    return rois


def read_cts(fname, dimensions):
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
            ct[z][y][x] = c
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
Skrypt wczytuje dane o planie radioterapii i zapisuje wyniki do kilku plików VTI, które
mogę być wyświetlone np. w programie ParaView. Wynik zostanie zapisany
do plików z rozszerzeniem vti. Nazwa plików będzie utworzona używając parametru OUT_NAME
podanego podczas uruchamiania skryptu.

Użycie: 
    {__file__} M_FILE OUT_NAME

Opcje:
    M_FILE - plik z metdanaymi o planie radioterapii,
    V_FILE - plik z danymi przynależności vokseli do roiów,
    OUT_NAME - nazwa na której będą bazować wynikowe pliki vti

Przykład:
      CP=/doses-nfs/sim/pacjent_5/output/PARETO_5; python3 {__file__} $CP/m_PARETO_5.txt ala
""")


if __name__=="__main__":
    if len(sys.argv) < 3:
        print_usage()
        exit(1)

    printp(f"Reading main file meta data information from {sys.argv[1]}...")
    m = read_m(sys.argv[1])
    printp("Done.\n")

    m_path = path.dirname(sys.argv[1])
    v_fname = f"{m_path}/v_{m['casename']}.txt"

    printp(f"Reading ROIs from voxel data file from {sys.argv[2]}...")
    r = read_rois(v_fname, m['pl_dim'])
    printp("Done.\n")

    roi_vti_fname = f'{sys.argv[2]}_rois.vti'
    printp(f"Saving roi data to vtk VTI format: {roi_vti_fname}...")
    saveToVTI(roi_vti_fname, r, m['pl_spacing'], m['pl_dim'], m['pl_origin'])
    printp("Done.\n")

    printp(f"Reading ROIs from voxel data file from {sys.argv[2]}...")
    ct = read_cts(v_fname, m['pl_dim'])
    printp("Done.\n")

    ct_vti_fname = f'{sys.argv[2]}_ct.vti'
    printp(f"Saving CT data to vtk VTI format: {roi_vti_fname}...")
    saveToVTI(ct_vti_fname, ct, m['pl_spacing'], m['pl_dim'], m['pl_origin'])
    printp("Done.\n")

    printp(f"Reading voxel data file meta data information from {v_fname}...")
    rois, coords, ct = read_v(v_fname, m['nvo'])
    printp(f"Done.\n")

    printp(f"Reading fluences data...\n")
    fluences = read_x(m_path, m['casename'], m['nbeams'])
    printp(f"Done.\n")

    printp(f"Reading dosess and applying fluences to the deposited doses...\n")
    totaldoses = read_d(m_path, m['casename'], m['nbeams'], coords, m['pl_dim'], fluences)
    printp(f"Done.\n")

    doses_vti_fname = f'{sys.argv[2]}_totaldoses.vti'
    print(f"Saving totaldoses data to vtk VTI format: {doses_vti_fname}...", end='')
    saveToVTI(doses_vti_fname, totaldoses, m['pl_spacing'], m['pl_dim'], m['pl_origin'])
    print("Done.")

    print("All done.")
    print(f"Rois information VTI data saved to {roi_vti_fname} file. ")
    print(f"CT information VTI data saved to {ct_vti_fname} file.")
    print(f"Todal doses (subjected to fluences) information VTI data saved to {doses_vti_fname} file.")
    print("You can use ParaView (https://www.paraview.org) to view the genearted files.")
