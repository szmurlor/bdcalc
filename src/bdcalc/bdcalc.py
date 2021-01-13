#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import pydicom as dicom
import gzip
import json
import numpy as np
import time
import struct
import dicomutils
import ray
from beamlets import Beamlets
from sys import argv
from myroi import MyRoi
from subprocess import call
from rass import RASSData

#print(os.environ["CFLAGS"])
#import pyximport; pyximport.install()
import create_pareto_vmc_c

from common import log

SOURCE_TO_SURFACE_DISTANCE = 1000

makeControlData = False
SCALE = 0.1
HIST_PTS = 50

class NoRSFileException(Exception):
    def __init__(self, directory):
        Exception.__init__(self, "No RS.* or rtss.* file in %s" % directory) #ray.init(redis_address="10.42.2.78:59999")

#ray.init(redis_address="10.0.2.15:59999")
#ray.init(redis_address=options["ray_redis_address"])        
#ray.init(redis_address="172.17.0.2:59422")
ray.init(address="ham-10:6379")
#ray.init(ignore_reinit_error=True)
import vmc

def save_beam_doses(fname, beam_doses):
    """
    :param beam_doses: must be np.array
    :return:
    """
    log.info("Saving beam doses to cache file: %s" % fname)
    bnint = struct.pack("i", np.prod(beam_doses.shape))

    fout = open(fname, "wb")
    fout.write(bnint)
    beam_doses.tofile(fout)
    fout.close()


# noinspection PyUnresolvedReferences
def read_beam_doses(fname, shape):
    beam_doses = None
    log.info("Reading beam doses from cache file: %s" % fname)
    if os.path.isfile(fname):
        fin = open(fname, "rb")
        bnint = fin.read(4)

        nint = struct.unpack("i", bnint)[0]

        if nint == np.prod(shape):
            beam_doses = np.fromfile(fin, np.float32, nint)
            beam_doses = np.reshape(beam_doses, shape)
        else:
            print("ERROR! Size of beam doses cache (%d) is not equal size of shape (%d)" % (nint, np.prod(shape)[0]))

        fin.close()

    return beam_doses


def save_beamlets_doses_map(fname, beamlets_doses):
    """
    :param treatment_name:
    :param beam_idx: integer - beam index
    :param beam_doses: must be np.array
    :return:
    """
    log.info("Saving beamlets doses to cache file: %s" % fname)
    nkeysint = struct.pack("i", len(beamlets_doses.keys()))

    fout = open(fname, "wb")
    fout.write(nkeysint)

    for btidx in sorted(beamlets_doses.keys()):
            fout.write(struct.pack("i", btidx))
            nrows = beamlets_doses[btidx].shape[0]
            fout.write(struct.pack("i", nrows))
            ncols = beamlets_doses[btidx].shape[1]
            fout.write(struct.pack("i", ncols))

            beamlets_doses[btidx].tofile(fout)

    fout.close()


# noinspection PyUnresolvedReferences
def read_beamlets_doses_map(fname):
    log.info("Reading beamlets doses from cache file: %s" % fname)

    beamlets_doses_map = None
    if os.path.isfile(fname):
        beamlets_doses_map = {}
        fin = open(fname, "rb")
        nkeys = struct.unpack("i", fin.read(4))[0]

        for i in range(nkeys):
            btidx = struct.unpack("i", fin.read(4))[0]
            nrows = struct.unpack("i", fin.read(4))[0]
            ncols = struct.unpack("i", fin.read(4))[0]

            beamlet_doses = np.fromfile(fin, np.float32, nrows*ncols)
            beamlet_doses = np.reshape(beamlet_doses, (nrows, ncols))

            beamlets_doses_map[btidx] = beamlet_doses

        fin.close()
        
    log.info(f"Done reading beamlet doses from cache file {fname}")
    return beamlets_doses_map


def histogram(doses, markerArray, sid, fname, scale, dv, npts):
    start = time.time()
    d = doses[(markerArray & sid) == sid]
    if len(d) < 1:
        return 0, 0, 0

    vol = len(d) * dv
    hist = [(0., 100.)]
    dmax = np.max(d)
    for s in range(npts):
        treshold = dmax * (s + 1) / npts
        i = np.sum(d < treshold)
        v = 1. - (i - 1) * dv / vol
        hist.append((scale * treshold, v * 100.))

    f = open(fname, 'w')
    for p in hist:
        f.write('%f %f\n' % p)
    f.close()

    end = time.time()
    log.debug("Histogram generated in %f seconds to the file %s." % (end - start, fname))

    return np.min(d), np.average(d), dmax


# noinspection PyShadowingNames
def saveJSONOnlyActive(beamlets, planGridInfo, output_fname, extra_options, dicom_folder):
    bdata = {'dicom_folder': dicom_folder,
             'doses_dos_path': extra_options["doses_dos_path"],
             'beamlets': [],
             'beam_number': beamlets.beam_number,
             "ncase": extra_options["ncase"] if "ncase" in extra_options else "50000",
             "nbatch": extra_options["nbatch"] if "nbatch" in extra_options else "10"
             }
    p = planGridInfo
    bdata['plan_grid'] = {
        'size': {'x': p['ixmax'], 'y': p['iymax'], 'z': p['izmax']},
        'orig': {'x': p['xorig'], 'y': p['yorig'], 'z': p['zorig']},
        'spacing': {'dx': p['dx'], 'dy': p['dy'], 'dz': p['dz']}
    }
    bdata['doses'] = {
        'min': float(p['minDose']), 'max': float(p['maxDose']), 'avg': float(p['avgDose']),
        'scaling': p['doseScaling']
    }

    iactive = 0
    for i in range(beamlets.size):
        if beamlets.active[i] >= 0:
            beamlet = {}
            beamlet['name'] = 'Beamlet %d' % (i)
            beamlet['idx'] = i  # beamlets.active[i]
            beamlet['origin'] = {
                'x': beamlets.source[0] * SCALE,
                'y': beamlets.source[1] * SCALE,
                'z': beamlets.source[2] * SCALE
            }
            beamlet['v1'] = {
                'x': beamlets.beamlet_edges_v1[i][0] * SCALE,
                'y': beamlets.beamlet_edges_v1[i][1] * SCALE,
                'z': beamlets.beamlet_edges_v1[i][2] * SCALE
            }
            beamlet['v2'] = {
                'x': beamlets.beamlet_edges_v2[i][0] * SCALE,
                'y': beamlets.beamlet_edges_v2[i][1] * SCALE,
                'z': beamlets.beamlet_edges_v2[i][2] * SCALE
            }
            beamlet['v3'] = {
                'x': beamlets.beamlet_edges_v3[i][0] * SCALE,
                'y': beamlets.beamlet_edges_v3[i][1] * SCALE,
                'z': beamlets.beamlet_edges_v3[i][2] * SCALE
            }
            bdata['beamlets'].append(beamlet)
            iactive += 1

    bdata.update(extra_options)

    log.info("Active beamlets: %d out of total %d saved to file %s" % (iactive, beamlets.size, output_fname))
    f = open(output_fname, "w")
    json.dump(bdata, f)
    f.close()

    return output_fname


def write_main_file(fname, all_beamlets, roi_marks, doseScaling, myROIs, ctVolumeData, planGridInfo):
    log.info('Writing mainfile...')

    f = open(fname, 'w')
    f.write(treatment_name + '\n')
    f.write('%d // liczba wiazek\n' % len(all_beamlets))
    for b in range(0, len(all_beamlets)):
        f.write('%d %d\n' % (all_beamlets[b].beam_number, all_beamlets[b].active_size))
    f.write('%d // liczba vokseli\n' % np.count_nonzero(roi_marks))
    f.write('%.7g // DoseGridScaling\n' % doseScaling)
    f.write('%d // liczba ROI\n' % len(myROIs))
    for r in range(len(myROIs)):
        f.write('%d %s\n' % (2 ** r, myROIs[r].name))

    f.write(f'{ctVolumeData.dimensions[0]} {ctVolumeData.dimensions[1]} {ctVolumeData.dimensions[2]} {ctVolumeData.origin[0]/10} {ctVolumeData.origin[1]/10} {ctVolumeData.origin[2]/10} {ctVolumeData.spacing[0]/10} {ctVolumeData.spacing[1]/10} {ctVolumeData.spacing[2]/10} // ct_volume_data nx ny nz origx origy origz spacingx spacingy spacingz [cm]\n')
    p = planGridInfo
    f.write(f'{p["ixmax"]} {p["iymax"]} {p["izmax"]} {p["xorig"]} {p["yorig"]} {p["zorig"]} {p["dx"]} {p["dy"]} {p["dz"]} // plan_volume_data nx ny nz origx origy origz spacingx spacingy spacingz [cm]\n')
    f.close()

    log.info('Finished writing mainfile...')


def write_rois(fname, totalDoses, roi_marks, kmax, jmax, imax, plan_grid_ct, v2Drow):
    log.info('Writing ROIs...')
    
    f = open(fname, 'w')
    for i in range(0, imax):
        for j in range(0, jmax):
            for k in range(0, kmax):
                if roi_marks[i][j][k] > 0:
                    ivoxel = k + j * kmax + i * (kmax * jmax)
                    f.write('%d %d %d %d %f\n' % (roi_marks[i][j][k], i, j, k, plan_grid_ct[v2Drow[ivoxel]-1]))
    f.close()

    log.info("Finished writing ROIs.")


def write_active_beamlets(fname, beamlets):
    log.info("Writing fluences for active beamlets for beam to file %s" % fname)
    f = open(fname, 'w')
    for t in range(0, beamlets.size):
        if beamlets.active[t] >= 0:
            f.write('%g\n' % beamlets.fluence[t])
    f.close()


def write_active_beamlets_coordinates(fname, beamlets):
    log.info("Writing coordinates of active beamlets for beam to file %s" % fname)
    f = open(fname, 'w')
    f.write("# sizex\n%d\n" % beamlets.columns)
    f.write("# sizey\n%d\n" % beamlets.rows)
    f.write("# spacingx\n%f\n" % beamlets.hx)
    f.write("# spacingy\n%f\n" % beamlets.hy)
    f.write("# originx\n%f\n" % beamlets.originx)
    f.write("# originy\n%f\n" % beamlets.originy)

    t = 0
    for j in range(0, beamlets.rows):
        for i in range(0, beamlets.columns):
            if beamlets.active[t] >= 0:
                f.write('%d %d\n' % (j, i))
            t += 1
    f.close()


def map_voxel_to_D_row(kmax, jmax, imax, roi_marks):
    row = 0
    v2Drow = np.zeros(kmax * jmax * imax, dtype=np.int)
    idx = 0
    for i in range(0, imax):
        for j in range(0, jmax):
            for k in range(0, kmax):
                if roi_marks[i][j][k] > 0:
                    v2Drow[idx] = row
                    row += 1
                else:
                    v2Drow[idx] = -1
                idx += 1
    return v2Drow


def save_voxel_to_D_row(v2Drow, fname):
    log.info("Saving total roimarks to cache file: %s" % fname)
    bnint = struct.pack("i", np.prod(v2Drow.shape))
    fout = open(fname, "wb")
    fout.write(bnint)
    v2Drow.tofile(fout)
    fout.close()


# noinspection PyUnresolvedReferences
def read_voxel_to_D_row(fname):
    res = None
    if os.path.isfile(fname):
        fin = open(fname, "rb")
        bnint = fin.read(4)
        nint = struct.unpack("i", bnint)[0]

        v2Drow = np.fromfile(fin, np.int, nint)
        res = v2Drow

        fin.close()

    return res


def get_beamlets_from(rtplan):
    beamlets_list = []
    for beam in rtplan.BeamSequence:
        beamlets_list.append(Beamlets(beam))
    return beamlets_list


def default_options():
    return {
        "mc_scale_to_max_factor": 1.0,   # współczynniki wagowe przy skalowaniu dawek Monte Carlo do Eclipse
        "mc_scale_to_avg_factor": 0.0,   # uzywane wg. wzoru: s = avg_eclipse / avg_mc * mc_scale_to_avg_factor +
                                         #                        max_eclipse / max_mc * mc_scale_to_max_factor
        "extra_doses_scale": 0.0,        # 0.0 - no extra scaling, else mcScale = mcScale * extra_doses_scale
        "ncase": 50000,                  # liczba próbek MC
        "nbatch": 10,                    # liczba iteracji MC
        "histograms": False,             # czy generować histogramy
        "run_gnuplot": False,             # czy generować histogramy
        "use_cached_doses": False,            # Nie licz dawek, tylko postaraj się je wczytać z binarnych plików Cache
        "augment_planning_grid": False,  # augment planning grid to geometrically  cover all strutures from RS
        "water_phantom": False,          # zastosuj poprawkę do CT - fantom wodny
        "debug_level": 3,
        "debug_beam_doses": False,
        "debug_mc_doses_as_vector": False,
        "debug_v2Drow": False,
        "out_main_file": False,
        "out_rois": False,
        "out_xcoords_file": True,             # plik z informacją o wymiarach i współrzędnych poszczególnych fluencji
        "out_recover_structure": False,
        "out_active_beamlets": True,
        "out_mc_doses": True,
        "out_mc_doses_fluence": True,
        "out_mc_doses_txt": True,   # czy zapisywać plik dla PANu
        "out_difference": True,
        "out_total_doses": True,
        "override_dicom_plan_grid": False,
        "vmc_home": None,
        "vmc_runs": "./runs",
        "vmc_ct_file": "phantoms/phantom.ct",
        "spectrum_filename": "spectra/var_CL2300_5_X6MV.spectrum",
        "delete_doses_file_after": True,
        "delete_vmc_file_after": True,
        "ppservers": None,
        "ppsecret": None,
        "ray_redis_address": "10.0.2.15:59999",
        "ray_calc_max_beamlets": 8,
        "doses_dos_path": None,  # folder do którego VMC++ zapisze dawki w formacie binarnym
        "cluster_config_file": None,  # lokalizacja pliku konfiguracji danego wezla jezeli doses_dos_path == cluster
        "scale_algorithm": "individual_voxels",  # "total_avg_max" lub "individual_voxels"
        "override_plan_grid": {
            "kmax": 0,
            "jmax": 0,
            "imax": 0,
            "xbase": 0,
            "ybase": 0,
            "zbase": 0,
            "dx": 0,
            "dy": 0,
            "dz": 0
        }
    }

if __name__ == '__main__':
    ctgriddata = None
    doHistograms = True

    if len(argv) < 2:
        print('Usage: %s <directory-with-data> [wp] [options config.json]' % (argv[0]))
        print("     where: ")
        print("         <directory-with-data> - directory with full data set DICOM files should be located\n" \
              "                                 in 'in' subfolder (RP* - radio plan, RD* - radio doses, \n" \
              "                                 RS* - structure set, CT* - CT data), \n")
        print("         options config.json - a json file with configuration (if skipped default values will be\n" \
              "                               applied),")
        exit()

    rass_data = RASSData(root_folder=argv[1])
    directory = rass_data.input()
    dicom_directory = rass_data.input("dicom")
    output_directory = rass_data.output()

    ################################################################
    # Wczytuję opcje z "in" folderu
    ################################################################
    options = default_options()
    cfname = rass_data.input("config.json")
    if os.path.isfile(cfname):
        log.info("Reading options from file: %s" % cfname)
        with open(cfname) as options_file:
            options.update(json.load(options_file))

    ################################################################
    # Przesłaniam opcje za pomocą pliku przekazanego za pomocą argumentów linii komend
    ################################################################
    for i in range(len(argv)):
        if "options" == argv[i]:
            fname = "%s" % (argv[i + 1])
            log.info("Reading options from file: %s" % fname)
            with open(fname) as options_file:
                options.update(json.load(options_file))

    dicomutils.DEBUG_LEVEL = options["debug_level"]

    ################################################################
    # Szukam plików DICOM
    ################################################################
    rtss, plan, ctlist, doseslist = dicomutils.find_ct_rs_rp_dicom(dicom_directory)
    if rtss is None or plan is None:
        raise NoRSFileException(dicom_directory)

    ################################################################
    # Wczytuję informacje o strukturach (ROIach) oraz plan
    ################################################################
    rtss = dicom.read_file(rtss)
    plan = dicom.read_file(plan)
    treatment_name = '-'.join(plan.PatientID.split('^'))
    log.info('Name: ' + treatment_name)

    ################################################################
    # Wczytuję dane CT
    ################################################################

    from ct import CTVolumeDataReader
    reader = CTVolumeDataReader(dicom_directory, ctfiles=ctlist)
    ctVolumeData = reader.read()


    if len(ctlist) > 0:
        ct = dicom.read_file(ctlist[0])
        ctgriddata = list(map(float, (ct.ImagePositionPatient[0], ct.ImagePositionPatient[1],
                                 ct.PixelSpacing[0], ct.PixelSpacing[1], ct.Columns, ct.Rows)))
    else:
        ctgriddata = None

    ################################################################
    # reading doses information for beams from DICOM
    ################################################################
    beams = [dicom.read_file(f) for f in doseslist]

    ##################################################################
    # Sumuję dawki z poszczególnych wiązek (beams) do całkowitej dawki
    ##################################################################
    beamDoses = {}
    totalDoses = None
    totalDosesFile = None
    doseScaling = None
    singleBeam = False
    for beam in beams:
        doseScaling = float(beam.DoseGridScaling)
        try:
            bn = int(beam.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber)
        except:
            print("ERROR! Semething wrong went...")
            if totalDoses is None:
                singleBeam = True
                totalDoses = beam.pixel_array.copy()
                totalDosesFile = beam.filename
            continue
        beamDoses[bn] = beam.pixel_array
        if doseScaling is not None and float(beam.DoseGridScaling) != doseScaling:
            log.warning('Strange data: DoseGridScaling is not same all beamlets!')

    if not singleBeam:
        bns = beamDoses.keys()
        totalDoses = beamDoses[bns[0]].copy()
        for i in range(1, len(bns)):
            log.info(f"Adding doses from beam {i}")
            totalDoses += beamDoses[bns[i]]

    log.info(f"Read doses for {len(beamDoses)} beams")

    totalDoses = np.array(totalDoses, dtype=np.float32)
    npTotalDoses = np.array(totalDoses, dtype=np.float32)

    minDose = np.min(totalDoses)
    averageDose = np.average(totalDoses)
    maxDose = np.max(totalDoses)

    if totalDosesFile is None:
        log.info('Total doses calculated as sum of beam doses (min dose=%f, average dose=%f, max dose=%f, doseScaling=%f)' % (
            minDose, averageDose, maxDose, doseScaling))
    else:
        log.info('Got total doses from %s (min dose=%f, average dose=%f, max dose = %f, doseScaling=%f)' % (
            totalDosesFile, minDose, averageDose, maxDose, doseScaling))

    kmax = beams[0].Columns
    jmax = beams[0].Rows
    imax = len(beams[0].GridFrameOffsetVector)
    xbase = float(beams[0].ImagePositionPatient[0]) * SCALE
    ybase = float(beams[0].ImagePositionPatient[1]) * SCALE
    zbase = float(beams[0].ImagePositionPatient[2]) * SCALE
    dx = float(beams[0].PixelSpacing[0]) * SCALE
    dy = float(beams[0].PixelSpacing[1]) * SCALE
    zoffsets = list(map(float, beams[0].GridFrameOffsetVector))
    for i in range(len(zoffsets)):
        zoffsets[i] *= SCALE
    dz = zoffsets[1] - zoffsets[0]
    dv = dx * dy * dz
    log.info('Original planning grid: %d x %d x %d in [%g:%g]x[%g:%g]x[%g:%g] dx,dy,dz=%g,%g,%g -> dv=%g' % (
        kmax, jmax, imax,
        xbase, xbase + kmax * dx, ybase, ybase + jmax * dy, zbase + zoffsets[0], zbase + zoffsets[-1],
        dx, dy, dz, dv))


    if options["override_dicom_plan_grid"]:
        pg = options["override_plan_grid"]
        kmax = pg["kmax"] if "kmax" in pg else kmax
        jmax = pg["jmax"] if "jmax" in pg else jmax
        imax = pg["imax"] if "imax" in pg else imax
        xbase = pg["xbase"] if "xbase" in pg else xbase
        ybase = pg["ybase"] if "ybase" in pg else ybase
        zbase = pg["zbase"] if "zbase" in pg else zbase
        dx = pg["dx"] if "dx" in pg else dx
        dy = pg["dy"] if "dy" in pg else dy
        dz = pg["dz"] if "dz" in pg else dz

        dv = dx * dy * dz
        log.info('NEW Overriden planning grid: %d x %d x %d in [%g:%g]x[%g:%g]x[%g:%g] dx,dy,dz=%g,%g,%g -> dv=%g' % (
            kmax, jmax, imax,
            xbase, xbase + kmax * dx, ybase, ybase + jmax * dy, zbase, zbase + dz * imax,
            dx, dy, dz, dv))

        totalDoses = np.zeros((imax, jmax, kmax), dtype=int)

    planGridInfo = {'ixmax': kmax, 'iymax': jmax, 'izmax': imax,
                    'xorig': xbase, 'yorig': ybase, 'zorig': zbase,
                    'dx': dx, 'dy': dy, 'dz': dz,
                    'minDose': minDose, 'avgDose': averageDose, 'maxDose': maxDose,
                    'doseScaling': doseScaling
                    }


    ####################################################
    # Analiza ROIów
    ####################################################
    start = time.time()
    myROIs = []
    idxROIBody = -1
    for i in range(0, len(rtss.StructureSetROISequence)):
        roiName = rtss.StructureSetROISequence[i].ROIName
        log.info(f"Finding contours for {roiName}")
        myROIs.append(MyRoi(dicomutils.findContours(rtss, rtss.StructureSetROISequence[i].ROINumber),
                            roiName, float(beams[0].PixelSpacing[0]) / 1000.0))

        if ("body" in roiName.lower() or "skin" in roiName.lower() or "outline" in roiName.lower()) and (idxROIBody == -1):
            idxROIBody = i
            log.info("Found ROI body (or skin): idx = %d" % idxROIBody)
    end = time.time()
    log.debug("Found contours in %s s" % (end - start))
    if idxROIBody == -1:
        raise Exception("The structure file does not contain any structure with 'body', 'outline' or 'skin' in the name.")


    ##########################################################################
    # Mark ROIs or read from cache (cache is a file in a working
    # directory, separate file for each ROI,
    # the filename pattern is: "%s_%s.markscache" % (treatment_name, ROIName)
    ##########################################################################
    roi_marks = np.zeros((imax, jmax, kmax), dtype=int)
    roi_marks_check = np.zeros((imax, jmax, kmax), dtype=int)
    for r in range(0, len(myROIs)):
        fcache = rass_data.processing("%s_%s.markscache" % (treatment_name, myROIs[r].name))
        if myROIs[r].read_marks(fcache, roi_marks) is False:
            log.info("Marking voxels for %s" % myROIs[r].name)
            log.info("CTGRID DATA %s" % list(ctgriddata))
            myROIs[r].mark(xbase / SCALE, ybase / SCALE, dx / SCALE, dy / SCALE, kmax, jmax, imax,
                           np.linspace(zbase, zbase + (imax - 1) * dz, imax) / SCALE, roi_marks, 2 ** r, ctgriddata=ctgriddata)
            myROIs[r].save_marks(fcache, roi_marks, 2 ** r)
        else:
            log.info("Read marking voxels for %s from cache" % myROIs[r].name)
            myROIs[r].countVoxels(roi_marks, 2 ** r)
            roi_marks_check = roi_marks

    for r in range(len(myROIs)):
        log.info("Statistics for %20s: ID=%8d, %7d voxels, vol=%8.1f discrete vol=%8.1f [cm3]" % (
            myROIs[r].name, 2 ** r, myROIs[r].count, myROIs[r].volume / 1000.,
            myROIs[r].count * dv / SCALE / SCALE / SCALE / 1000.0))


    #################################################################
    # Map rows to voxels: try to read from cache or calculate
    # The cache file name pattern: "%s.v2drowcache" % treatment_name
    #################################################################
    log.info("Preparing map voxel -> D_row (r_* file)")
    fcache = rass_data.processing("%s.v2drowcache" % (treatment_name))
    v2Drow = read_voxel_to_D_row(fcache)
    if v2Drow is None:
        v2Drow = map_voxel_to_D_row(kmax, jmax, imax, roi_marks)
        save_voxel_to_D_row(v2Drow, fcache)
    log.info(f'{len(v2Drow)} voxels classified to ROIs')

    # Main doses matrix obtained from MonteCarlo and rebuild from Doses v2Doses map
    mcDoses = np.zeros(totalDoses.shape, dtype=np.float32)
    mcDosesFluence = np.zeros(totalDoses.shape, dtype=np.float32)

    # Doses obtained directly from VMC++ MC simulator (not scaled)
    mcDosesVMC = np.zeros(np.prod(totalDoses.shape), dtype=np.float32)

    ######################################################################
    log.info("Starting marking voxels")
    ######################################################################
    voxels = np.zeros(totalDoses.shape, dtype=np.float32)
    create_pareto_vmc_c.mark_voxels(voxels, v2Drow, kmax, jmax, imax)
    log.info("Finished marking voxels")
    vmc.saveToVTI(rass_data.output("voxels.vti"), voxels, [dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])

    #####################################################################
    all_beamlets = get_beamlets_from(plan)
    #####################################################################

    ######################################################################
    log.info('Processing bundles')
    ######################################################################
    for beamlets in all_beamlets:
        beamNo = beamlets.beam_number

        if options["out_active_beamlets"]:
            beamlets.write_active_vtp(rass_data.output("wiazki_aktywne_%d.vti" % beamNo), SCALE)
            beamlets.write_active_edges_vtp(rass_data.output("wiazki_aktywne_krawedzie_%d.vti" % beamNo), SCALE)

        beamlets_doses_cachefile = rass_data.processing("%s_%d.beamlets_doses_map_cache" % (treatment_name, beamNo))
        beam_doses_cachefile = rass_data.processing("%s_%d.beam_doses_cache" % (treatment_name, beamNo))
        needs_calculate = True
        if options["use_cached_doses"]:
            beamTotalDoses = read_beam_doses(beam_doses_cachefile, mcDosesVMC.shape)
            beamlets_doses = read_beamlets_doses_map(beamlets_doses_cachefile)
            needs_calculate = beamTotalDoses is None or beamlets_doses is None

        write_active_beamlets(rass_data.output('x_%s_%d.txt' % (treatment_name, beamNo), subfolder="%s" % treatment_name), beamlets)
        write_active_beamlets_coordinates(rass_data.output('xcoords_%s_%d.txt' % (treatment_name, beamNo), subfolder="%s" % treatment_name), beamlets)


        beamletDataFilename = rass_data.processing("beam_data_%d.json" % beamNo)
        beamSpecJSON = saveJSONOnlyActive(beamlets, planGridInfo, beamletDataFilename, options, dicom_directory)

        vmcCalculator = vmc.VMC(rass_data)
        plan_grid_ct = vmcCalculator.getApproximatedCT(beamSpecJSON, v2Drow=v2Drow, voxels=voxels, options=options, ctfiles=ctlist)

        if needs_calculate:
            ###################################################################################################
            ########################### RUN ###################################################################
            ###################################################################################################
            vmcCalculator = vmc.VMC(rass_data)
            beamlets_doses = vmcCalculator.run(beamSpecJSON, v2Drow=v2Drow, voxels=voxels, options=options, ctfiles=ctlist)
            save_beamlets_doses_map(beamlets_doses_cachefile, beamlets_doses)
            ###################################################################################################

            beamTotalDoses = vmcCalculator.total_doses
            save_beam_doses(beam_doses_cachefile, beamTotalDoses)

        mcDoses += np.reshape(beamTotalDoses,[imax, jmax, kmax])
        mcDosesVMC += beamTotalDoses

        if options["debug_beam_doses"]:
            log.info("beamTotalDoses = %s" % beamTotalDoses)
            log.info("beamTotalDoses.shape = %s" % beamTotalDoses.shape)
            log.info("max beamTotalDoses = %lf, avg beamTotalDoses = %lf" % (np.max(beamTotalDoses), np.average(beamTotalDoses)))

    if options["out_rois"]:
        write_rois(rass_data.output('v_%s.txt' % treatment_name, subfolder="%s" % treatment_name), totalDoses, roi_marks, kmax, jmax, imax, plan_grid_ct, v2Drow)

    vmc.saveToVTI(rass_data.output("mc_doses_test"), np.reshape(mcDoses, np.prod([imax, jmax, kmax])), [dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])

    #################################
    # Obliczam scaling coefficient
    #################################
    mcDosesForScaling = np.zeros(totalDoses.shape, dtype=np.float32)
    for beamlets in all_beamlets:
        beamNo = beamlets.beam_number
        beamlets_doses_cachefile = rass_data.processing("%s_%d.beamlets_doses_map_cache" % (treatment_name, beamNo))
        beamlets_doses = read_beamlets_doses_map(beamlets_doses_cachefile)

        for btidx in sorted(beamlets_doses.keys()):
            create_pareto_vmc_c.postprocess_fluence_for_scaling(beamlets_doses[btidx], float(beamlets.fluence[btidx]),
                                                                mcDosesForScaling, kmax, jmax, imax)

    vmc.saveToVTI(rass_data.output("mc_doses_for_scalingg"), np.reshape(mcDosesForScaling, np.prod([imax, jmax, kmax])), [dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])    

    if options["scale_algorithm"].lower() == "total_avg_max":
        if not options["override_dicom_plan_grid"]:
            mcMax = np.max(mcDosesForScaling[totalDoses > 0])
            totMax = np.max(totalDoses[totalDoses > 0])
            mcAvg = np.average(mcDosesForScaling[totalDoses > 0])
            totAvg = np.average(totalDoses[totalDoses > 0])
        else:
            mcMax = np.max(mcDosesForScaling)
            totMax = maxDose
            mcAvg = np.average(mcDosesForScaling)
            totAvg = averageDose

        if mcMax != 0 and mcAvg != 0:
            log.info("********** Monte Carlo scaling factor to Max: %f" % options["mc_scale_to_max_factor"])
            log.info("********** Monte Carlo scaling factor to Avg: %f" % options["mc_scale_to_avg_factor"])
            monteCarloScalingCoefficient = options["mc_scale_to_max_factor"] * totMax / mcMax + options["mc_scale_to_avg_factor"] * totAvg / mcAvg
        else:
            monteCarloScalingCoefficient = 1
            raise Exception("Average or Max VMC++ Monte Carlo dose is equal 0!")

        if options["extra_doses_scale"] != 0:
            log.info("Scaling Monte Carlo scaling coeffitient by %f" % options["extra_doses_scale"])
            monteCarloScalingCoefficient *= options["extra_doses_scale"]

        log.info("*************************************************************************")
        log.info("Final Monte Carlo scaling coefficient: %f " % monteCarloScalingCoefficient)
        log.info("*************************************************************************")

        ################################################################
        # Skaluję i zapisuję dawki do plików dla poszczególnych wiązek
        ################################################################
        for beamlets in all_beamlets:
            beamNo = beamlets.beam_number

            beamlets_doses_cachefile = rass_data.processing("%s_%d.beamlets_doses_map_cache" % (treatment_name, beamNo))
            beamlets_doses = read_beamlets_doses_map(beamlets_doses_cachefile)

            beam_doses_cachefile = rass_data.processing("%s_%d.beam_doses_cache" % (treatment_name, beamNo))
            beamletTotalDoses = read_beam_doses(beam_doses_cachefile, totalDoses.shape)

            f = None
            if options["out_mc_doses_txt"]:
                log.info('Writing doses for bundle %d' % beamNo)
                f = open(rass_data.output('d_%s_%d.txt' % (treatment_name, beamNo), subfolder="%s" % treatment_name), 'w')
                f.write('%d\n' % sum([beamlets_doses[k].shape[0] for k in beamlets_doses.keys()]))

            for btidx in sorted(beamlets_doses.keys()):
                create_pareto_vmc_c.postprocess_fluence(monteCarloScalingCoefficient, beamlets_doses[btidx],
                                                        float(beamlets.fluence[btidx]), mcDoses, mcDosesFluence,
                                                        kmax, jmax, imax, options["out_mc_doses_txt"],
                                                        f, v2Drow, beamlets.active[btidx])

            if options["out_mc_doses_txt"]:
                f.close()
    else:
        mcDoses = np.zeros(totalDoses.shape, dtype=np.float32)
        mcDosesScalingFactors = np.zeros(totalDoses.shape, dtype=np.float32)
        mcDosesScalingFactors[mcDosesForScaling > 1e-8] = np.divide(totalDoses[mcDosesForScaling > 1e-8], mcDosesForScaling[mcDosesForScaling > 1e-8])
        mcDosesScalingFactorsFlatSorted = np.sort(np.reshape(mcDosesScalingFactors, np.prod(mcDosesScalingFactors.shape)))
        mcDosesScalingFactorsFlatSorted = mcDosesScalingFactorsFlatSorted[mcDosesScalingFactorsFlatSorted > 1e-8]
        minScale = mcDosesScalingFactorsFlatSorted[ mcDosesScalingFactorsFlatSorted.shape[0]//4 ] # take value from 1/4 of the sorted set
        maxScale = mcDosesScalingFactorsFlatSorted[ 3*mcDosesScalingFactorsFlatSorted.shape[0]//4 ] # take value at 3/4 of the sorted set
        log.info("Using minScale = %f, maxscale %f" % (minScale, maxScale))
        mcDosesScalingFactors[ (mcDosesScalingFactors > 0) & (mcDosesScalingFactors < minScale) ] = minScale
        mcDosesScalingFactors[ (mcDosesScalingFactors > 0) & (mcDosesScalingFactors > maxScale) ] = maxScale
        vmc.saveToVTI(rass_data.output("total_doses_before_scaling"), np.reshape(totalDoses, np.prod([imax, jmax, kmax])), [dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])
        vmc.saveToVTI(rass_data.output("mc_doses_for_scaling_before_scaling"), np.reshape(mcDosesForScaling, np.prod([imax, jmax, kmax])), [dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])
        vmc.saveToVTI(rass_data.output("mcDosesScalingFactors"), np.reshape(mcDosesScalingFactors, np.prod([kmax, jmax, imax])),[dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])

        ###############################################################################################################
        # Skaluję i zapisuję dawki do plików dla poszczególnych wiązek na podstawie skal związanych z każdym wokselem.
        ###############################################################################################################
        for beamlets in all_beamlets:

            beamNo = beamlets.beam_number
            beamlets_doses_cachefilename = rass_data.processing(f"{treatment_name}_{beamNo}.beamlets_doses_map_cache")
            beamlets_doses = read_beamlets_doses_map(beamlets_doses_cachefilename)

            if options["out_mc_doses_txt"]:
                log.info(f'Writing doses for bundle {beamNo}')
                f = open(rass_data.output('d_%s_%d.txt' % (treatment_name, beamNo), subfolder="%s" % treatment_name), 'w')
                f.write('%d\n' % sum([beamlets_doses[k].shape[0] for k in beamlets_doses.keys()]))
            else:
                f = None

            log.info("Starting postprocessing beamlets...")
            for btidx in sorted(beamlets_doses.keys()):
                create_pareto_vmc_c.postprocess_fluence_individual(mcDosesScalingFactors, beamlets_doses[btidx],
                                                        float(beamlets.fluence[btidx]), mcDoses, mcDosesFluence,
                                                        kmax, jmax, imax, options["out_mc_doses_txt"],
                                                        f, v2Drow, beamlets.active[btidx])

            log.info("Finished posprocessing beamlets from file %s" % beamlets_doses_cachefilename)

            if options["out_mc_doses_txt"]:
                f.close()

        if options["out_mc_doses"]:
            vmc.saveToVTI(rass_data.output("mc_doses_tuz_after"), np.reshape(mcDoses, np.prod([imax, jmax, kmax])), [dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])

        # end of - skalowanie lokalne na poziomie pojedynczych voxeli

    ##########################################################
    # Print total doses statistics from Monte Carlo
    ##########################################################
    log.info(f'Total doses (Monte Carlo) calculated as sum of beam doses (min dose={100. * np.min(mcDoses) * doseScaling}, average dose={100. * np.average(mcDoses) * doseScaling}, max dose={100. * np.max(mcDoses) * doseScaling})')
    log.info(f'Total doses flunced (Monte Carlo) calculated as sum of beam doses and weighted with plan fluence (min dose={100. * np.min(mcDosesFluence) * doseScaling}, average dose={100. * np.average(mcDosesFluence) * doseScaling}, max dose={100. * np.max(mcDosesFluence) * doseScaling})')

    ##########################################################
    # Print total doses statistics from ECLIPSE
    ##########################################################
    if totalDosesFile is None:
        log.info('Total doses (ECLIPSE) calculated as sum of beam doses (min dose=%f, average dose=%f, max dose=%f, doseScaling=%f)' % (
            minDose, averageDose, maxDose, doseScaling))
        log.info('Total doses scaled (ECLIPSE) calculated as sum of beam doses (min dose=%f, average dose=%f, max dose=%f)' % (
            100. * minDose * doseScaling, 100. * averageDose * doseScaling, 100. * maxDose * doseScaling))
    else:
        log.info('Got total doses (ECLIPSE) from %s (min dose=%f, average dose=%f, max dose = %f, doseScaling=%f)' % (
            totalDosesFile, minDose, averageDose, maxDose, doseScaling))
        log.info('Got total doses scaled (ECLIPSE) from %s (min dose=%f, average dose=%f, max dose = %f)' % (
            totalDosesFile, 100. * minDose * doseScaling, 100. * averageDose * doseScaling, 100. * maxDose * doseScaling))

    if options["out_main_file"]:
        write_main_file(rass_data.output('m_%s.txt' % treatment_name, subfolder="%s" % treatment_name), all_beamlets, roi_marks, doseScaling, myROIs, ctVolumeData, planGridInfo)
    if options["out_rois"]:
        write_rois(rass_data.output('v_%s.txt' % treatment_name, subfolder="%s" % treatment_name), totalDoses, roi_marks, kmax, jmax, imax, plan_grid_ct, v2Drow)
    if options["debug_beam_doses"]:
        vmc.saveToVTI(rass_data.output("mcp_doses"), mcDosesVMC, [dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])
    if options["out_mc_doses"]:
        vmc.saveToVTI(rass_data.output("mc_doses"), np.reshape(mcDoses, np.prod([imax, jmax, kmax])), [dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])
    if options["out_mc_doses_fluence"]:
        vmc.saveToVTI(rass_data.output("mc_doses_fluence"), np.reshape(mcDosesFluence, np.prod([imax, jmax, kmax])), [dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])
    npTotalDoses = np.array(totalDoses, dtype=np.float32)
    if options["out_difference"]:
        vmc.saveToVTI(rass_data.output("npTotalDoses"), np.reshape(npTotalDoses, np.prod([kmax, jmax, imax])), [dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])
        vmc.saveToVTI(rass_data.output("difference"), np.reshape(mcDosesFluence-npTotalDoses, np.prod([kmax, jmax, imax])), [dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])
    if options["out_total_doses"]:
        log.info("Max total npTotalDoses: %f" % np.max(npTotalDoses))
        vmc.saveToVTI(rass_data.output("totalDoses"), np.reshape(npTotalDoses, np.prod([kmax, jmax, imax])), [dx, dy, dz], [kmax, jmax, imax], [xbase, ybase, zbase])

    ray.timeline("/tmp/ray-timeline.json")
    
    if options["histograms"]:

        def generate_gnuplot_histograms(name, subfolder, doses_to_plot):
            stats_fname = rass_data.output(f"statistics_{name}.txt");
            png_fname = '%s_histogram_%s.png' % (treatment_name, name)

            log.info(f"Generating histograms for {name}... (statistics will be saved to: {stats_fname})")
            
            fout = open(stats_fname,'w')
            f = open(rass_data.output('histograms.gpt', subfolder=subfolder), 'w')
            f.write('set grid\nset style data lp\nset xlabel \'Dose [cGy]\'\n'
                    'set ylabel \'%% of volume\'\nset yrange [0:110]\nset term png size 1024,768\nset output "%s"\nplot ' % png_fname)
            for r in range(len(myROIs)):
                minD, avgD, maxD = histogram(doses_to_plot, roi_marks, 2 ** r, rass_data.output("%s.hist" % myROIs[r].name, subfolder=subfolder), 100. * doseScaling, dv, HIST_PTS)
                
                s = f'Voxel doses in {myROIs[r].name:>20}: min={minD * doseScaling:>12.6} avg={avgD * doseScaling:>12.6} max={maxD * doseScaling:>12.6} [cGy]'
                log.info(s) 
                fout.write(s)

                if maxD > 0:
                    f.write('\'' + myROIs[r].name + '.hist\', ')
            f.write('\n#set term x11 size 1024,768\n#replot\n#pause 120\n')
            f.close()
            fout.close()

        generate_gnuplot_histograms("ecplise_PlanRT", "hist", totalDoses)
        generate_gnuplot_histograms("MC_calculated_with_Fluence_1", "red", mcDoses)
        generate_gnuplot_histograms("MC_calculated_with_PlanRT", "fluence", mcDosesFluence)
