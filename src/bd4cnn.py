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
from beamlets import Beamlets
from sys import argv
from myroi import MyRoi
from rass import RASSData

from common import log

SOURCE_TO_SURFACE_DISTANCE = 1000

makeControlData = False
SCALE = 0.1


class NoRSFileException(Exception):
    def __init__(self, directory):
        Exception.__init__(self, "No RS.* or rtss.* file in %s" % directory) #ray.init(redis_address="10.42.2.78:59999")


def default_options():
    return {
        "debug_level": "info"
    }


if __name__ == '__main__':
    ctgriddata = None

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
    # rass_data.input()
    # rass_data.input("dicom")
    # rass_data.output()

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
    #path, rtss, plan = utils.findRSandRP(dicom_directory)
    rtss, plan, ctlist, doseslist = dicomutils.find_ct_rs_rp_dicom(rass_data.input("dicom"))
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
    reader = CTVolumeDataReader(rass_data.input("dicom"), ctfiles=ctlist)
    ctVolumeData = reader.read()
    cdData = ctVolumeData.getCTDataAsNumpyArray()


    # ctlist = (glob.glob(dicom_directory + '/ct*') + glob.glob(dicom_directory + '/CT*'))
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
            #log.info(f"Read beam {dir(beam.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0])}")
            bn = int(beam.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber)
        except:
            print("Semething wrong went...")
            if totalDoses is None:
                singleBeam = True
                totalDoses = beam.pixel_array.copy()
                totalDosesFile = beam.filename
            continue
        log.info(f"Read beam {bn}")
        beamDoses[bn] = beam.pixel_array
        if doseScaling is not None and float(beam.DoseGridScaling) != doseScaling:
            log.warning('Strange data: DoseGridScaling is not same all beams!')
        log.debug('Got doses from bundle %d' % bn)

    if not singleBeam:
        bns = beamDoses.keys()
        totalDoses = beamDoses[bns[0]].copy()
        for i in range(1, len(bns)):
            log.info(f"Adding doses from beam {i}")
            totalDoses += beamDoses[bns[i]]

    totalDoses = np.array(totalDoses, dtype=np.float32)
    log.info("Read doses for %d beams" % len(beamDoses))
    npTotalDoses = np.array(totalDoses, dtype=np.float32)

    minDose = np.min(totalDoses)
    averageDose = np.average(totalDoses)
    maxDose = np.max(totalDoses)

    if totalDosesFile is None:
        log.info('Total doses calculated as sum of beam doses (min dose=%f, average dose=%f, max dose=%f, doseScaling=%f)' % (
            minDose, averageDose, maxDose, doseScaling))
    else:
        log.info('Got total doses from file %s (min dose=%f, average dose=%f, max dose = %f, doseScaling=%f)' % (
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
    log.info('Planning grid: %d x %d x %d in [%g:%g]x[%g:%g]x[%g:%g] dx,dy,dz=%g,%g,%g -> dv=%g' % (
        kmax, jmax, imax,
        xbase, xbase + kmax * dx, ybase, ybase + jmax * dy, zbase + zoffsets[0], zbase + zoffsets[-1],
        dx, dy, dz, dv))

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
        log.info("Finding contours for %s" % roiName)
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


    ######################################################################
    log.info('Processing bundles')
    ######################################################################
    for beamlets in all_beamlets:
        beamNo = beamlets.beam_number

        beamlets_doses_cachefile = rass_data.processing("%s_%d.beamlets_doses_map_cache" % (treatment_name, beamNo))
        beam_doses_cachefile = rass_data.processing("%s_%d.beam_doses_cache" % (treatment_name, beamNo))
        if options["use_cached_doses"]:
            beamTotalDoses = read_beam_doses(beam_doses_cachefile, mcDosesVMC.shape)
            beamlets_doses = read_beamlets_doses_map(beamlets_doses_cachefile)
            needs_calculate = beamTotalDoses is None or beamlets_doses is None

    print("max v2Drow {}".format(np.max(v2Drow)))
