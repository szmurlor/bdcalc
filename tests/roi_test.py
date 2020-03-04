import sys
import pydicom
import dicomutils
from common import log
from myroi import MyRoi
import numpy as np


dicom_directory = sys.argv[1]
rtss, plan, ctlist, doseslist = dicomutils.find_ct_rs_rp_dicom(dicom_directory)

log.info(f"File name: {rtss}")
rtss = pydicom.read_file(rtss)
log.info(f"Dicomfile: {rtss}")
beams = [pydicom.read_file(f) for f in doseslist]

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

SCALE = 0.1
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

roi_marks = np.zeros((imax, jmax, kmax), dtype=int)
for r in range(len(myROIs)):        
    myROIs[r].mark(xbase / SCALE, ybase / SCALE, dx / SCALE, dy / SCALE, kmax, jmax, imax,
                    np.linspace(zbase, zbase + (imax - 1) * dz, imax) / SCALE, roi_marks, 2 ** r, ctgriddata=None)

log.info(myROIs)
log.info(myROIs[0].paths)

from matplotlib import pyplot as plt
j = 0
last_z = None
for i, p in enumerate( myROIs[1].paths ):
    if (p is not None):
        if (last_z is None or last_z != myROIs[1].z[i]):
            j += 1 
            plt.clf()
            plt.cla()
            last_z = myROIs[1].z[i]
            log.info(f"last_z = {last_z}, zoffsets[0] ={zoffsets[0]}")
            i_layer = int((last_z - (zoffsets[0]+zbase)/SCALE) / (dz/SCALE))
            log.info(f"i_layer ={i_layer}")
            plt.imshow( roi_marks[i_layer]& 2, extent=(xbase/SCALE, (xbase+dx*kmax)/SCALE, ybase/SCALE, (ybase+dy*jmax)/SCALE), origin='lower')  
        plt.scatter(p.vertices[:,0], p.vertices[:,1])
        plt.savefig(f"c{j}.png", dpi=400)

#for r in range(0, len(myROIs)):
#    log.info("Marking voxels for %s" % myROIs[r].name)
#    log.info("CTGRID DATA %s" % list(ctgriddata))
#    myROIs[r].mark(xbase / SCALE, ybase / SCALE, dx / SCALE, dy / SCALE, kmax, jmax, imax,
#                    np.linspace(zbase, zbase + (imax - 1) * dz, imax) / SCALE, roi_marks, 2 ** r, ctgriddata=ctgriddata)
#    myROIs[r].save_marks(fcache, roi_marks, 2 ** r)

#for r in range(len(myROIs)):
#    log.info("Statistics for %20s: ID=%8d, %7d voxels, vol=%8.1f discrete vol=%8.1f [cm3]" % (
#        myROIs[r].name, 2 ** r, myROIs[r].count, myROIs[r].volume / 1000.,
#        myROIs[r].count * dv / SCALE / SCALE / SCALE / 1000.0))