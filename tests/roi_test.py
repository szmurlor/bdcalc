import sys
import pydicom
import dicomutils
from common import log
from myroi import MyRoi


dicom_directory = sys.argv[1]
rtss, plan, ctlist, doseslist = dicomutils.find_ct_rs_rp_dicom(dicom_directory)

log.info(f"File name: {rtss}")
rtss = pydicom.read_file(rtss)
log.info(f"Dicomfile: {rtss}")

myROIs = []
idxROIBody = -1
for i in range(0, len(rtss.StructureSetROISequence)):
    roiName = rtss.StructureSetROISequence[i].ROIName
    log.info("Finding contours for %s" % roiName)
    myROIs.append(MyRoi(dicomutils.findContours(rtss, rtss.StructureSetROISequence[i].ROINumber),
                        roiName, float(beams[0].PixelSpacing[0]) / 1000.0))

    if ("body" in roiName.lower() or "skin" in roiName.lower() or "outline" in roiName.lower()) and (idxROIBody == -1):
        idxROIBody = i
        info("Found ROI body (or skin): idx = %d" % idxROIBody)

log.info(myROIs)