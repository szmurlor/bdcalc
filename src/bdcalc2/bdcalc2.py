import pydicom as dicom
import numpy as np
import sys
import os
import multiprocessing

from common import log
from rass import RASSData
from matplotlib.path import Path

import dicomutils

SOURCE_TO_SURFACE_DISTANCE = 1000
SCALE = 0.1

class NoRSFileException(Exception):
    def __init__(self, directory):
        Exception.__init__(self, "No RS.* or rtss.* file in %s" % directory) #ray.init(redis_address="10.42.2.78:59999")

def f_mark(args):
    """
        returns: z_indices - indexes of the z-rows (image indexes) in the volume 
                             referenced by the contour
                 b - 2D array of shape (ny,nx) holding boolean values
    """

    # extract variables from tuple
    # expected: map_ctsopid - map: ID to z row index
    #           points - list of points with shape: (n,2) (raveled meshgrid points, later used by reshape)
    contour, map_ctsopid, points = args

    cd = np.array(contour.ContourData) * SCALE
    cd = cd.reshape(cd.shape[0] // 3, 3)
    cd = cd[:,:2] # ignore z axis and build 2D planar points
    p = Path(vertices=cd, closed=True)

    ##############################
    b = p.contains_points(points)
    b = b.astype(int)
    ##############################


    # will hold indices in the rows referenced by the ContourSequence 
    # with SOPInstanceUIDs in the CT sequence
    z_indices = []

    # mark points on all referenced CT slices
    if len(contour.ContourImageSequence) < 1:
        log.warn(f"The Contour Image Sequence is empty with contour data size: " +
                 f"{len(contour.ContourData) // 3} - {len(contour.ContourImageSequence)}.")
    else:
        for im_seq in contour.ContourImageSequence:
            sopid = im_seq.ReferencedSOPInstanceUID
            idxs = map_ctsopid[sopid]['idx']
            z_indices.extend(idxs)

    return z_indices, b

class RT:
    def __init__(self, folder):
        self.rd = RASSData(root_folder=folder)
        self.dicom_files = {}

        self._ctgriddata = None # "Call read_ct() to initialize."
        self.map_ctsopid = "Call read_ct() to initialize."
        self._ctdata_array = None # "Call read_ct() to initialize."
        self._ctdata_plan_array = None # "Call get_ct(plan=True) to initialize."
        self._rtss = None # "Call get_rtss() to initialize."
        self._rdata_ct = None # "Call get_roi_labels_ct() to initialize."
        self._rdata_plan = None # "Call get_roi_labels_plan() to initialize."
        self._plangriddata = None # "Call get_plan_griddata() to initialize."
        self._find_dicoms()
        self._total_doses = None
        self._beam_doses = None

    def __str__(self) -> str:
        s = "[RT: dicom_files:" + str(self.rd) + "\n"
        s += str(self.dicom_files)
        s += "]"

        return s

    def _find_dicoms(self, dicom_folder=None):
        if (dicom_folder is None):
            dicom_folder = self.rd.input("dicom")

        #######################################################################################
        # Searching DICOM files
        #######################################################################################
        rtss, plan, ct_files_list, doses_files_list = dicomutils.find_ct_rs_rp_dicom(dicom_folder)
        if rtss is None:
            raise NoRSFileException(dicom_folder)

        self.dicom_files['rtss'] = rtss
        self.dicom_files['rtplan'] = plan
        self.dicom_files['ct'] = ct_files_list
        self.dicom_files['rtdoses'] = doses_files_list

    def get_ct(self, plan=False):
        if self._ctdata_array is None:
            self.read_ct()
        if plan:
            if self._ctdata_plan_array is None:
                self._ctdata_plan_array = self.sample_ct_over_plan()
            return self._ctdata_plan_array
        else:
            return self._ctdata_array


    def read_ct(self):
        self.map_ctsopid = {}
        
        ctdata = []
        minz =  None
        for ctf in self.dicom_files["ct"]:
            ct = dicom.read_file(ctf)

            if minz is None or ct.ImagePositionPatient[2] < minz:
                minz = ct.ImagePositionPatient[2]

            d = {
                    "location": ct.SliceLocation, 
                    "thickness": float(ct.SliceThickness), 
                    "image_data": ct.pixel_array, 
                    "SOPInstanceUID": ct.SOPInstanceUID,
                    "idx": []
                }
            ctdata.append(d)
            self.map_ctsopid[ct.SOPInstanceUID] = d

        ctdata = sorted(ctdata, key=lambda v: v["location"])
        for idx, d in enumerate(ctdata):
            d["idx"].append(idx)

        # get the slice thickness
        thicknesses = []
        thicknesses_dc = []
        for i in range(len(ctdata)-1):
            thicknesses.append(ctdata[i+1]["location"] - ctdata[i]["location"])
            thicknesses_dc.append(ctdata[i]["thickness"])

        if (min(thicknesses) != max(thicknesses)):
            raise Exception(f"Invalid data! The distances (based on location) between CT slices are not equal: min: {min(thicknesses)} != max: {max(thicknesses)}")

        if (min(thicknesses_dc) != max(thicknesses_dc)):
            raise Exception(f"Invalid data! The distances (based on DICOM data) between CT slices are not equal: min: {min(thicknesses)} != max: {max(thicknesses)}")

        if (min(thicknesses_dc) != min(thicknesses)):
            raise Exception(f"Invalid data! The distances between CT slices are not equal in the DICOM data and based on location atribute: min_dicom_data: {min(thicknesses_dc)} != min_location: {min(thicknesses)}")


        self._ctgriddata = (     ct.ImagePositionPatient[0]*SCALE, ct.ImagePositionPatient[1]*SCALE, minz*SCALE,
                                ct.PixelSpacing[0]*SCALE, ct.PixelSpacing[1]*SCALE, min(thicknesses)*SCALE,
                                int(ct.Columns), int(ct.Rows), len(ctdata))

        self._ctdata_array = np.zeros( (self.ctnz(), self.ctny(), self.ctnx()) )
        for i, ct in enumerate(ctdata):
            self._ctdata_array[i,:,:] = ct["image_data"]

        return self._ctdata_array

    def get_ct_griddata(self):
        if self._ctgriddata is None:
            self.get_ct()

        return self._ctgriddata

    def get_plan_griddata(self):
        if self._plangriddata is None:
            dfs = self.dicom_files["rtdoses"]
            if len(dfs) > 0:
                # read from the first file
                doses = dicom.read_file(dfs[0])

                dnx = doses.Columns
                dny = doses.Rows
                dnz = int(doses.NumberOfFrames)

                ddx = doses.PixelSpacing[0] * SCALE
                ddy = doses.PixelSpacing[1] * SCALE

                if (len(doses.GridFrameOffsetVector) > 2):
                    zoffsets = list(map(float, doses.GridFrameOffsetVector))
                    for i in range(len(zoffsets)):
                        zoffsets[i] *= SCALE
                    ddz = zoffsets[1] - zoffsets[0]
                    ddv = ddx * ddy * ddz
                else:
                    raise Exception(f"GridFrameOffsetVector in the {dfs[0]} has less than 2 items.")

                dox = float(doses.ImagePositionPatient[0]) * SCALE
                doy = float(doses.ImagePositionPatient[1]) * SCALE
                doz = float(doses.ImagePositionPatient[2]) * SCALE

                self._plangriddata = (dox, doy, doz, ddx, ddy, ddz, dnx, dny, dnz)
            else:
                raise Exception("The RTDoses files list is empty. Use _find_dicoms() first, or there are no RT doses files in the dicom folder.")

        return self._plangriddata

    def read_plan(self):
        plan = dicom.read_file(self.dicom_files["rtplan"])

        
    def cto(self):
        return self.get_ct_griddata()[0:3]

    def ctox(self):
        return self.get_ct_griddata()[0]

    def ctoy(self):
        return self.get_ct_griddata()[1]

    def ctoz(self):
        return self.get_ct_griddata()[2]

    def ctn(self, inverted=False):
        if inverted:
            return self.get_ct_griddata()[6:9][::-1]
        else:
            return self.get_ct_griddata()[6:9]

    def ctnx(self):
        return int(self.get_ct_griddata()[6])

    def ctny(self):
        return int(self.get_ct_griddata()[7])

    def ctnz(self):
        return int(self.get_ct_griddata()[8])

    def ctd(self):
        return self.get_ct_griddata()[3:6]

    def ctdx(self):
        return self.get_ct_griddata()[3]

    def ctdy(self):
        return self.get_ct_griddata()[4]

    def ctdz(self):
        return self.get_ct_griddata()[5]

    def ct_x_range(self, center=True):
        dx2 = self.ctdx()/2 if center else 0
        return np.arange(self.ctox() + dx2, 
                         self.ctox()+(self.ctnx())*self.ctdx() + dx2, 
                         self.ctdx())

    def ct_y_range(self, center=True):
        dy2 = self.ctdy()/2 if center else 0
        return np.arange(self.ctoy() + dy2, 
                         self.ctoy() + (self.ctny()) * self.ctdy() + dy2, 
                         self.ctdy())

    def ct_z_range(self, center=True):
        dz2 = self.ctdz()/2 if center else 0
        return np.arange(self.ctoz() + dz2, 
                         self.ctoz() + (self.ctnz()) * self.ctdz() + dz2, 
                         self.ctdz())

    def plano(self):
        return self.get_plan_griddata()[0:3]

    def planox(self):
        return self.get_plan_griddata()[0]

    def planoy(self):
        return self.get_plan_griddata()[1]

    def planoz(self):
        return self.get_plan_griddata()[2]

    def plann(self, inverted=False):
        if inverted:
            return self.get_plan_griddata()[6:9][::-1]
        else:
            return self.get_plan_griddata()[6:9]

    def plannx(self):
        return int(self.get_plan_griddata()[6])

    def planny(self):
        return int(self.get_plan_griddata()[7])

    def plannz(self):
        return int(self.get_plan_griddata()[8])

    def pland(self):
        return self.get_plan_griddata()[3:6]

    def plandx(self):
        return self.get_plan_griddata()[3]

    def plandy(self):
        return self.get_plan_griddata()[4]

    def plandz(self):
        return self.get_plan_griddata()[5]

    def plan_z_range(self, center=True):
        _dz = self.plandz() / 2 if center else 0
        return np.arange(
                self.planoz() + _dz, 
                self.planoz() + self.plannz()*self.plandz() + _dz, 
                self.plandz() 
            )


    def point2ct_idx(self, p):
        """
         p - a tuple representing a point (x,y,z), or an array 
             with n rows and 2 or 3 columns (n,2 or 3), 
             representing a list of points
         returns - index coordinates (int) of the voxel containing point 
                   (or a list of coordinates) 
        """
        if type(p) is tuple or type(p) is list:
            # single point
            return (
                int(round( (p[0] - self.ctox()) / self.ctdx())),
                int(round( (p[1] - self.ctoy()) / self.ctdy())),
                int(round( (p[2] - self.ctoz()) / self.ctdz())),
            )
        else:
            # array or matrix of point coordinates
            res = np.zeros(p.shape, dtype=int)
            res[:,0] = np.round((p[:,0] - self.ctox()) / self.ctdx())
            res[:,1] = np.round((p[:,1] - self.ctoy()) / self.ctdy())
            if len(res.shape) > 2:
                res[:,2] = np.round((p[:,2] - self.ctoz()) / self.ctdz())

            return res.astype(int)


    def point2plan_idx(self,p):
        """
         p - a tuple representing a point (x,y,z), or an array 
             with n rows and 2 or 3 columns (n,2 or 3), 
             representing a list of points
         returns - index coordinates (int) of the voxel containing point 
                   (or a list of coordinates) 
        """
        if type(p) is tuple or type(p) is list:
            # single point
            return (
                int(round( (p[0] - self.planox())  / self.plandx())),
                int(round( (p[1] - self.planoy())  / self.plandy())),
                int(round( (p[2] - self.planoz())  / self.plandz()))
            )
        else:
            # array or matrix of point coordinates
            res = np.zeros(p.shape, dtype=int)
            res[:,0] = np.round((p[:,0] - self.planox()) / self.plandx())
            res[:,1] = np.round((p[:,1] - self.planoy()) / self.plandy())
            if len(res.shape) > 2:
                res[:,2] = np.round((p[:,2] - self.planoz()) / self.plandz())

            return res.astype(int)

    
    def roi_markers(self, roi_name=None, roi_number=None, rdata=None, plan=False):
        """
        By default returns roi markes for CT grid.
        """
        r = None

        if rdata is None:
            if plan:
                rdata = self.get_roi_labels_plan()
            else:
                rdata = self.get_roi_labels_ct()

        if roi_number is not None:
            r = self.get_rtss()[roi_number]
        else:
            for k,v in self.get_rtss().items():
                if v["name"] == roi_name:
                    r = v
        
        if r is not None:
            roi_bit = r["roi_bit"]
            
            return (np.bitwise_and(rdata, roi_bit) // roi_bit).astype(np.bool)
        else:
            raise Exception(f"ROI not found for arguments: roi_name={roi_name}, roi_number={roi_number}")            
        

    def get_rtss(self, force=False):
        if self._rtss is None or force:
            rs = dicom.read_file(self.dicom_files['rtss'])

            rtss = {}
            for seqno, r in enumerate(rs.StructureSetROISequence):
                roi_bit = 2 ** seqno
                roi_number = int(r.ROINumber)
                rtss[roi_number] = {
                    "name": r.ROIName,
                    "roi_bit": roi_bit
                }
            self._rtss = rtss
        
        return self._rtss


    def get_roi_labels_ct(self, force_refresh=False, force_relabel=False):
        if self._rdata_ct is None or force_refresh:
            roi_cached_filename = self.rd.processing(f"_cache_rois_ct.npy")
            if os.path.isfile(roi_cached_filename) and not force_relabel:
                log.info(f"Loading labels for CT ROIs from cache file: {roi_cached_filename}")
                self._rdata_ct = np.load(roi_cached_filename)
                log.info(f"Loading CT ROI labels done.")
            else:
                self._rdata_ct = np.zeros(self.ctn(inverted=True), dtype=int )
                
                x = self.ct_x_range()
                y = self.ct_y_range()
                xx,yy = np.meshgrid(x,y)
                xxx = xx.ravel()
                yyy = yy.ravel()
                gp = np.vstack( (xxx,yyy) ).T # grid points

                rs = dicom.read_file(self.dicom_files['rtss'])

                for seqno, r in enumerate(rs.StructureSetROISequence):
                    roi_number = int(r.ROINumber)
                    roi_bit = self.get_rtss()[roi_number]["roi_bit"]
                    
                    log.info(f"Marking labels for ROI: {r.ROIName} with {len(rs.ROIContourSequence[seqno].ContourSequence)} contours (seqno={seqno})...")
                    rdata = np.zeros( (self.ctnz(), self.ctny(), self.ctnx()), dtype=int )

                    cseq = rs.ROIContourSequence[seqno].ContourSequence
                    for c in cseq:
                        indices, b = f_mark( (c, self.map_ctsopid, gp) ) 

                        b = b.reshape(self.ctny(), self.ctnx())
                        for z_idx in indices:
                            rdata[z_idx, :, :] = np.bitwise_xor( rdata[z_idx, :, :], b )

                    # apply roi_bit for all voxels with value one to roi mask 
                    self._rdata_ct = np.bitwise_or( self._rdata_ct, rdata*roi_bit )

                log.info(f"Saving to cache file for CT ROI labels: {roi_cached_filename} with approximate file length: {self._rdata_ct.nbytes}")
                np.save(roi_cached_filename, self._rdata_ct)
                        
        return self._rdata_ct


    def get_roi_labels_plan(self, force_refresh=False):
        if self._rdata_plan is None or force_refresh:
            self._rdata_plan = self.sample_ct_over_plan(data=self.get_roi_labels_ct())

        return self._rdata_plan


    def get_plan_grid_points(self):

        x = np.arange(self.planox() + self.plandx()/2, 
                self.planox()+(self.plannx())*self.plandx() + self.plandx()/2, 
                self.plandx())
        y = np.arange(self.planoy() + self.plandy()/2, 
                      self.planoy()+(self.planny())*self.plandy() + self.plandy()/2, 
                      self.plandy())
        xx,yy = np.meshgrid(x,y)
        xxx = xx.ravel()
        yyy = yy.ravel()
        gp = np.vstack( (xxx,yyy) ).T # grid points

        return gp


    def sample_ct_over_plan(self, data=None):
        if data is None:
            data = self.get_ct()
        else:
            if np.prod(data.shape) != np.prod(self.ctn()):
                raise Exception(f"Invalid data to map_ct_over_plan. Expected shape: {self.ctn()}, got: {data.shape}")

        # najpierw tworzę macierz o takich samych rozmiarach co plan
        ct_over_plan = np.zeros(self.plann()[::-1], dtype=data.dtype)

        # pobieram współrzędne środków wokseli 
        gp = self.get_plan_grid_points()

        # mapuję współrzędne srodków wokseli na indeksy w siatce
        gp_idx = self.point2ct_idx(gp)

        # przechodzę po wszystkich przekrojach i wyciągam wartość CT dla współrzędnych indeksowych
        for plan_iz, z in enumerate(self.plan_z_range()):
            _,_,ct_iz = self.point2ct_idx((0,0, z))
            ct_over_plan[plan_iz,:,:] = data[ct_iz, gp_idx[:,1], gp_idx[:,0]].reshape( (self.planny(), self.plannx()) )

        return ct_over_plan

    def get_doses(self, beam_no=None):
        if self._total_doses is None:
            self._beams = {}
            plan = dicom.read_file(self.dicom_files['rtplan'])
            for beam in plan.BeamSequence:
                bno = int(beam.BeamNumber)
                rows = beam.CompensatorSequence[0].CompensatorRows
                cols = beam.CompensatorSequence[0].CompensatorColumns
                data = np.array(beam.CompensatorSequence[0].CompensatorTransmissionData)
                data = data.reshape((rows,cols))
                self._beams[bno] = {
                    'fluence_data': data,
                    'sourceDistance': beam.SourceAxisDistance
                }

            self._total_doses = np.zeros(self.plann(inverted=True))
            doses = []
            for dose_filename in self.dicom_files['rtdoses']:
                d = dicom.read_file(dose_filename)
                print(f"Read {dose_filename}")
                doses.append(d)

                bn = 0
                try:
                    bn = int(d.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber)
                    self._beams[bn]['doses'] = d.pixel_array
                    self._total_doses[:] = self._total_doses + d.pixel_array
                except Exception as e:
                    log.error(e)

        return self._total_doses




        if beam_no is None:
            return self._total_doses
        else:
            return self._beam_doses[beam_no]


if __name__ == "__main__":
    log.info("Starting testing data...")

    rt = RT(sys.argv[1])
    ds = rt.get_doses()
    print(ds.shape)
    np.save("/tmp/doses", ds)
    # log.info(rt)

    # arr = rt.get_ct()
    # log.info(f"The arr ct has shape:{arr.shape}")
    # np.save("/tmp/arr", arr)


    # pgi = rt.get_plan_griddata()
    # log.info(pgi)
    # log.info(rt.get_ct_griddata())

    # rtss = rt.get_rtss()
    
    # rname = "Larynx"
    # np.save(f"/tmp/{rname}_roi_ct.npy", rt.roi_markers(roi_name=rname).astype(int))

    # # Here testing the mapping of the geomterical coordinates to the CT index coordinates
    # bb = np.zeros( rt.ctn()[::-1], dtype=np.float32 )
    # xx = np.arange( rt.planox(), rt.planox() + rt.plandx()*rt.plannx(), rt.plandx())
    # for px in xx:
    #     p = (px, rt.planoy(), rt.planoz())
    #     ip = rt.point2ct_idx(p)
    #     bb[ip[2], ip[1], ip[0]] = 1

    #     p = (px, rt.planoy()+rt.plandy()*rt.planny(), rt.planoz())
    #     ip = rt.point2ct_idx(p)
    #     bb[ip[2], ip[1], ip[0]] = 1
    # np.save("/tmp/bb", bb)

    # # Here testing the mapping of the geomterical coordinates to the PLAN index coordinates
    # d = np.zeros( rt.plann()[::-1], dtype=np.float32 )
    # n = rt.point2plan_idx( (0,0,0) )
    # d[n[2], n[1], n[0]] = 1
    # np.save("/tmp/doses", d)

    # # Here testing the sampling over the planning grid of a data.
    # planct = rt.sample_ct_over_plan()
    # np.save("/tmp/planct", planct)
    

    # rdata_plan = rt.get_roi_labels_plan()
    # # rdata_plan = rt.sample_ct_over_plan(data=rt._rdata_ct)
    # np.save("/tmp/rdata_plan", rdata_plan)

    # log.info(rt.get_rtss())


