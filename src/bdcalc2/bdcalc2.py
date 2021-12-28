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

        self.ctgriddata = "Call read_ct() to initialize."
        self.map_ctsopid = "Call read_ct() to initialize."
        self.ctdata_array = "Call read_ct() to initialize."
        self.ctdata = "Call read_ct() to initialize."
        self.rtss = "Call read_rs() to initialize."
        self._rdata = "Call read_rs() to initialize."
        self.ctgriddata = "Call read_plan_geometry_from_doses() to initialize."

        self._find_dicoms()

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


        self.ctgriddata = (     ct.ImagePositionPatient[0]*SCALE, ct.ImagePositionPatient[1]*SCALE, minz*SCALE,
                                ct.PixelSpacing[0]*SCALE, ct.PixelSpacing[1]*SCALE, min(thicknesses)*SCALE,
                                int(ct.Columns), int(ct.Rows), len(ctdata))
        self.ctdata = ctdata

        self.ctdata_array = np.zeros( (self.ctnz(), self.ctny(), self.ctnx()) )
        for i, ct in enumerate(self.ctdata):
            self.ctdata_array[i,:,:] = ct["image_data"]

        return self.ctdata_array

    def read_plan_geometry_from_doses(self):
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

            self.plangriddata = (dox, doy, doz, ddx, ddy, ddz, dnx, dny, dnz)

            return self.plangriddata
        else:
            raise Exception("The RTDoses files list is empty. Use _find_dicoms() first, or there are no RT doses files in the dicom folder.")


    def read_plan(self):
        plan = dicom.read_file(self.dicom_files["rtplan"])

        
    def cto(self):
        return self.ctgriddata[0:3]

    def ctox(self):
        return self.ctgriddata[0]

    def ctoy(self):
        return self.ctgriddata[1]

    def ctoz(self):
        return self.ctgriddata[2]

    def ctn(self, inverted=False):
        if inverted:
            return self.ctgriddata[6:9][::-1]
        else:
            return self.ctgriddata[6:9]

    def ctnx(self):
        return int(self.ctgriddata[6])

    def ctny(self):
        return int(self.ctgriddata[7])

    def ctnz(self):
        return int(self.ctgriddata[8])

    def ctd(self):
        return self.ctgriddata[3:6]

    def ctdx(self):
        return self.ctgriddata[3]

    def ctdy(self):
        return self.ctgriddata[4]

    def ctdz(self):
        return self.ctgriddata[5]

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
        return self.plangriddata[0:3]

    def planox(self):
        return self.plangriddata[0]

    def planoy(self):
        return self.plangriddata[1]

    def planoz(self):
        return self.plangriddata[2]

    def plann(self):
        return self.plangriddata[6:9]

    def plannx(self):
        return int(self.plangriddata[6])

    def planny(self):
        return int(self.plangriddata[7])

    def plannz(self):
        return int(self.plangriddata[8])

    def pland(self):
        return self.plangriddata[3:6]

    def plandx(self):
        return self.plangriddata[3]

    def plandy(self):
        return self.plangriddata[4]

    def plandz(self):
        return self.plangriddata[5]

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

    
    def roi_markers(self, roi_name=None, roi_number=None, rdata=None):
        r = None

        if rdata is None:
            rdata = self._rdata

        if roi_number is not None:
            r = self.rtss[roi_number]
        else:
            for k,v in self.rtss.items():
                if v["name"] == roi_name:
                    r = v
        
        if r is not None:
            roi_bit = r["roi_bit"]
            
            return (np.bitwise_and(self.rdata, roi_bit) // roi_bit).astype(np.bool)
        else:
            raise Exception(f"ROI not found for arguments: roi_name={roi_name}, roi_number={roi_number}")            
        

    def read_rs(self):
        self._rdata = np.zeros(self.ctn(inverted=True), dtype=int )
        
        x = self.ct_x_range()
        y = self.ct_y_range()
        xx,yy = np.meshgrid(x,y)
        xxx = xx.ravel()
        yyy = yy.ravel()
        gp = np.vstack( (xxx,yyy) ).T # grid points

        rs = dicom.read_file(self.dicom_files['rtss'])
        rtss = {}
        for seqno, r in enumerate(rs.StructureSetROISequence):
            roi_bit = 2 ** seqno
            roi_number = int(r.ROINumber)
            rtss[roi_number] = {
                "name": r.ROIName,
                "roi_bit": roi_bit
            }

            roi_cached_filename = self.rd.processing(f"_cache_{r.ROIName}.npy")
            if os.path.isfile(roi_cached_filename):
                log.info(f"Loading labels for ROI: {r.ROIName} from cache file: {roi_cached_filename}")
                rdata = np.load(roi_cached_filename)
                log.info(f"Loading done.")
            else: 
                log.info(f"Marking labels for ROI: {r.ROIName} with {len(rs.ROIContourSequence[seqno].ContourSequence)} contours (seqno={seqno})...")
                rdata = np.zeros( (self.ctnz(), self.ctny(), self.ctnx()), dtype=int )

                #if r.ROIName != "Patient Outline":
                # if r.ROIName != "Larynx":
                #     continue

                cseq = rs.ROIContourSequence[seqno].ContourSequence
                for c in cseq:
                    indices, b = f_mark( (c, self.map_ctsopid, gp) ) 

                    b = b.reshape(self.ctny(), self.ctnx())
                    for z_idx in indices:
                        rdata[z_idx, :, :] = np.bitwise_xor( rdata[z_idx, :, :], b )

                log.info(f"Saving to cache file: {roi_cached_filename}")
                np.save(roi_cached_filename, rdata)
                
            # apply roi_bit for all voxels with value one to roi mask 
            self._rdata = np.bitwise_or( self._rdata, rdata*roi_bit )
                       
        self.rtss = rtss
        return rtss

    def mark_voxels(self):
        pass

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
            data = self.ctdata_array
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


if __name__ == "__main__":
    log.info("Starting testing data...")

    rt = RT(sys.argv[1])
    log.info(rt)
    log.info("config.json " + rt.rd.input("config.json"))

    arr = rt.read_ct()
    np.save("/tmp/arr", arr)

    pgi = rt.read_plan_geometry_from_doses()
    log.info(pgi)
    log.info(rt.ctgriddata)

    rtss = rt.read_rs()
    
    rname = "Larynx"
    np.save(f"/tmp/{rname}_roi.npy", rt.roi_markers(roi_name=rname).astype(int))

    bb = np.zeros( rt.ctn()[::-1], dtype=np.float32 )

    xx = np.arange( rt.planox(), rt.planox() + rt.plandx()*rt.plannx(), rt.plandx())
    for px in xx:
        p = (px, rt.planoy(), rt.planoz())
        ip = rt.point2ct_idx(p)
        bb[ip[2], ip[1], ip[0]] = 1

        p = (px, rt.planoy()+rt.plandy()*rt.planny(), rt.planoz())
        ip = rt.point2ct_idx(p)
        bb[ip[2], ip[1], ip[0]] = 1

    np.save("/tmp/bb", bb)

    d = np.zeros( rt.plann()[::-1], dtype=np.float32 )
    n = rt.point2plan_idx( (0,0,0) )
    d[n[2], n[1], n[0]] = 1
    np.save("/tmp/doses", d)

    planct = rt.sample_ct_over_plan()
    np.save("/tmp/planct", planct)
    
    rdata_plan = rt.sample_ct_over_plan(data=rt._rdata)
    np.save("/tmp/rdata_plan", rdata_plan)

#     for roinum,rd in rtss.items():
#         #if rd["name"] == "Patient Outline":
#         if rd["name"] == "Larynx":
#             Larynx = rt.map_ct_over_plan(data=rd['rdata'])
#             np.save("/tmp/Larynx", Larynx)

    log.info(rt.rtss)
    log.info(d.shape)


