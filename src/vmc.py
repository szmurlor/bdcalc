# -*- coding: utf-8 -*-
"""Skrypt do uruchamiania VMC++
"""
import sys
import os
import random
import _thread as thread
import json
import struct
import numpy
import time
import ray
import traceback
import shutil
import socket

import subprocess
from ct import CTVolumeDataReader
from volumedata import VolumeData
from dicomutils import debug, error, warning, info


DEBUG_MAX_BEAMLETS_KEY = "debug_max_beamlets"

def write_beamlet(beamlet, filename, opt):
    o = beamlet["origin"]
    v1 = beamlet["v1"]
    v2 = beamlet["v2"]
    v3 = beamlet["v3"]

    f = open(filename, "w")
    f.write("! RASS Beam: " + beamlet["name"] + "\n\n")
    f.write(" :start scoring options:\n")
    f.write("     start in geometry: phantom\n")
    f.write("      :start dose options:\n")
    f.write("          score in geometries: phantom\n")
    f.write("          score dose to water: True\n")
    f.write("      :stop dose options:\n")
    f.write("      :start output options phantom:\n")
    f.write("          dump dose:  1\n")
    f.write("      :stop output options phantom:\n")
    f.write(" :stop scoring options:\n")

    f.write(" :start geometry:\n")
    f.write("     :start XYZ geometry:\n")
    f.write("         my name = phantom\n")
    f.write("         method of input = CT-PHANTOM\n")
    f.write("         phantom file    = %s\n" % opt["ct_file"])
    f.write("     :stop XYZ geometry:\n")
    f.write(" :stop geometry:\n")
    f.write("\n")
    f.write(" :start beamlet source:\n")
    f.write("     my name = source 1\n")
    f.write("     monitor units source 1 = 1\n")
    f.write("\n")
    f.write("     spectrum = %s\n" % opt["spectrum_filename"])
    f.write("     charge       = 0\n")
    f.write("     beamlet edges = %f %f %f %f %f %f %f %f %f\n" % (v1["x"], v1["y"], v1["z"], v2["x"], v2["y"], v2["z"], v3["x"], v3["y"], v3["z"]))
    f.write("     virtual point source position = %f %f %f\n" % (o["x"], o["y"], o["z"]))
    f.write(" :stop beamlet source:\n")
    f.write("\n")
    f.write(" :start MC Parameter:\n")
    f.write("     automatic parameter = yes\n")
    f.write(" :stop MC Parameter:\n")
    f.write("\n")
    f.write(" :start MC Control:\n")
    f.write("     ncase  = %s\n" % opt["ncase"])
    f.write("     nbatch = %s\n" % opt["nbatch"])
    f.write("     rng seeds = %d   %d\n" % (int(random.random() * 30000), int(random.random() * 30000)))
    f.write(" :stop MC Control:\n")
    f.write("\n")
    f.write(" :start variance reduction:\n")
    f.write("     repeat history   = 0.251\n")
    f.write("     split photons = Yes\n")
    f.write("     photon split factor = -40\n")
    f.write(" :stop variance reduction:\n")
    f.write("\n")
    f.write(" :start quasi:\n")
    f.write("     base      = 2\n")
    f.write("     dimension = 60\n")
    f.write("     skip      = 1\n")
    f.write(" :stop quasi:")
    f.close()


# noinspection PyUnusedLocal
def read_doses(filename):
    f = open(filename, 'rb')
    data = f.read(5*4)
    nreg = struct.unpack_from('i', data, 0)[0]
    ncase = struct.unpack_from('l', data, 4)[0]
    nbatch = struct.unpack_from('l', data, 8)[0]
    n_beamlet = struct.unpack_from('i', data, 12)[0]
    n_dumpdose = struct.unpack_from('i', data, 16)[0]
    if (n_dumpdose == 1):
        doses = numpy.fromfile(f, numpy.float32, nreg)
        errors = numpy.fromfile(f, numpy.float32, nreg)
    else:
        data = f.read(8)
        dmax = struct.unpack_from('d', data, 0)[0]
        doses = numpy.fromfile(f, numpy.short, nreg)
    f.close()
    return doses

def ala(x,y):
    return x+y

@ray.remote
def calculate_single_beamlet(beamlets, opt):
    res = {"beamlets": []}
    try:
        condInterestingVoxels = read_matrix(opt["interesting_voxels"])
        dose_tolerance_min = float(opt["dose_tolerance_min"])
        beam_no = opt["beam_no"]
        hostname = socket.gethostname()

        import tempfile

        with tempfile.TemporaryDirectory() as node_processing_folder:
            print("Using node processing folder: %s" % node_processing_folder)

            first_idx = None
            last_idx = None
            for beamlet in beamlets:
                print(f"Processing beamlet no: {beamlet}")
                idx = beamlet["idx"]
                print(f"Beamlet idx is: {idx}")

                # --------------------- OBLICZ UZYWAJAC VNC ----------------------------------------------------
                vmc_beamlet_spec_filename = "%s/beamlet_%s.vmc" % (node_processing_folder, idx)
                vmc_beamlet_spec_name = "beamlet_%s" % idx
                write_beamlet(beamlet, vmc_beamlet_spec_filename, opt)
                write_beamlet(beamlet, "/tmp/akuku.vmc", opt)

                #if "ncpu" in opt and opt["ncpu"] > 1:
                print("Calling in parallel: %s/vmc_wrapper %s %s %s %s %s" % (opt["vmc_home"], opt["vmc_home"], node_processing_folder, opt["xvmc_dir"], "%s/bin/vmc_Linux.exe" % opt["vmc_home"], vmc_beamlet_spec_name))
                p = subprocess.Popen(["%s/vmc_wrapper" % opt["vmc_home"], opt["vmc_home"], node_processing_folder, opt["xvmc_dir"], "%s/bin/vmc_Linux.exe" % opt["vmc_home"], vmc_beamlet_spec_name], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                p.wait()
                #else:
                #    print("Calling sequential: %s/vmc_wrapper %s %s %s %s %s" % (opt["vmc_home"], opt["vmc_home"], node_processing_folder, opt["xvmc_dir"], "%s/bin/vmc_Linux.exe" % opt["vmc_home"], vmc_beamlet_spec_name))
                #    p = subprocess.Popen(["%s/vmc_wrapper" % opt["vmc_home"], opt["vmc_home"], node_processing_folder, opt["xvmc_dir"], "%s/bin/vmc_Linux.exe" % opt["vmc_home"], vmc_beamlet_spec_name])
                #    p.wait()

                doses_filename = "%s/%s_phantom.dos" % (node_processing_folder, vmc_beamlet_spec_name)

                beamlet_doses = read_doses(doses_filename)
                # --------------------------------------------------------------------------------------------

                if opt["delete_doses_file_after"]:
                    os.remove(doses_filename)

                if opt["delete_vmc_file_after"]:
                    os.remove(vmc_beamlet_spec_filename)

                if beamlet_doses is not None:
                    last_idx = idx

                    if condInterestingVoxels is not None:
                        ####################################################################################################
                        # Wybierz tylko dawki voxeli, których wartość większa od 0.001 mediany oraz należy do jakiegoś ROIa
                        # Wynik zapisz w macierzy nwierszy na dwie kolumny. Pierwsza kolumna to indeks voxela, druga dawka
                        ####################################################################################################
                        maxDoseThisBeamlet = numpy.max(beamlet_doses)
                        print("Wycinam tylko voksele, w ktorych dawka jest wieksza od: %f (%f%%)" % (maxDoseThisBeamlet * dose_tolerance_min, dose_tolerance_min))
                        print(f"Max dose in beamlet_doses={numpy.max(beamlet_doses)}")
                        cond = (beamlet_doses > (maxDoseThisBeamlet * dose_tolerance_min)) & (condInterestingVoxels)
                        vdoses = beamlet_doses[cond]
                        vindexes = numpy.where(cond)[0]  # zwraca indeksy pasujących
                        print(f"Max dose in vdoses={numpy.max(vdoses)}")
                        mdoses = numpy.zeros((len(vdoses), 2), dtype=numpy.float32)
                        mdoses[:, 0] = vindexes
                        mdoses[:, 1] = vdoses

                        beamlet["doses_map"] = mdoses
                else:
                    print("ERROR! beamlet_doses == None!")

                res['beamlets'].append(beamlet)

        return res
    except:
        traceback.print_exc()
        return None


def save_matrix(mat, fname):
    dt = mat.dtype.name
    type = -1
    if dt == 'bool':
        type = 1
    if dt == 'float32':
        type = 2

    if type == -1:
        raise Exception("Unsupported dtype=%s for matrix in save_matrix." % dt)

    fout = open(fname, "wb")
    # format pliku
    btype = struct.pack("i", type)
    fout.write(btype)

    # wymiar macierzy
    dim = len(mat.shape)
    bdint = struct.pack("i", dim)
    fout.write(bdint)

    # poszczegolne rozmiary wymiarow
    for d in range(dim):
        bnint = struct.pack("i", mat.shape[d])
        fout.write(bnint)

    mat.tofile(fout)
    fout.close()


# noinspection PyUnresolvedReferences
def read_matrix(fname):
    mat = None
    if os.path.isfile(fname):
        fin = open(fname, "rb")

        btype = fin.read(4)
        type = struct.unpack("i", btype)[0]

        if type == 1:
            dtype = numpy.bool_
        elif type == 2:
            dtype = numpy.float32
        else:
            raise Exception("Unsupported dtype=%d for matrix in save_matrix." % type)

        bdint = fin.read(4)
        dim = struct.unpack("i", bdint)[0]
        sizes = []
        for d in range(dim):
            bsint = fin.read(4)
            s = struct.unpack("i", bsint)[0]
            sizes.append(s)
        shape = tuple(sizes)

        nint = numpy.prod(shape)
        mat = numpy.fromfile(fin, dtype, nint)
        mat = numpy.reshape(mat, shape)

        fin.close()

    return mat


def saveToVTI(filename, beamlet_doses, spacing, n, orig):
    # Zapisywanie danych do formatu vti
    volume = VolumeData()
    grid = volume.createGrid(spacing, n, orig)
    ar = volume.createFloatArray(grid)
    ar.SetVoidArray(beamlet_doses, n[0] * n[1] * n[2], 1)
    grid.GetPointData().SetScalars(ar)
    volume = VolumeData(grid)
    volume.save(filename)


class VMC:
    def __init__(self, rass_data):
        self.rass_data = rass_data
        self.delete_doses_file_after = None
        self.delete_vmc_file_after = None
        self.conf_data = None
        self.dicom_folder = None
        self.ct_file = None
        self.run_folder = None
        self.vmc_home = None
        self.vmc_runs = None
        self.xvmc_dir = None
        self.n = None
        self.orig = None
        self.spacing = None
        self.total_doses = None
        self.plan_n = None
        self.plan_origin = None
        self.plan_spacing = None
        self.ncase = None
        self.nbatch = None
        self.ctfiles = None

        # To jest infromacja o ile trzeba było rozszerzyć siatkeplanowania
        # aby pokryła cały Patient outline
        # W formacie: -dx, -dy, -dz, +dx, +dy, +dz
        # gdzie -dx, -dy, -dz to współrzędne o ile trzeba było rozszerzyć od "dołu".
        # gdzie +dx, +dy, +dz to współrzędne o ile trzeba było rozszerzyć od "góry".
        self.augmented_bbox = None
        # W formacie: -ndx, -ndy, -ndz, +ndx, +ndy, +ndz
        # gdzie -ndx, -ndy, -ndz to liczby voxeli o ile trzeba było rozszerzyć od "dołu".
        # gdzie +ndx, +dny, +dzn to liczby voxeli o ile trzeba było rozszerzyć od "góry".
        self.augmented_nbbox = None

        # Z poniższą tolerancją do wartości średniej będą brane dawki voxeli do mapy wyniku
        self.dose_tolerance_min = 0.001

        #self.spectrum_filename = "./spectra/var_6MV.spectrum"
        #self.spectrum_filename = "./spectra/var_CL2300_5_X6MV.spectrum"
        self.spectrum_filename = None

        # Wszystko powinno być zapisane w centymetrach
        self.geometric_scale = None
        self.lock = thread.allocate_lock()
        self.ncpu = 1
        self.debug_max_beamlets = None

        # If water phantom, then fill in CT with ones everywhere.
        self.water_phantom = False

        self.beam_no = None
        self.doses_dos_path = None
        self.cluster_config_file = None

    def postprocess(self, response):
        start = time.time()

        for beamlet in response['beamlets']:
            beamlet_idx = beamlet['idx']
            info("Starting postprocessing of beamlet [%d]" % beamlet_idx)
            mdoses = beamlet["doses_map"]
            info("Size of interesting doses for %s is: %d" % (beamlet_idx, mdoses.shape[0]))

            print(f"max mdoses = {numpy.max(mdoses[:,1])}")
            self.total_doses[mdoses[:,0].astype(int)] = self.total_doses[mdoses[:,0].astype(int)] + mdoses[:,1]

            self.lock.acquire()
            try:
                self.beamlets_doses[beamlet_idx] = mdoses
            except:
                traceback.print_exc()
            finally:
                self.lock.release()

        end = time.time()
        info("Postprocessing time: %s s" % (end - start))


    def run(self, config_file=None, v2Drow=None, voxels=None, options=None, ctfiles=None):
        self.ctfiles = ctfiles
        self.check_config(config_file)

        if not os.path.isdir(self.vmc_home):
            info("Created new folder for vmc++: %s" % self.vmc_home)
            shutil.copytree("%s/vmc++_dist" % os.path.dirname(os.path.realpath(__file__)), self.vmc_home)

        """
        # ustawiam się w katalogu tego skryptu bo pliki są ustalone względem tej ścieżki
        oldWorkingDirectory = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        """
        oldWorkingDirectory = os.getcwd()
        os.chdir(self.vmc_home)

        ct_ramp_in = self.rass_data.input("ct_ramp.data")
        if os.path.isfile(ct_ramp_in):
            ct_ramp_dest = "%s/data/ct_ramp.data" % self.vmc_home
            info("Taking ct_ramp.data from input folder: %s to %s" % (ct_ramp_in, ct_ramp_dest))
            shutil.copy(ct_ramp_in, ct_ramp_dest)

        spectrum_in = self.rass_data.input("spectrum")
        if os.path.isfile(spectrum_in):
            spectrum_dest = "%s" % self.spectrum_filename
            info("Taking energy spectrum from input folder: %s, to file: %s" % (spectrum_in, spectrum_dest))
            shutil.copy(spectrum_in, spectrum_dest)

        

        debug("Size of v2Drow = %d" % v2Drow.shape)
        self.condInterestingVoxels = (v2Drow >= 0)
        debug("Shape of condInterestingVoxels = %d" % self.condInterestingVoxels.shape)

        interesting_voxels_file = self.rass_data.processing('interesting_voxels.dat');
        save_matrix(self.condInterestingVoxels, interesting_voxels_file)


        ######################################################################################################
        # Tutaj tworzę transformowany obraz CT. Obraz musi byc rozszerzony co najmniej do konturów Patient
        # outline, albo skin albo body.
        ######################################################################################################
        self.prepare_ct_file(v2Drow)

        if self.total_doses is None:
            self.total_doses = numpy.zeros(self.n[0] * self.n[1] * self.n[2], dtype=numpy.float32)

        self.beamlets_doses = {}
        opt = self.opt_for_job()
        opt["interesting_voxels"] = interesting_voxels_file
        if "dose_tolerance_min" in options:
            opt["dose_tolerance_min"] = options["dose_tolerance_min"]

        bb = []
        imax = 0
        for b in self.conf_data['beamlets']:
            bb.append(b)
            imax += 1
            if "ray_calc_max_beamlets" in options and options["ray_calc_max_beamlets"] >= 0:
                if imax >= options["ray_calc_max_beamlets"]:
                    break

        r_ids = [calculate_single_beamlet.remote([b],opt) for b in bb]

        r_finished, r_waiting = ray.wait(r_ids, 1)
        while r_waiting:            
            for r in r_finished:
                info(f"Posprocesing...")
                self.postprocess(ray.get(r))
            info("Waiting for results...")
            r_finished, r_waiting = ray.wait(r_waiting, timeout=2.0)
            info(f"Got {len(r_finished)} results to posprocess. Still waiting for {len(r_waiting)}...")

        for r in r_finished:
            info(f"Posprocesing...")
            self.postprocess(ray.get(r))

        info("Min total dose = %f, Max total dose = %f" % (numpy.min(self.total_doses), numpy.max(self.total_doses)))
        saveToVTI(self.rass_data.output("beamlet_total_for_beam_%d" % self.conf_data["beam_number"]), self.total_doses, self.spacing, self.n, self.orig)

        os.chdir(oldWorkingDirectory)
        return self.beamlets_doses.copy()


    def getApproximatedCT(self, config_file=None, v2Drow=None, voxels=None, options=None, ctfiles=None):
            self.ctfiles = ctfiles
            self.check_config(config_file)

            """
            # ustawiam się w katalogu tego skryptu bo pliki są ustalone względem tej ścieżki
            oldWorkingDirectory = os.getcwd()
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            """
            oldWorkingDirectory = os.getcwd()
            os.chdir(self.vmc_home)

            os.chdir(oldWorkingDirectory)
            return self.prepare_ct_file(v2Drow)

    #############################
    # noinspection PyTypeChecker
    def prepare_ct_file(self, v2Drow):

        info("Reading CT Dicom data from folder: %s" % self.dicom_folder)
        # wczytuję oryginalny CT z DICOMa
        reader = CTVolumeDataReader(self.dicom_folder, ctfiles=self.ctfiles)
        ctVolumeData = reader.read()
        ctVolumeData.save(self.rass_data.output("phantom_ct"))  # zapisuję do vti do debugingu
        bDimFromCT = True if self.plan_n is None else False

        self.n = self.plan_n if self.plan_n is not None else ctVolumeData.dimensions
        self.orig = self.plan_origin if self.plan_origin is not None else ctVolumeData.origin
        self.spacing = self.plan_spacing if self.plan_spacing is not None else ctVolumeData.spacing

        bnx = struct.pack("i", self.n[0])
        bny = struct.pack("i", self.n[1])
        bnz = struct.pack("i", self.n[2])

        fout = open("%s" % (self.ct_file), "wb")
        fout.write(bnx)
        fout.write(bny)
        fout.write(bnz)

        dcorr = [-0.5, -0.5, -0.5]
        for d in range(3):
            #print "orig: %f" % self.orig[d]
            for i in range(0, self.n[d]+1):
                b = struct.pack("f", (self.orig[d] + self.spacing[d] * i + dcorr[d] * self.spacing[d]))
                #print "[%d,%d]=%f" % (d, i, self.orig[d] + self.spacing[d] * i + dcorr[d] * self.spacing[d])
                fout.write(b)

        if bDimFromCT:
            npar = ctVolumeData.getCTDataAsNumpyArray()
        else:
            npar = self.approximateCTOnPlanGrid(ctVolumeData)


        ###################################################################################
        # Skalowanie wartości Hounsfielda (-1000-6000) do zakresu (0-4) - używanego przez VMC
        # Skalowanie dwuetapowe:
        # 1. Najpierw z HU do Rel.Electron Ddensity - na podstawie danych Eclipse
        # 2. Z Rel.Electron Ddensity do gęstości masy - na podstawie artykułu:
        #    Kanematsu, N., Inaniwa, T., & Koba, Y. (2012). Relationship between electron density
        #    and effective densities of body tissues for stopping, scattering, and nuclear
        #    interactions of proton and ion beams. Medical Physics, 39(2), 1016–20. http://doi.org/10.1118/1.3679339
        ###################################################################################

# Pierwszy etap
#CT_Calibration Curve: Def_CTScanner Electron Density
#
#HU Value [HU]         Rel.Density
#
#-1050.000		0.000
#-1000.000		0.000    a = 0.001, b = 1
#  100.000		1.100    a = 4.800e-4, b = 1.0520
# 1000.000		1.532    a = 4.7760e-04, b = 1.0544
# 6000.000		3.920
        npar -= 1000
        npar[npar < -1000] = 0
        cond1 = (npar > -1000) & (npar <= 100)
        cond2 = (npar > 100) & (npar <= 1000)
        cond3 = npar > 1000
        npar[cond1] *= 0.001
        npar[cond1] += 1
        npar[cond2] *= 4.800e-4
        npar[cond2] += 1.0520
        npar[cond3] *= 4.7760e-04
        npar[cond3] += 1.0544
        npar[npar > 3.92] = 3.920

# Drugi etap:
# Relative Electron Density   Mass density
# 0                               0              a=0.98901, b=0
# 0.91                            0.9            a=1.1538, b=-0.15000
# 2.73                            3
        cond1 = npar < 0.9
        cond2 = npar >= 0.9
        npar[cond1] *= 0.98901
        npar[cond2] *= 1.1538
        npar[cond2] += -0.15000


        if (self.water_phantom):
            warning("Warning! Applying WATER PHANTOM transformation to CT data. All voxel will have scaled Hounsfield number equal to 1.")
            npar[:] = 1

        if (v2Drow is not None):
            info("Zeruję gęstość masy dla wszystkich 'nienteresujących voxeli' dla %d voxeli" % numpy.sum(v2Drow < 1))
            npar[v2Drow < 1] = 0


        # zapisuję cały wektor do pliku binarnie
        npar.tofile(fout)
        fout.close()

        # Zapisywanie danych do formatu vti
        info("Zapisuję przeskalowane dane gęstości masy z CT do pliku: %s" % self.rass_data.output("approximated_ct"))
        volume = VolumeData()
        grid = volume.createGrid((self.spacing[0], self.spacing[1], self.spacing[2]), (self.n[0], self.n[1], self.n[2]), (self.orig[0], self.orig[1], self.orig[2]))
        array = volume.createFloatArray(grid)
        array.SetVoidArray(npar, numpy.prod(npar.shape), 1)
        # for i in range(np.size(dens, 0)):
        #    array.SetValue(i, dens[i])
        grid.GetPointData().SetScalars(array)
        volume = VolumeData(grid)
        volume.save(self.rass_data.output("approximated_ct"))
        info("Zapisałem przeskalowane dane gęstości masy z CT do pliku: %s" % self.rass_data.output("approximated_ct"))

        return npar[v2Drow > 0]

    def read_config(self, config_file):
        with open(config_file) as conf_file:
            self.conf_data = json.load(conf_file)

        conf = self.conf_data
        self.dicom_folder = conf['dicom_folder']
        if 'vmc_ct_file' in conf:
            self.ct_file = conf['vmc_ct_file'] if conf['vmc_ct_file'].startswith("/") else "%s/%s" % (conf['vmc_home'], conf['vmc_ct_file'])
        else:
            self.ct_file = "phantoms/phantom.ct"

        if 'spectrum_file' in conf:
            self.spectrum_filename = conf['spectrum_file'] if conf['spectrum_file'].startswith("/") else "%s/%s" % (conf['vmc_home'], conf['spectrum_file'])
        else:
            self.spectrum_filename = "%s/spectra/var_CL2300_5_X6MV.spectrum" % conf['vmc_home']

        self.vmc_home = conf['vmc_home'] \
            if 'vmc_home' in conf and conf['vmc_home'] is not None \
            else self.rass_data.processing("vmc++")
        self.xvmc_dir = self.vmc_home
        self.vmc_runs = conf['vmc_runs'] if 'vmc_runs' in conf else "runs"
        self.geometric_scale = conf['geometric_scale'] if 'geometric_scale' in conf else 0.1
        if 'plan_grid' in conf:
            pg = conf['plan_grid']
            self.plan_n = [int(pg['size']['x']), int(pg['size']['y']), int(pg['size']['z'])]
            self.plan_origin = [float(pg['orig']['x']), float(pg['orig']['y']), float(pg['orig']['z'])]
            self.plan_spacing = [float(pg['spacing']['dx']), float(pg['spacing']['dy']), float(pg['spacing']['dz'])]
        else:
            error("Brak informacji o siatce planowania w pliku ze specyfikacją. Użyję do obliczeń siatki CT.")
        if "water_phantom" in conf:
            self.water_phantom = bool(conf["water_phantom"])

        self.beam_no = conf['beam_number'] if 'beam_number' in conf else 0
        self.ncase = conf['ncase'] if 'ncase' in conf else "50000"
        self.nbatch = conf['nbatch'] if 'nbatch' in conf else "10"
        self.delete_doses_file_after = conf['delete_doses_file_after'] if 'delete_doses_file_after' in conf else True
        self.delete_vmc_file_after = conf['delete_vmc_file_after'] if 'delete_vmc_file_after' in conf else True
        self.doses_dos_path = conf["doses_dos_path"] if "doses_dos_path" in conf and conf["doses_dos_path"] is not None else self.rass_data.processing()
        self.cluster_config_file = conf["cluster_config_file"] if "cluster_config_file" in conf and conf["cluster_config_file"] is not None else None

        print("beam_no = %d" % self.beam_no)

    def approximateCTOnPlanGrid(self, ctVolumeData):
        start = time.time()
        o = self.orig
        s = self.spacing
        scale = self.geometric_scale
        n = self.n

        error("Approximating CT grid over Planning Grid (%d x %d x %d) ..." % (n[0], n[1], n[2]))
        npar = ctVolumeData.getCTDataAsNumpyArray()

        N = numpy.array([numpy.linspace(0, n[0] - 1, n[0]), numpy.linspace(0, n[1] - 1, n[1]), numpy.linspace(0, n[2] - 1, n[2])])
        PX = numpy.array(o[0] + N[0] * s[0] + 0.5 * s[0])
        PY = numpy.array(o[1] + N[1] * s[1] + 0.5 * s[1])
        PZ = numpy.array(o[2] + N[2] * s[2])

        CIX = numpy.array((PX / scale - ctVolumeData.origin[0]) / ctVolumeData.spacing[0], dtype=numpy.int_)
        CIY = ctVolumeData.dimensions[1] - numpy.array((PY / scale - ctVolumeData.origin[1]) / ctVolumeData.spacing[1], dtype=numpy.int_)
        CIZ = numpy.array((PZ / scale - ctVolumeData.origin[2]) / ctVolumeData.spacing[2], dtype=numpy.int_)

        (MCIX, MCIY, MCIZ) = numpy.meshgrid(CIX, CIY, CIZ, indexing='ij')
        MCIX = MCIX.flatten()
        MCIY = MCIY.flatten()
        MCIZ = MCIZ.flatten()
        MCI = MCIX + MCIY * ctVolumeData.dimensions[0] + MCIZ * ctVolumeData.dimensions[0] * ctVolumeData.dimensions[1]

        H = numpy.zeros(numpy.prod(n))
        bMCI = MCI < len(npar)
        H[bMCI] = npar[MCI[bMCI]]

        (MIX, MIY, MIZ) = numpy.meshgrid(N[0], N[1], N[2], indexing='ij')
        MIX = MIX.flatten()
        MIY = MIY.flatten()
        MIZ = MIZ.flatten()
        MI = numpy.array(MIX + MIY * n[0] + MIZ * n[0] * n[1], dtype=numpy.int_)

        r = numpy.zeros(numpy.prod(n), dtype=numpy.float32)
        r[MI] = H

        #############################################################################################################
        # Poniżej stary kod aproksymujący dane CT nad siatką planowania.
        #############################################################################################################
        # oldr = numpy.zeros(numpy.prod(n))
        # getIndexCoordinates = ctVolumeData.getIndexCoordinatesOfPoint
        # getIndex = ctVolumeData.getIndex
        # ctVtkArray = ctVolumeData.data.GetPointData().GetScalars()
        # GetTuple = ctVtkArray.GetTuple

        # OLDI = numpy.zeros(numpy.prod(n))
        # for ix in range(n[0]):
        #     px = o[0] + ix * s[0] + 0.5 * s[0]
        #     for iy in range(n[1]):
        #         py = o[1] + iy * s[1] + 0.5 * s[1]
        #         for iz in range(n[2]):
        #             pz = o[2] + iz * s[2]
        #
        #             (cix, ciy, ciz) = getIndexCoordinates(px, py, pz, scale)
        #
        #             ci = getIndex(cix, ciy, ciz)
        #             h = GetTuple(ci)
        #
        #             i = ix + iy * n[0] + iz * n[0] * n[1]
        #             oldr[i] = int(h[0])
        #
        # print "Difference r == oldr: %s" % (oldr == r)
        # print "Number of different: %d" % (numpy.sum(oldr != r))

        elapsed = time.time() - start
        info("Finished aproximating CT grid over Planning Grid in %s seconds" % elapsed)
        return r

    def check_config(self, config_file):
        if config_file is not None:
            self.read_config(config_file)
        if self.conf_data is None:
            raise Exception("Proszę najpierw skonfigurować klasę numpy. za pomocą metody: read_config(). "
                            "Alternatywnie można podać argument config_file.")


    def opt_for_job(self):
        return {
                "spectrum_filename": self.spectrum_filename,
                "vmc_runs": self.vmc_runs,
                "ct_file": self.ct_file,
                "xvmc_dir": self.xvmc_dir,
                "vmc_home": self.vmc_home,
                "delete_doses_file_after": self.delete_doses_file_after,
                "delete_vmc_file_after": self.delete_vmc_file_after,
                "ncpu": self.ncpu,
                "ncase": self.ncase,
                "nbatch": self.nbatch,
                "processing": self.doses_dos_path,
                "cluster_config_file": self.cluster_config_file,
                "beam_no": self.beam_no,
                DEBUG_MAX_BEAMLETS_KEY: self.debug_max_beamlets,
                "dose_tolerance_min": self.dose_tolerance_min
        }


if __name__ == '__main__':
    configfile = "runs/vmc.example.json" if len(sys.argv) < 2 else sys.argv[1]
    vmc = VMC()
    vmc.run(configfile)


