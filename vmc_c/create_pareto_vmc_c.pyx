#import numpy as np
cimport numpy as np
#def myfunc(np.ndarray[np.float64_t, ndim=2] A):

def mark_voxels(np.ndarray[np.float32_t, ndim=3] voxels, np.ndarray[np.int_t, ndim=1] v2Drow, int kmax, int jmax, int imax):
    cdef Py_ssize_t kx, jx, ix, voxel
    for kx in range(kmax):
        for jx in range(jmax):
            for ix in range(imax):
                voxel = kx + jx * kmax + ix * (kmax*jmax)
                if v2Drow[voxel] >= 0:
                    voxels[ix][jx][kx] = 1


def postprocess_fluence_for_scaling(np.ndarray[np.float32_t, ndim=2] beamlet_doses, float fluence, np.ndarray[np.float32_t, ndim=3] mcDosesForScaling, int kmax, int jmax, int imax):
    cdef Py_ssize_t row, kx, jx, ix, voxel
    for row in range(beamlet_doses.shape[0]):
        mcDoseFluence = beamlet_doses[row, 1] * fluence
        voxel = int( beamlet_doses[row, 0] )
        ix = voxel / (kmax * jmax)	                # x-index
        jx = (voxel - ix * (kmax * jmax)) / kmax	# y-index
        kx = voxel % kmax 			                # z-index
        mcDosesForScaling[ix][jx][kx] += mcDoseFluence


def postprocess_fluence(float monteCarloScalingCoefficient, np.ndarray[np.float32_t, ndim=2] beamlet_doses,
                        float fluence, np.ndarray[np.float32_t, ndim=3] mcDoses,
                        np.ndarray[np.float32_t, ndim=3] mcDosesFluence,
                        int kmax, int jmax, int imax, bint saveToFile,
                        f, np.ndarray[np.int_t, ndim=1] v2Drow, bt_file_idx):
    cdef Py_ssize_t row, kx, jx, ix, voxel
    for row in range(beamlet_doses.shape[0]):
        dose = beamlet_doses[row, 1] * monteCarloScalingCoefficient
        voxel = int(beamlet_doses[row, 0])
        ix = voxel / (kmax * jmax)	                # x-index
        jx = (voxel - ix * (kmax * jmax)) / kmax	# y-index
        kx = voxel % kmax 			                # z-index
        mcDoses[ix][jx][kx] += dose
        mcDosesFluence[ix][jx][kx] += dose * fluence

        if saveToFile:
            f.write('%d %d %d\n' % (v2Drow[voxel], bt_file_idx, dose))

def postprocess_fluence_individual(np.ndarray[np.float32_t, ndim=3] scaling_factors, np.ndarray[np.float32_t, ndim=2] beamlet_doses,
                        float fluence, np.ndarray[np.float32_t, ndim=3] mcDoses,
                        np.ndarray[np.float32_t, ndim=3] mcDosesFluence,
                        int kmax, int jmax, int imax, bint saveToFile,
                        f, np.ndarray[np.int_t, ndim=1] v2Drow, bt_file_idx):
    cdef Py_ssize_t row, kx, jx, ix, voxel
    for row in range(beamlet_doses.shape[0]):
        voxel = int(beamlet_doses[row, 0])
        ix = voxel / (kmax * jmax)	                # x-index
        jx = (voxel - ix * (kmax * jmax)) / kmax	# y-index
        kx = voxel % kmax 			                # z-index
        s = scaling_factors[ix][jx][kx]
        dose = beamlet_doses[row, 1] * s
        mcDoses[ix][jx][kx] += dose
        mcDosesFluence[ix][jx][kx] += dose * fluence

        if saveToFile:
            f.write('%d %d %d\n' % (v2Drow[voxel], bt_file_idx, dose))
