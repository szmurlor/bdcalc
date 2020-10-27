import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import logging as log

from bdfileutils import read_ndarray
from rass import RASSData
import bd4cnn

log.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=log.INFO)

required_files = [
    'approximated_ct.nparray',
    'total_doses.nparray',
    'roi_marks.nparray',
    'roi_mapping.txt'
]

def do_run(args):
    root_folder=args.root_folder

    subs = next(os.walk(root_folder))[1]
    for sub in subs:
        folder = os.path.join(root_folder, sub)
        rd = RASSData(root_folder=folder)

        files_exist = True
        for rf in required_files:
            if not rd.output_exists(rf):
                log.info(f"Missing file {rd.output(rf)}")
                files_exist = False
        
        if not files_exist:
            log.info(f"Brakuje plikow w folderze: {folder}")
            
            args4bd4cnn = lambda: None
            args4bd4cnn.rass_data = rd
            bd4cnn.do_run(args4bd4cnn)

    adims = []
    for sub in subs:
        folder = os.path.join(root_folder, sub)
        rd = RASSData(root_folder=folder)
        doses = read_ndarray(rd.output("total_doses.nparray"))
        adims.append(doses.shape)

    nadims = np.array(adims)
    minz = np.min(nadims[:,0])
    miny = np.min(nadims[:,1])
    minx = np.min(nadims[:,2])

    final_shape = (minz, miny, minx)

    print(f"final shape: {final_shape}")

    for sub in subs:
        folder = os.path.join(root_folder, sub)
        rd = RASSData(root_folder=folder)

        doses = read_ndarray(rd.output("total_doses.nparray"))
        roi_marks = read_ndarray(rd.output("roi_marks.nparray"), dtype=np.int64)
        ct = read_ndarray(rd.output("approximated_ct.nparray"))
        
        marks_patient_outline = read_ndarray(rd.output(f"roi_marks_Patient Outline.nparray"), dtype=np.int32)
        print(marks_patient_outline.shape)

        middle = marks_patient_outline.shape[0] // 2
        ref_slice = marks_patient_outline[middle, :,:]
        print(ref_slice.shape)


        idx = np.where( ref_slice == 1 )
        print(idx)
        bbox = ( (min(idx[0]), min(idx[1])) , (max(idx[0]), max(idx[1])) ) 
        m =  ( (bbox[0][0] + bbox[1][0]) // 2,
               (bbox[0][1] + bbox[1][1]) // 2 )
        print(f"bbox={bbox}")
        print(m)
        
        ref_slice[m[0],:] = 2
        ref_slice[:,m[1]] = 2
        ref_slice[bbox[0][0],:] = 3
        ref_slice[bbox[1][0],:] = 3
        ref_slice[:,bbox[0][1]] = 4
        ref_slice[:, bbox[1][1]] = 4
        ref_slice[:,m[1]] = 2
        plt.imsave((f"{root_folder}/ref_slice_{sub}.png"), ref_slice)

        xfrom = ref_slice.shape[1]//2-final_shape[2]//2
        xto = ref_slice.shape[1]//2+final_shape[2]//2
        yfrom = ref_slice.shape[0]-final_shape[1]
        yto = ref_slice.shape[0]
        print(xfrom)
        print(xto)
        print(yfrom)
        print(yto)
        ref_slice_cropped = ref_slice[ yfrom:yto, xfrom:xto]
        plt.imsave((f"{root_folder}/ref_slice_{sub}_cropped.png"), ref_slice_cropped)

    return

    # ct[z,y,x]
    ct = read_ndarray(rass_data.output("approximated_ct.nparray"))
    for i in range(ct.shape[0]):
        plt.imsave(rass_data.output(f"ct_{i}.png"), ct[i,:,:])

    # doses[z,y,x]
    doses = read_ndarray(rass_data.output("total_doses.nparray"))
    for i in range(doses.shape[0]):
        plt.imsave(rass_data.output(f"doses_{i}.png"), doses[i,:,:])

    # roi_marks[z,y,x]
    roi_marks = read_ndarray(rass_data.output("roi_marks.nparray"), dtype=np.int64)
    for i in range(roi_marks.shape[0]):
        plt.imsave(rass_data.output(f"roi_marks_{i}.png"), roi_marks[i,:,:])


    # ponizej, jezeli w katalogu */input znajduje się plik o nazwie 'roi_mapping.json',
    # przykładowa struktura tego pliku:
    # {
    #    "ptv-plan": 6,
    #    "kanal kreg.": 5,
    #    "serce": 4,
    #    "pluco P": 3,
    #    "pluco L": 2,
    #    "Patient Outline": 1
    #}
    #
    # najważniejsze są roie o najwyższych numerach. Czyli jak woksel należy do do kilku roi-ów,
    # to w zmapowanym obrazie zostanie dla niego przypisany najwyższy numer - zazwyczaj PTV jest najważniejszy.

    roi_marks_mapped = np.zeros(roi_marks.shape, dtype=np.int32)
    if os.path.isfile(rass_data.input("roi_mapping.json", check=False)):
        m = {}
        with open(rass_data.input("roi_mapping.json")) as fin:
            m.update(json.load(fin))

        lst = [(rvalue, rname) for rname, rvalue in m.items()]
        lst = sorted(lst, key=lambda i: i[0])

        
        for rvalue, rname in lst:
            marks = read_ndarray(rass_data.output(f"roi_marks_{rname}.nparray"), dtype=np.int32)
            b = (marks == 1)
            roi_marks_mapped[b] = rvalue

        for i in range(roi_marks_mapped.shape[0]):
            plt.imsave(rass_data.output(f"roi_marks_mapped_{i}.png"), roi_marks_mapped[i,:,:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Przechodzi przez podfoldery i przetwarza dane przypadków do postaci, która może być wykorzystana w uczeniu sieci neuronowych (CNN)")
    parser.add_argument('root_folder', help="główny folder, który zawiera analizowane podfoldery")
    args = parser.parse_args()

    do_run(args)