import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import logging as log

from bdfileutils import read_ndarray
from rass import RASSData
from PIL import Image

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
            
            ########################################################
            # Uruchamiam obliczenia w folderze folder (obiekt rd)
            ########################################################
            args4bd4cnn = lambda: None
            args4bd4cnn.rass_data = rd
            bd4cnn.do_run(args4bd4cnn)

    ##############################################################
    # Analizuję wymiary (dimensions) - w tym celu wczytuję dawki
    ##############################################################
    adims = []
    for sub in subs:
        rd = RASSData(root_folder=os.path.join(root_folder, sub))
        doses = read_ndarray(rd.output("total_doses.nparray"))
        adims.append(doses.shape)

    nadims = np.array(adims)
    
    #minz = np.min(nadims[:,0])
    #miny = np.min(nadims[:,1])
    #minx = np.min(nadims[:,2])
    #final_shape_min = (minz, miny, minx)
    final_shape_min=np.min(nadims, axis=0)
    log.info(f"final shape min: {final_shape_min}")

    #maxz = np.max(nadims[:,0])
    #maxy = np.max(nadims[:,1])
    #maxx = np.max(nadims[:,2])
    #final_shape_max = (maxz, maxy, maxx)
    final_shape_max=np.max(nadims, axis=0) # [z,y,x]
    log.info(f"final shape max: {final_shape_max}")


    ##############################################################
    # Przycinam lub rozszerzam obszary ROIów
    ##############################################################
    for sub in subs:
        folder = os.path.join(root_folder, sub)
        rd = RASSData(root_folder=folder)

        doses = read_ndarray(rd.output("total_doses.nparray"))

        doses = np.round(doses/np.max(doses) * args.dose_levels)

        roi_marks = read_ndarray(rd.output("roi_marks.nparray"), dtype=np.int64)
        ct = read_ndarray(rd.output("approximated_ct.nparray"))
        
        # wczytuję outline aby zbudować boundingbox, którego ostatecznie nie uzywam, ale łądnie wygląda
        marks_patient_outline = read_ndarray(rd.output(f"roi_marks_Patient Outline.nparray"), dtype=np.int32)

        # środkowy slice
        middle_z_idx = marks_patient_outline.shape[0] // 2
        ref_slice = marks_patient_outline[middle_z_idx, :,:]

        # zbiór współrzędnych pikseli, która maja wartość równą 1
        idx = np.where( ref_slice == 1 )
        bbox = ( (min(idx[0]), min(idx[1])) , (max(idx[0]), max(idx[1])) ) 
        m =  ( (bbox[0][0] + bbox[1][0]) // 2,
               (bbox[0][1] + bbox[1][1]) // 2 )
        log.info(f"bbox of Patient Outline middle z slice ={bbox}")
        
        # zaznaczam na referencyjnym przekroju krzyżyk środka bounding boxa
        ref_slice[m[0],:] = 2
        ref_slice[:,m[1]] = 2

        # zaznaczam na referencyjnym przekroju bounding boxa
        ref_slice[bbox[0][0],:] = 3
        ref_slice[bbox[1][0],:] = 3
        ref_slice[:,bbox[0][1]] = 4
        ref_slice[:, bbox[1][1]] = 4

        # zapisuję referencyjny obrazek do głównego folderu symulacji
        plt.imsave((f"{root_folder}/ref_slice_{sub}.png"), ref_slice)

        ################################################################################################
        # Określam rozmiary dla każdego obrazka jak obcinać zgodnie z algorytmem
        # dopasowania do najmniejszego obrazka - robimy obcięcie od góry po y i symetrycznie po x-ach
        ################################################################################################
        ref_x_m = ref_slice.shape[1]//2
        final_x = final_shape_min[2]
        xfrom = ref_x_m - final_x//2
        xto = ref_x_m + final_x//2

        # tutaj obcinam początkowe yki, gdy ref_slice jest za duży
        yfrom = ref_slice.shape[0] - final_shape_min[1] 
        # tutaj jadę do końca do dołu obrazka
        yto = ref_slice.shape[0] 

        log.debug(f"[{sub}] min_xfrom: {xfrom}")
        log.debug(f"[{sub}] min_xto: {xto}")
        log.debug(f"[{sub}] min_yfrom: {yfrom}")
        log.debug(f"[{sub}] min_yto: {yto}")

        ref_slice_cropped_to_min = ref_slice[ yfrom:yto, xfrom:xto]
        plt.imsave((f"{root_folder}/ref_slice_{sub}_cropped_to_min.png"), ref_slice_cropped_to_min)


        ################################################################################################
        # Określam rozmiary dla każdego obrazka jak obcinać zgodnie z algorytmem
        # dopasowania do NAJWIĘKSZEGO obrazka - robimy doklejenie pustych pikseli od góry po y 
        # i symetrycznie po x-ach
        ################################################################################################
        final_x = final_shape_max[2]
        ref_x_m = ref_slice.shape[1]//2
        xfrom = final_x//2 - ref_x_m
        yfrom = final_shape_max[1] - ref_slice.shape[0]
        print(xfrom)
        print(yfrom)
        ref_slice_cropped_to_max = np.zeros( (final_shape_max[1], final_shape_max[2]))

        ref_slice_cropped_to_max[ yfrom:yfrom+ref_slice.shape[0],xfrom:xfrom+ref_slice.shape[1]] = ref_slice
        plt.imsave((f"{root_folder}/ref_slice_{sub}_cropped_to_max.png"), ref_slice_cropped_to_max)


        if (rd.input_exists("roi_mapping.json")):
            # w obecnej wersji uzywam max
            m = {}
            with open(rd.input("roi_mapping.json")) as fin:
                m.update(json.load(fin))

            lst = [(rvalue, rname) for rname, rvalue in m.items()]
            lst = sorted(lst, key=lambda i: i[0])

            # zetów jest tyle co w roi_marks.shape[0], natomiast rozmiary x i y są z final_shape_max
            roi_marks_mapped_full = np.zeros( (roi_marks.shape[0], final_shape_max[1], final_shape_max[2]) )
                
            for (rvalue, rname) in lst:
                marks = read_ndarray(rd.output(f"roi_marks_{rname}.nparray"), dtype=np.int32) # mniejszy
                b = (marks == 1) # gdzie wstawic ta wartosc?

                tmp = roi_marks_mapped_full[:, yfrom:yfrom+ref_slice.shape[0], xfrom:xfrom+ref_slice.shape[1]]
                tmp[b] = rvalue
                roi_marks_mapped_full[:, yfrom:yfrom+ref_slice.shape[0],xfrom:xfrom+ref_slice.shape[1]] = tmp

            for i in range(roi_marks_mapped_full.shape[0]):
                plt.imsave(rd.output(f"roi_marks_mapped_{i}.png", "roi_mapped_to_max"), roi_marks_mapped_full[i,:,:])
                pil_im = Image.fromarray(roi_marks_mapped_full[i,:,:].astype(np.uint32))
                pil_im.save(rd.output(f"pil_im_{i}.png", "roi_mapped_to_max"))


            doses_full = np.zeros( (doses.shape[0], final_shape_max[1], final_shape_max[2]) )
            doses_full[:, yfrom:yfrom+ref_slice.shape[0],xfrom:xfrom+ref_slice.shape[1]] = doses
            for i in range(doses_full.shape[0]):
                plt.imsave(rd.output(f"doses_{i}.png", "doses_to_max"), doses_full[i,:,:])
                pil_im = Image.fromarray(doses_full[i,:,:].astype(np.uint32))
                pil_im.save(rd.output(f"pil_im_{i}.png", "doses_to_max"))


            ct_full = np.zeros( (doses.shape[0], final_shape_max[1], final_shape_max[2]) )
            ct_full[:, yfrom:yfrom+ref_slice.shape[0],xfrom:xfrom+ref_slice.shape[1]] = ct
            for i in range(ct_full.shape[0]):
                plt.imsave(rd.output(f"doses_{i}.png", "ct_to_max"), ct_full[i,:,:])
                pil_im = Image.fromarray(ct_full[i,:,:].astype(np.uint32))
                pil_im.save(rd.output(f"pil_im_{i}.png", "ct_to_max"))

        else:
            log.warn(f"Pomijam katalog {sub}, ponieważ w katalogu input brakuje pliku `roi_mapping.json`")


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Przechodzi przez podfoldery i przetwarza dane przypadków do postaci, która może być wykorzystana w uczeniu sieci neuronowych (CNN)")
    parser.add_argument('root_folder', help="główny folder, który zawiera analizowane podfoldery")
    parser.add_argument('--dose_levels', type=float, help="liczba poziomów dawek w wyjściowych plikach png", default=255)
    args = parser.parse_args()

    do_run(args)