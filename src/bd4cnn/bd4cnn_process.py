import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import logging as log
import pathlib

from bdfileutils import read_ndarray, save_ndarray
from rass import RASSData
from PIL import Image
from matplotlib import cm

import bd4cnn

log.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=log.INFO)

required_files = [
    'approximated_ct.nparray',
    'total_doses.nparray',
    'roi_marks.nparray',
    'roi_mapping.txt'
]

def do_run(args):
    root_folder = args.root_folder

    if hasattr(args,'single') and args.single:
        print(r"Single folder: {args.root_folder}")
        subs = ["."]
    else:
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
            log.info(f"Brakuje wymaganych plików w folderze: {folder}. Uruchamiam obliczenia.")
            
            ########################################################
            # Uruchamiam obliczenia w folderze folder (obiekt rd)
            ########################################################
            args4bd4cnn = lambda: None
            args4bd4cnn.rass_data = rd
            bd4cnn.do_run(args4bd4cnn)

            log.info(f"Zakończyłem obliczenia w folderze: {folder}")

    ####################################################################################
    # Analizuję wymiary w wszystkich folderach (dimensions) - w tym celu wczytuję dawki
    # Dodatkowo ciągam maksymalną dawk globalnie!
    ####################################################################################
    log.info("Finding maximum dimensions and dose value.")
    adims = []
    max_dose_global = 0
    for sub in subs:
        rd = RASSData(root_folder=os.path.join(root_folder, sub))
        doses = read_ndarray(rd.output("total_doses.nparray"))
        m = np.max(doses)
        if max_dose_global < m:
            max_dose_global = m
            log.info(f"Updating max dose: {m}")
        adims.append(doses.shape)
    log.info(f"Final, total max dose: {max_dose_global}")

    nadims = np.array(adims)
    
    final_shape_min=np.min(nadims, axis=0)
    log.info(f"W przypadku dopasowania do najmniejszego rozmiaru: final shape min: {final_shape_min}")

    final_shape_max=np.max(nadims, axis=0) # [z,y,x]
    log.info(f"W przypadku dopasowania do największego rozmiaru: final shape max: {final_shape_max}")

    ################################################################
    # Generuje dane z przycinamiem lub rozszerzaniem obszarów ROIów
    ################################################################
    for sub in subs:
        folder = os.path.join(root_folder, sub)
        rd = RASSData(root_folder=folder)

        meta_data = {}
        if os.path.isfile(rd.input("meta_data.json")):
            with open(rd.input("meta_data.json")) as fin:
                meta_data.update(json.load(fin))
        if os.path.isfile(rd.root("meta.json")):
            with open(rd.root("meta.json")) as fin:
                meta_data.update(json.load(fin))

        meta_processing = {}
        meta_processing['cmd'] = " ".join(sys.argv)

        patient_id = meta_data["patient_id"] if "patient_id" in meta_data else sub
        meta_processing['patient_id'] = patient_id

        log.info(f"Analizuję pacjenta: {patient_id}")

        log.info("Wczytuję dawki")
        meta_processing['doses_source'] =  rd.output("total_doses.nparray")
        doses = read_ndarray(meta_processing['doses_source'])

        ########################################################################
        # Wykonuję progowanie dawek (czyli wartości przewidywanych)
        # Normalizacja dawki do wartości maksymalnej dawki dla danego pacjenta
        ########################################################################
        log.info(f"Proguję dawki do {args.dose_levels} poziomów względem wartości maksymalnej: {max_dose_global}")
        meta_processing['dose_levels'] = int(args.dose_levels)
        meta_processing['max_dose_global'] = float(max_dose_global)
        meta_processing['max_dose'] = float(np.max(doses))
        meta_processing['dose_levels_scale'] =  float(args.dose_levels / max_dose_global)
        meta_processing['dose_levels_upscale'] =  float(max_dose_global / args.dose_levels)

        doses = np.round(doses/max_dose_global * args.dose_levels)

        log.info("Wczytuję informację o znacznikach ROI.")
        meta_processing['rois_source'] =  rd.output("roi_marks.nparray")
        roi_marks = read_ndarray(meta_processing['rois_source'], dtype=np.int64)


        log.info("Wczytuję informację o danych CT.")
        meta_processing['ct_source'] =  rd.output("approximated_ct.nparray")
        ct = read_ndarray(rd.output("approximated_ct.nparray"))
        
        # wczytuję outline aby zbudować boundingbox, którego ostatecznie nie uzywam, ale ładnie wygląda
        log.info("Wczytuję Patient Outline do ładnych wizualizacji.")
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
        plt.imsave((f"{root_folder}/ref_slice_{patient_id}.png"), ref_slice)

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

        log.debug(f"[{patient_id}] min_xfrom: {xfrom}")
        log.debug(f"[{patient_id}] min_xto: {xto}")
        log.debug(f"[{patient_id}] min_yfrom: {yfrom}")
        log.debug(f"[{patient_id}] min_yto: {yto}")

        ref_slice_cropped_to_min = ref_slice[ yfrom:yto, xfrom:xto]
        plt.imsave((f"{root_folder}/ref_slice_{patient_id}_cropped_to_min.png"), ref_slice_cropped_to_min)


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

        ref_slice_cropped_to_max[ yfrom:yfrom+ref_slice.shape[0],xfrom:xfrom+ref_slice.shape[1] ] = ref_slice
        plt.imsave((f"{root_folder}/ref_slice_{patient_id}_cropped_to_max.png"), ref_slice_cropped_to_max)

        meta_processing['shape_original'] = ", ".join(map(str,roi_marks.shape))

        mapping_file = rd.input("roi_mapping.json")
        if not os.path.exists(mapping_file):
            mapping_file = rd.root("roi_mapping.json")

        if (os.path.exists(mapping_file)):
            # w obecnej wersji uzywam max
            m = {}
            with open(mapping_file) as fin:
                m.update(json.load(fin))

            lst = [(rvalue, rname) for rname, rvalue in m.items()]
            lst = sorted(lst, key=lambda i: i[0])

            # zetów jest tyle co w roi_marks.shape[0], natomiast rozmiary x i y są z final_shape_max
            roi_marks_mapped_full = np.zeros( (roi_marks.shape[0], final_shape_max[1], final_shape_max[2]) )
            roi_marks_original_full = np.zeros( (roi_marks.shape[0], final_shape_max[1], final_shape_max[2]) )

            meta_processing['shape_final'] = ", ".join(map(str,roi_marks_mapped_full.shape))                
            meta_processing['shape_final_yfrom'] = int(yfrom)
            meta_processing['shape_final_xfrom'] = int(xfrom)
            meta_processing['shape_final_yto'] = int(yfrom+ref_slice.shape[0])
            meta_processing['shape_final_xto'] = int(xfrom+ref_slice.shape[1])

            for (rvalue, rname) in lst:
                marks = read_ndarray(rd.output(f"roi_marks_{rname}.nparray"), dtype=np.int32) # mniejszy
                b = (marks == 1) # gdzie wstawic przyporzadkowana w mapowaniu wartosc rvalue?

                # rozmiar marks jest taki sam jak tmp
                tmp = roi_marks_mapped_full[:, yfrom:yfrom+ref_slice.shape[0], xfrom:xfrom+ref_slice.shape[1]]
                tmp[b] = rvalue
                roi_marks_mapped_full[:, yfrom:yfrom+ref_slice.shape[0],xfrom:xfrom+ref_slice.shape[1]] = tmp

            roi_marks_original_full[:, yfrom:yfrom+ref_slice.shape[0],xfrom:xfrom+ref_slice.shape[1]] = roi_marks

            log.info(f"Rozpoczynam zapisywanie {roi_marks_mapped_full.shape[0]} plików z obrazami ROI (zmapowane) w formacie uint8")
            for i in range(roi_marks_mapped_full.shape[0]):
                if hasattr(args, "savepng") and args.savepng:
                    plt.imsave(rd.root_path(args.cnn_output, "roi_mapped_to_max_png", fname=f"roi_marks_mapped_{patient_id}_{i}.png"), roi_marks_mapped_full[i,:,:])
                pil_im = Image.fromarray(roi_marks_mapped_full[i,:,:].astype(np.uint8))
                pil_im.save(rd.root_path(args.cnn_output,"roi_mapped_to_max_pil", fname=f"pil_im_{patient_id}_{i}.png"))
            
            save_ndarray(rd.root_path(args.cnn_output, fname="rois_marks_original.nparray"), roi_marks_original_full.astype(np.int64))
            save_ndarray(rd.root_path(args.cnn_output, fname="rois_marks_mapped_to_max.nparray"), roi_marks_mapped_full.astype(np.int32))

            meta_processing['rois_marks_original'] =  "rois_marks_original.nparray"
            meta_processing['rois_marks_mapped'] =  "rois_marks_mapped_to_max.nparray"

            ## DOSES
            doses_full = np.zeros( (doses.shape[0], final_shape_max[1], final_shape_max[2]) )
            doses_full[:, yfrom:yfrom+ref_slice.shape[0],xfrom:xfrom+ref_slice.shape[1]] = doses
            log.info(f"Rozpoczynam zapisywanie {doses_full.shape[0]} plików z dawkami (progowane do {args.dose_levels} poziomów) w formacie uint8 będą pliki pil")
            for i in range(doses_full.shape[0]):
                if hasattr(args, "savepng") and args.savepng:
                    plt.imsave(rd.root_path(args.cnn_output, "doses_to_max_png", fname=f"doses_{patient_id}_{i}.png"), doses_full[i,:,:])
                pil_im = Image.fromarray(doses_full[i,:,:].astype(np.uint8))
                pil_im.save(rd.root_path(args.cnn_output, "doses_to_max_pil", fname=f"pil_im_{patient_id}_{i}.png"))

            save_ndarray(rd.root_path(args.cnn_output, fname=f"doses_to_max.nparray"), doses_full.astype(np.float32))
            meta_processing['doses_mapped'] =  "doses_to_max.nparray"

            # CT
            ct_full = np.zeros( (doses.shape[0], final_shape_max[1], final_shape_max[2]) )
            ct_full[:, yfrom:yfrom+ref_slice.shape[0],xfrom:xfrom+ref_slice.shape[1]] = ct
            log.info(f"Rozpoczynam zapisywanie {ct_full.shape[0]} plików z danymi CT w formacie uint8 będą pliki pil")
            for i in range(ct_full.shape[0]):
                if hasattr(args, "savepng") and args.savepng:
                    plt.imsave(rd.root_path(args.cnn_output, "ct_to_max_png", fname=f"ct_{patient_id}_{i}.png"), ct_full[i,:,:], cmap=cm.gray)
                pil_im = Image.fromarray(ct_full[i,:,:].astype(np.uint8))
                pil_im.save(rd.root_path(args.cnn_output, "ct_to_max_pil", fname=f"pil_im_{patient_id}_{i}.png"))


            if args.mask_rcnn_output != "no_ouput_maskrcnn":
                log.info(f"Rozpoczynam zapisywanie {doses_full.shape[0]} plików z danymi o obrazach ROI i dawek w formacie uint8 będą pliki pil dla sieci Mask-RCNN")
                output = rd.root_path(args.mask_rcnn_output)
                log.info(f"Folder for mark-rcnn: {output}")
                pathlib.Path(output).mkdir(exist_ok=True)
                for i in range(doses_full.shape[0]):
                    image_id = f"{patient_id}_{i}"
                    output_image_id_path = os.path.join(output, image_id)
                    pathlib.Path(output_image_id_path).mkdir(exist_ok=True)

                    output_pngs = os.path.join(output_image_id_path, "pngs")                
                    pathlib.Path(output_pngs).mkdir(exist_ok=True)
                    plt.imsave(os.path.join(output_pngs, f"{image_id}_rois.png"), roi_marks_mapped_full[i,:,:]) # obrazek z roiami

                    output_images = os.path.join(output_image_id_path, "images")
                    pathlib.Path(output_images).mkdir(exist_ok=True)
                    pil_im = Image.fromarray(roi_marks_mapped_full[i,:,:].astype(np.uint8))
                    pil_im.save(os.path.join(output_images, f"pil_{image_id}.png"))

                    output_masks = os.path.join(output_image_id_path, "masks")
                    pathlib.Path(output_masks).mkdir(exist_ok=True)
                    for level in range(int(args.dose_levels)):
                        level_mask = doses_full[i,:,:].astype(np.int32)
                        level_mask[ level_mask != level ] = 0
                        level_mask[ level_mask == level ] = 1
                        plt.imsave(os.path.join(output_pngs, f"{image_id}_{level}.png"), level_mask) # obrazek z roiami

                        pil_im = Image.fromarray(level_mask.astype(np.uint8))
                        pil_im.save(os.path.join(output_masks, f"pil_{image_id}_{level}.png"))
        else:
            log.warn(f"Pomijam katalog {sub}, ponieważ w katalogu input brakuje pliku `roi_mapping.json`")

        with open(rd.root_path(args.cnn_output, fname=f"meta_processing.json"), "w") as f:
            json.dump(meta_processing, f, indent=4,sort_keys=True)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Przechodzi przez podfoldery i przetwarza dane przypadków do postaci, która może być wykorzystana w uczeniu sieci neuronowych (CNN)")
    parser.add_argument('root_folder', help="główny folder, który zawiera analizowane podfoldery")
    parser.add_argument('--dose-levels', type=float, help="liczba poziomów dawek w wyjściowych plikach png", default=255)
    parser.add_argument('--cnn-output', help="nazwa katalogu wynikowego z obrazami", default="ouput_cnn")
    parser.add_argument('--mask-rcnn-output', help="nazwa katalogu wynikowe z obrazami dla algorytmu Mask-RCNN", default="no_ouput_maskrcnn")
    parser.add_argument('-s', "--single", action="store_true", help="gdy podany, to będzie analizować tylko jeden folder bez poszukiwania głęboko")
    parser.add_argument('-p', "--savepng", action="store_true", help="gdy podany, to będzie zapisywać również obrazy w formacie png")
    args = parser.parse_args()

    do_run(args)