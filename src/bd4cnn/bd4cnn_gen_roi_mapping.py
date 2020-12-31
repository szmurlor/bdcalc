import sys
import os
import json
import argparse
import logging as log
import shutil
import pydicom
import re

log.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=log.INFO)

def do_run(args):
    root_folder = args.root_folder
    mapping_config = args.roi_mapping_config

    roi_mapping_config = {}
    with open(mapping_config) as f:
        roi_mapping_config.update(json.load(f))
    
    log.info(roi_mapping_config)

    # Lista podkatalogów z plikami DICOM
    r,subs,f = next(os.walk(root_folder))
    log.debug(f"Lista folderów: {subs}")

    for sub in subs:
        log.info(f"Working on genearation of roi mapping in subdir: {sub}")
        from_sub = os.path.join(root_folder, sub)
        meta_fname = os.path.join(from_sub, "meta.json")

        meta = {}
        with open(meta_fname) as f:
            meta.update(json.loads("".join(f.readlines())))
        log.debug(meta)

        meta.pop("unmatched_rois", None)

        mapping = {}
        for roi in roi_mapping_config['rois']:
            key = roi['id']
            found = False
            for m in roi['matches']: 
                log.debug(m)
                for roiname in meta["rois"]:
                    log.debug(roiname)
                    if re.match(m, roiname) is not None:
                        mapping[roiname] = key
                        found = True
            
            if not found:
                meta["unmatched_rois"] = ", ".join([meta["unmatched_rois"], f"Missing: {key}"]) if "unmatched_rois" in meta else f"Missing: {key}"
        
        if "unmatched_rois" in meta:
            log.warn(f'Unmatched rois: {meta["unmatched_rois"]}')
        with open(os.path.join(from_sub, "roi_mapping.json"), "w") as f:
            f.write(json.dumps(mapping, indent="    "))

        with open(os.path.join(from_sub, "meta.json"), "w") as fout:
            fout.write(json.dumps(meta, indent="    "))

    log.info("Finished.")    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Przechodzi przez podfoldery zadanego folderu próbuje dopasować nazwy ROIów do danych z pliku mapującego.\n\n" +
                                                 "Plik mapujący to np.: \n" +
                                                 '{\n'+
                                                 '   "rois": [ \n' +
                                                 '   {\n'+
                                                 '       "id": 14,\n'+
                                                 '       "matches": ["*blizna*"]\n'+
                                                 '   }\n' +
                                                 ']}')
    parser.add_argument('root_folder', help="folder żródłowy z podfolderami, które zawierają dane w formacie Rass Data (input, ouput)")
    parser.add_argument('roi_mapping_config', help="plik w formacie JSON z konfiguracją dopasowywania roiów i przypisanych im numerów")
    args = parser.parse_args()

    do_run(args)