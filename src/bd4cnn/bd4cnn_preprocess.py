import sys
import os
import json
import argparse
import logging as log
import shutil
import pydicom

log.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=log.INFO)

def do_run(args):
    from_folder = args.from_folder
    to_folder = args.to_folder

    # Lista podkatalogów z plikami DICOM
    subs = next(os.walk(from_folder))[1]
    log.debug(f"Lista folderów: {subs}")

    if not os.path.exists(to_folder):
        os.mkdir(to_folder)

    for sub in subs:
        log.info(f"Working on subdir: {sub}")
        from_sub = os.path.join(from_folder, sub)
        to_sub = os.path.join(to_folder, sub)
        to_sub_input = os.path.join(to_sub, "input")
        to_sub_input_dicom = os.path.join(to_sub_input, "dicom")

        meta = {
            "from_folder": from_folder,
            "subfolder_name": sub
        }

        if not os.path.exists(to_sub):
            os.mkdir(to_sub)

        if not os.path.exists(to_sub_input):
            os.mkdir(to_sub_input)

        if not os.path.exists(to_sub_input_dicom):
            os.mkdir(to_sub_input_dicom)

        r,d,f = next(os.walk(from_sub))
        for from_file in f:
            if from_file.endswith('.dcm'):
                log.debug(f"Copying file: {from_file}")
                shutil.copy(os.path.join(from_sub, from_file), os.path.join(to_sub_input_dicom, from_file))

                if (from_file.startswith("RP")):
                    rp = pydicom.read_file(os.path.join(to_sub_input_dicom, from_file))

                    meta["plan_label"] = rp.RTPlanLabel
                    log.debug(rp.RTPlanLabel)
                    meta["patient_id"] = rp.PatientID
                    lbl = rp.RTPlanLabel.lower()
                    if "piers" in lbl:
                        meta["piers"] = True
                    else:
                        meta["piers"] = False
                    if "blizn" in lbl:
                        meta["blizna"] = True
                    else:
                        meta["blizna"] = False
                    if "wezl" in lbl:
                        meta["wezly"] = True
                    else:
                        meta["wezly"] = False
                    if " p " in lbl or "p+" in lbl or "+l" in lbl:
                        meta["prawa"] = True
                    else:
                        meta["prawa"] = False
                    if " l " in lbl or "l+" in lbl or "+l" in lbl:
                        meta["lewa"] = True
                    else:
                        meta["lewa"] = False


                if (from_file.startswith("RS")):
                    rs = pydicom.read_file(os.path.join(to_sub_input_dicom, from_file))
                    rois = {}                    
                    for roi in rs.StructureSetROISequence:
                        rois[roi.ROIName] = roi.ROINumber
                    
                    meta["rois"] = rois


        with open(os.path.join(to_sub, "meta.json"), "w") as fout:
            fout.write(json.dumps(meta, indent="    "))


    log.info("Finished.")    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Przechodzi przez podfoldery zadanego folderu i zakłada, że każdym podfolderze znajduje się " +
                                                 " jeden przypadek (pacjent) z czterema rodzajami plików DICOM: RD (dawki), RS (kontury), CT (obrazy) i RP (plan). " +
                                                 "W wyniku w folderze docelowym tworzy strukturę plików przygotowaną do przetwarzania ROIów i dawek. " + 
                                                 "Jest to narzędzie do przygotowywania bazy danych plikowych do uczenia sieci CNN.")
    parser.add_argument('from_folder', help="folder żródłowy z podfolderami, które zawierają pliki DICOM")
    parser.add_argument('to_folder', help="folder docelowy, gdzie skopiowane zostaną pliki")
    args = parser.parse_args()

    do_run(args)