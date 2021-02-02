import os
import json
import argparse
import logging as log
import numpy as np
from bdfileutils import read_ndarray
from rass import RASSData

log.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=log.DEBUG)

def do_run(args):
    root_folder = args.root_folder
    if (hasattr(args,"single") and args.single):
        subs = [args.root_folder]
    else:
        subs = next(os.walk(root_folder))[1]

    rows = []
    for sub in subs:
        row = []
        folder = os.path.join(root_folder, sub)
        rd = RASSData(root_folder=folder)

        with open(rd.root("meta.json")) as f:
            meta = json.load(f)
        
        if hasattr(args,"folder") and args.folder:
            row.append(sub)

        if hasattr(args,"patient_id") and args.patient_id:
            row.append(meta["patient_id"])

        if hasattr(args,"absolute_folder") and args.absolute_folder:
            row.append(folder)

        if hasattr(args,"plan_label") and args.plan_label:
            row.append(meta["plan_label"])

        rows.append(row)
    
    res = None
    for r in rows:
        rjoined = ";".join(map(str,r))
        res = "\n".join([res, rjoined]) if res is not None else rjoined
    
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Przechodzi przez podfoldery (lub jeden folder --single) i przetwarza dane i wyświetla wynik w stabularyzowanej postaci")
    parser.add_argument('root_folder', help="główny folder, który zawiera analizowane podfoldery")
    parser.add_argument('--folder',  action="store_true", help="wyśwetl podfolder")
    parser.add_argument('--absolute-folder',  action="store_true", help="wyświetl bezględną ścieżkę do folderu")
    parser.add_argument('--patient-id',  action="store_true", help="wyświetl identyfikator pacjenta")
    parser.add_argument('--plan-label',  action="store_true", help="wyświetla nazwę planu")
    parser.add_argument('--single',  action="store_true", help="przetwarza tylko jeden folder podany bezpośrednio jako root")
    parser.add_argument('-f', '--output-file', help="nazwa pliku do której zapisać wynik", default=None)
    args = parser.parse_args()

    res = do_run(args)

    if hasattr(args, 'output_file') and args.output_file is not None:
        with open(args.output_file,"w") as f:
            f.write(res)
    else:
        print(res)

    log.info("DONE.")