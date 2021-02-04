import os
import json
import argparse
import logging as log
import numpy as np
from bdfileutils import read_ndarray
from rass import RASSData

import numpy as np
import pydicom as dicom
import scipy.interpolate
from scipy.interpolate import interp1d

log.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=log.DEBUG)

"""
 Dose constraints
OAR Constraints in 25-30 Fractions
Spinal cord Max ≤ 50 Gy
Lung V20 ≤ 20~30%; V5 ≤ 70%; MLD ≤ 20Gy
Heart V30 ≤ 46%, V40 ≤ 5%
Esophagus Mean ≤ 34Gy; Max ≤ 105% of prescription dose
Vxx=% of the whole OAR receiving ≥ xx Gy

Zródło: http://sub.chimei.org.tw/55700/images/pdf/03.pdf, str. 5 
"""
def calc_statistics(meta, meta_processing, roi_file, doses_file, dtype):

    # reading data
    rois = read_ndarray(roi_file, dtype=np.int64)
    doses = read_ndarray(doses_file, dtype=dtype)

    # getting meta data
    target_dose = meta['TargetPrescriptionDose']
    gridDoseScaling = meta['DoseGridScaling']
    total_max = float(meta_processing["max_dose_global"])
    levels = int(meta_processing["dose_levels"])
 
    # To jest porypane, bo musimy przeskalować wartości np. 50 progów na odpowiednią dawkę, 
    # a w dodatku, to skalowanie mogło być robione z porypaną skalą- nie wiadomo jakim total maxem 
    # kurde < warto byłoby to do meta danych zapisywać. 
    #
    # the doses are scaled to arg.level number of integer classes
    # we need to scale the classes (eg. 50) to original doses
    # we need to use doseGrdiScaling value to get absolute value in grays
    upscale =  gridDoseScaling * float(total_max) / float(levels)
    
    print("-"*120)
    print(f"   Prescribed dose: {target_dose}")
    print("-"*120)
    print(f"{'ROI Name':^30} [{'Bit':^10}]{'min':^6} | {'max':^6} | {'avg':^6} | {'d99%':^6} | {'d98%':^6} | {'d95%':^6} | {'V95%':^6} | {'V98%':^6} | {'V105%':^6}")
    print("-"*120)
    for k,r in meta['roi_bits'].items():
        b = 2**(r-1)
        mask = np.bitwise_and(rois,b) == b 
        min = None
        max = None
        avg = None
        d99 = 0
        d98 = 0
        d95 = 0
        v95 = 0
        v98 = 0
        v105 = 0
        if np.max(mask) > 0:
            the = doses[mask]*upscale
            min = np.min(the)
            max = np.max(the)
            avg = np.mean(the)
            if "ptv" in k.lower():
                roi_vol = np.sum(mask)

                yk = target_dose * np.array([0.4, 0.5, 0.60, 0.70, 0.80, 0.85, 0.88, 0.92, 0.94, 0.99, 1, 1.01, 1.05, 1.1])
                xk = [np.sum( the > y ) / roi_vol*100 for y in yk]    
                #print(xk)
                #print(yk)
                cs = interp1d(xk,yk)
                if np.max(xk) > 99:
                    d99 = cs( 99)
                    d98 = cs( 98)
                    d95 = cs( 95)

                v90 = np.sum( the > target_dose * 0.90 ) / roi_vol * 100
                v95 = np.sum( the > target_dose * 0.95 ) / roi_vol * 100
                v98 = np.sum( the > target_dose * 0.98 ) / roi_vol * 100
                v99 = np.sum( the > target_dose * 0.99 ) / roi_vol * 100
                v105 = np.sum( the > target_dose * 1.05 ) / roi_vol * 100
                v107 = np.sum( the > target_dose * 1.07 ) / roi_vol * 100

                #fig, ax = plt.subplots(figsize=(6.5, 4))
                #ax.plot(xk, yk, 'o', label='data')
                #print(np.arange(p80,p100-0.01,0.01))
                #ax.plot(np.arange(p80,p100-0.01,0.01), cs(np.arange(p80,p100-0.01,0.01)), label='spline')
                #plt.show()
        
        if not "zz" in k:
            print(f"{k:>30} [{b:>10}]{min:>6.2f} | {max:>6.2f} | {avg:>6.2f} | {d99:>6.2f} | {d98:>6.2f} | {d95:>6.2f} | {v95:>6.2f} | {v98:>6.2f} | {v105:>6.2f}")
    

def do_run(args):
    root_folder = args.root_folder
    if (hasattr(args,"single") and args.single):
        subs = [args.root_folder]
    else:
        subs = [os.join(root_folder,sub) for sub in next(os.walk(root_folder))[1] ]

    rows = []
    for sub in subs:
        row = []
        rd = RASSData(root_folder=sub)

        with open(rd.root("meta.json")) as f:
            meta = json.load(f)

        roi_subfolder_full = os.path.join(sub, args.roi_subfolder)

        meta_processing = {}
        mp_path = os.path.join(roi_subfolder_full, "meta_processing.json")
        if os.path.isfile(mp_path):
            with open(mp_path) as f:
                meta_processing = json.load(f)

        if hasattr(args, "total_max") and args.total_max is not None:
            meta_processing['max_dose_global'] = float(args.total_max)
        if hasattr(args, "levels") and args.levels is not None:
            meta_processing['dose_levels'] = int(args.levels)

        total_max = float(meta_processing["max_dose_global"])
        levels = int(meta_processing["dose_levels"])

        meta_processing['dose_levels_scale'] = float(levels) / float(total_max)
        meta_processing['dose_levels_upscale'] = float(total_max) / float(levels)

        calc_statistics(meta, meta_processing, 
                os.path.join(sub, args.roi_subfolder, meta_processing['rois_marks_original']), 
                os.path.join(sub, args.doses_subfolder_file),
                dtype=np.dtype(args.doses_dtype))
        
        rows.append(row)
    
    res = None
    for r in rows:
        rjoined = ";".join(map(str,r))
        res = "\n".join([res, rjoined]) if res is not None else rjoined
    
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Przechodzi przez podfoldery (lub jeden folder --single) i przetwarza dane i wyświetla wynik w stabularyzowanej postaci")
    parser.add_argument('root_folder', help="główny folder, który zawiera analizowane podfoldery")
    parser.add_argument('roi_subfolder', help="katalog z danymi o roiach oraz plikim 'meta_processing.json' w podkatalogach z danymi np.: read_ndarray([root_folder]/1093545/[rois_subfolder])")
    parser.add_argument('doses_subfolder_file', help="nazwa pliku z danymi o dawkach w podkatalogach z danymi np.: read_ndarray([root_folder]/1093545/[doses_subfolder_file])")
    parser.add_argument('--single',  action="store_true", help="przetwarza tylko jeden folder podany bezpośrednio jako root")
    parser.add_argument('--total-max', help="skaluj do wartości maksymalnej (jeżeli nie zostanie podana żadna wartość to użyję z pliku '[roi_subfolder]/meta_processing.json')")
    parser.add_argument('--levels', help="liczba poziomów  (jeżeli nie zostanie podana żadna wartość to użyję z pliku '[roi_subfolder]/meta_processing.json')")
    parser.add_argument('--doses-dtype', help="typ danych w plikach nparray w dawkach - domyślnie 'f4' (np.float32), inne opcje: 'u1' (np.uint8) (roi zawsze są np.int64)", default="f4")
    parser.add_argument('-f', '--output-file', help="nazwa pliku do której zapisać wynik", default=None)
    args = parser.parse_args()

    res = do_run(args)

    if hasattr(args, 'output_file') and args.output_file is not None:
        with open(args.output_file,"w") as f:
            f.write(res)
    else:
        print(res)

    log.info("DONE.")