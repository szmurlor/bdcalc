import sys
import os
from bdfileutils import read_ndarray
from rass import RASSData
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import json
from PIL import Image

rass_data = RASSData(root_folder=sys.argv[1])

# utworz katalog do zapisu wejscia sieci
roi_path = rass_data.output() + "roi"
if not os.path.isdir(roi_path):
    os.makedirs(roi_path)

#utworz katalog do zapisu wyjscia sieci
dose_path = rass_data.output() + "dose"
if not os.path.isdir(dose_path):
    os.makedirs(dose_path)

#liczba klas do ktorych ma byc klasyfikowana dawka
if len(sys.argv) > 2:
    levels = int(sys.argv[2])
else:
    levels = 255

# ct[z,y,x]
ct = read_ndarray(rass_data.output("approximated_ct.nparray"))
for i in range(ct.shape[0]):
    plt.imsave(rass_data.output(f"ct_{i}.png"), ct[i,:,:], cmap = cm.gray)

# doses[z,y,x]
doses = read_ndarray(rass_data.output("total_doses.nparray"))
doses = np.round(doses/np.max(doses) * levels)
for i in range(doses.shape[0]):
    plt.imsave(rass_data.output(f"doses_{i}.png"), doses[i,:,:])
    pil_im = Image.fromarray(doses[i,:,:].astype(np.uint32))
    pil_im.save(rass_data.output(os.path.join("dose", f"{i}.png")))

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
        pil_im = Image.fromarray(roi_marks_mapped[i,:,:].astype(np.uint32))
        pil_im.save(rass_data.output(os.path.join("roi", f"{i}.png")))


