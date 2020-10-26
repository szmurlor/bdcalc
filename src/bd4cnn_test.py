import sys
import os
from bdfileutils import read_ndarray
from rass import RASSData
import matplotlib.pyplot as plt
import numpy as np
import json

rass_data = RASSData(root_folder=sys.argv[1])

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
# przykładowstruktura tego pliku:
# {
#    "ptv-plan": 6,
#    "kanal kreg.": 5,
#    "serce": 4,
#    "pluco P": 3,
#    "pluco L": 2,
#    "Patient Outline": 1
#}
#
# najważniejsze są roie o najwyzszych numerach. Czyli jak woksel należydo do kilku roiow,
# to w zmapowanym obrazie zostanie dla nie przypisany najwyzszy numer - zazwyczaj PTV jest najwazniejszy.

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


