#!/bin/bash

# To ponizej aby wyciągnąć katlog w którym jest uruchamiany skrypt
SCRIPT_DIR="$( cd "$(dirname $0)" && pwd)"
echo "You have started script from folder: $SCRIPT_DIR"
echo "Starting bdcalc.py from folder: $SCRIPT_DIR/../src/"
cd $SCRIPT_DIR/../src
python3 bdcalc/bdcalc.py /doses-nfs/sim/NIO_2021_04_02/ -c /doses-nfs/sim/NIO_2021_04_02/input/config.json
cd $SCRIPT_DIR/../utils/
make
./convdoses 9 /doses-nfs/sim/NIO_2021_04_02/output/PARETO_5/d_PARETO_5_ 6
