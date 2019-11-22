#!/bin/bash

# To ponizej aby wyciągnąć katlog w którym jest uruchamiany skrypt
SCRIPT_DIR="$( cd "$(dirname $0)" && pwd)"
echo "You have started script from folder: $SCRIPT_DIR"
echo "Starting bdcalc.py from folder: $SCRIPT_DIR/../src/"
cd $SCRIPT_DIR/../src
python3 /doses-nfs/bdcalc/src/bdcalc.py /doses-nfs/sim/pacjent_3/ -c /doses-nfs/sim/pacjent_3/input/config.json
