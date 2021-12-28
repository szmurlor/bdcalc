#!/bin/bash

if [ "$#" -lt 1 ]; then
	echo -e "ERROR!\n\nRequired positional argument is missing: the subfolder of the case to calculate. The correct usage is:\n\t$0 case_name\n"
	exit 2
fi

# To ponizej aby wyciągnąć katlog w którym jest uruchamiany skrypt
SCRIPT_DIR="$( cd "$(dirname $0)" && pwd)"
echo "You have started script from folder: $SCRIPT_DIR"
echo "Starting bdcalc.py from folder: $SCRIPT_DIR/../src/"

set -e

cd $SCRIPT_DIR/../src
python3 /doses-nfs/bdcalc/src/bdcalc/bdcalc.py /doses-nfs/sim/$1/ -c /doses-nfs/sim/$1/input/config.json
