# To ponizej aby wyciągnąć katlog w którym jest uruchamiany skrypt
SCRIPT_DIR="$( cd "$(dirname $0)" && pwd)"
echo "You have started script from folder: $SCRIPT_DIR"
echo "Starting bdcalc.py from folder: $SCRIPT_DIR/../src/"
cd $SCRIPT_DIR/../src

python3 bdcalc/bdcalc.py /doses-nfs/sim/3_x_3/ -c /doses-nfs/sim/3_x_3/input/config.json
