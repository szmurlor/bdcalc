import logging
import sys

log = logging.getLogger("bdcalc")
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

log.handlers.clear()
log.addHandler(ch)