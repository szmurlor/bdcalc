#!/usr/bin/env python
import sys

NC='\033[0m' # No Color
ORANGE='\033[0;33m'
RED='\033[0;31m'
for l in sys.stdin:
    if "- error -" in l.lower(): 
       print(f"{RED}{l}{NC}",end='')
    elif "- warning -" in l.lower(): 
       print(f"{ORANGE}{l}{NC}",end='')
    else:
       print(l,end='')