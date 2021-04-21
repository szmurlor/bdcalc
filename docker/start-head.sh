#!/bin/bash
RES=$(ray start --head --node-ip-address ray-1.iem.pw.edu.pl --port 6379 --temp-dir=/bdcalc/ray_session/)
#RES=$(ray start --head --node-ip-address ham-10.iem.pw.edu.pl --port 6379 --temp-dir=/bdcalc/ray_session/)
sleep infinity
