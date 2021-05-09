#!/bin/bash
sleep 5
RES=$(ray start --address goope-gpu-5.iem.pw.edu.pl:6379 --temp-dir=/bdcalc/ray_session/)
#RES=$(ray start --address ray-1.iem.pw.edu.pl:6379 --temp-dir=/bdcalc/ray_session/)
#RES=$(ray start --address ham-10.iem.pw.edu.pl:6379 --temp-dir=/bdcalc/ray_session/)
sleep infinity
