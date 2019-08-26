#!/bin/bash
RES=$(ray start --head --node-ip-address ray-head --redis-port 40000 --temp-dir=/bdcalc/ray_session/)
sleep infinity