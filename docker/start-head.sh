#!/bin/bash
RES=$(ray start --head --node-ip-address ham-10 --redis-port 6379 --temp-dir=/bdcalc/ray_session/)
sleep infinity