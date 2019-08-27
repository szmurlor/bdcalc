#!/bin/bash
RES=$(ray start --head --node-ip-address ray-0 --redis-port 6379 --temp-dir=/bdcalc/ray_session/)
sleep infinity
