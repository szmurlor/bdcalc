#!/bin/bash
RES=$(ray start --head --node-ip-address ray-head --redis-port 6379 --temp-dir=/bdcalc/ray_session/)
sleep infinity