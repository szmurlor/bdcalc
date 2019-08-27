#!/bin/bash
RES=$(ray start --redis-address ray-0:6379 --temp-dir=/bdcalc/ray_session/)
sleep infinity
