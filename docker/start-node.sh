#!/bin/bash
RES=$(ray start --redis-address ham-10:6379 --temp-dir=/bdcalc/ray_session/)
sleep infinity