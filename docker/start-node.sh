#!/bin/bash
RES=$(ray start --redis-address ray-head:6379 --temp-dir=/bdcalc/ray_session/)
sleep infinity