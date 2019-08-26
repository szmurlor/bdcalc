#!/bin/bash
RES=$(ray start --redis-address ray-head:40000 --temp-dir=/bdcalc/ray_session/)
sleep infinity