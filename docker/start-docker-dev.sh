#!/bin/bash
docker run -ti -v /doses-nfs:/doses-nfs -v $PWD:/bdcalc-dev szmurlor/bdcalc:0.1 bash
