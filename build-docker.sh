#!/bin/bash
set -x

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --no-cache)
    NO_CACHE="--no-cache"
    ;;
    --output-sha)
    # output the SHA sum of the last built file, suppressing all other output. 
    # This is useful for scripting tests, especially when builds of different versions
    # are running on the same machine. It also can facilitate cleanup.
    OUTPUT_SHA=YES
    ;;
    *)
    echo "Usage: build-docker.sh [ --no-cache ] [ --sha-sums ]"
    exit 1
esac
shift
done


# Build the current bdcalc source
if [ $OUTPUT_SHA ]; then
    IMAGE_SHA=$(docker build $NO_CACHE -q -f docker/Dockerfile -t szmurlor/bdcalc:0.1 .)
else
    #docker build --no-cache -q -f docker/Dockerfile -t szmurlor/bdcalc:0.1 .
    docker build $NO_CACHE -f docker/Dockerfile -t szmurlor/bdcalc:0.1 .
fi

if [ $OUTPUT_SHA ]; then
    echo "$IMAGE_SHA" | sed 's/sha256://'
fi

