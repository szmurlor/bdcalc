#!/bin/bash
#set -x

NFS_VOLUME=doses-nfs
NFS_EXPORT_PATH=/nfs/doses
NFS_HOST=goope-nas-2.iem.pw.edu.pl
BDCALC_IMAGE=szmurlor/bdcalc:0.1
NFS_SIM_PATH=/doses-nfs
DNS=10.40.33.2

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -i)
    INTERACTIVE=-ti
    ;;
    *)
    echo "Usage: run-docker-head.sh [ -i ] "
    echo "Starts a docker continer for the head node. It checks for existence of a volume to the common NFS share with simulation results."
    echo "Options:"
    echo "    -i: starts the container in an interactive mode allocating a terminal"
    exit 1
esac
shift
done

EXISTING_IMAGE=$(docker volume ls | grep $NFS_VOLUME | awk '{print $2}')
if [ -z $EXISTING_IMAGE ]; then
    echo "-------------------------------------"
    echo "Creating a docker volume $NFS_VOLUME"
    echo "-------------------------------------"
    docker volume create --name $NFS_VOLUME --opt type=nfs --opt device=:NFS_EXPORT_PATH --opt o=addr=NFS_HOST
    if [ $? != 0 ]; then
        echo "ERROR! Unable to create NFS docker volume"
        exit 1
    fi
else
    echo "Docker volume $NFS_VOLUME exists."
fi

echo "-------------------------------------------"
echo "Starting a docker container with the image "
echo "-------------------------------------------"
if [ $INTERACTIVE ]; then
    docker run $INTERACTIVE --dns=$DNS -v $NFS_VOLUME:/$NFS_SIM_PATH $BDCALC_IMAGE bash
else
    docker start $INTERACTIVE --dns=$DNS -v $NFS_VOLUME:/$NFS_SIM_PATH $BDCALC_IMAGE
fi
