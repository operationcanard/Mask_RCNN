#!/bin/bash

data=/data

echo "Main data folder: ${data}"
echo "User: ${USER}"
sudo docker rm ${USER}_maskrcnn

echo "Choose a port for Tensorboard: "
read tport

echo "Choose a port for Jupyter Notebook: "
read jport

sudo nvidia-docker run -it --name ${USER}_maskrcnn -p ${tport}:6006 -p ${jport}:8888 -v ${data}:/data -v ${HOME}:/workspace maskrcnn bash
