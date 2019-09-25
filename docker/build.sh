#! /bin/bash

updir="`dirname \"$PWD\"`"
docker build --no-cache --tag nben/neuropythy "$updir"
