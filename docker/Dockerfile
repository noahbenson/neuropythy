# This Dockerfile constructs a docker image that contains an installation
# of the Neuropythy library.
#
# Example build:
#   docker build --no-cache --tag nben/neuropythy `pwd`
#

# Start with the Ubuntu 
FROM ubuntu:trusty

# Note the Maintainer.
MAINTAINER Noah C. Benson <nben@nyu.edu>

# Install some stuff...
RUN apt-get -y update
RUN apt-get -y install g++ make python2.7 python-dev python-setuptools python-igraph python-numpy python-scipy libxml2 libxml2-dev zlib1g zlib1g-dev git default-jre

# Copy the README and license over.
COPY contents/LICENSE.txt  /LICENSE.txt
COPY contents/README.md    /README.md

# Install python, and get the neuropythy library working.
RUN cd /opt && git clone https://github.com/noahbenson/neuropythy
RUN cd /opt/neuropythy && git submodule init && git submodule update && python setup.py install

# Make sure we have the run.sh script ready:
COPY contents/main.sh /main.sh
RUN chmod 755 /main.sh
# And mark it as the entrypoint
ENTRYPOINT ["/main.sh"]



