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
RUN apt-get -y install g++ make python2.7 python-dev python-setuptools python-numpy \
                       python-scipy python-pip libxml2 libxml2-dev zlib1g zlib1g-dev git \
                       software-properties-common python-software-properties
RUN add-apt-repository ppa:webupd8team/java -y && apt-get -y update
RUN echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections
RUN echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections
RUN apt-get -y install oracle-java8-installer

RUN pip install --upgrade setuptools
RUN mkdir -p /required_subjects

# Copy the README and license over.
COPY contents/LICENSE.txt              /LICENSE.txt
COPY contents/README.md                /README.md
COPY contents/required_subjects.tar.gz /

RUN cd / && tar zxvf required_subjects.tar.gz && rm /required_subjects.tar.gz

# Install python, and get the neuropythy library working.
RUN cd /opt && git clone https://github.com/noahbenson/neuropythy
RUN cd /opt/neuropythy && git submodule init && git submodule update && python setup.py install

# Make sure we have the run.sh script ready:
COPY contents/main.sh /main.sh
RUN chmod 755 /main.sh
# And mark it as the entrypoint
ENTRYPOINT ["/main.sh"]

