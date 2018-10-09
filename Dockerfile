# This Dockerfile constructs a docker image that contains an installation
# of the Neuropythy library.
#
# Example build:
#   docker build --no-cache --tag nben/neuropythy `pwd`
#
#   (but really, use docker-compose up instead).
#

# Start with the Ubuntu for now
FROM jupyter/scipy-notebook

# Note the Maintainer.
MAINTAINER Noah C. Benson <nben@nyu.edu>

# Install some stuff...
RUN conda update --yes -n base conda && conda install --yes py4j nibabel s3fs
RUN conda install --yes -c conda-forge ipywidgets
RUN pip install 'ipyvolume>=0.5.1'

RUN mkdir /neuropythy
COPY ./neuropythy ./setup.py ./setup.cfg ./MANIFEST.in ./LICENSE.txt ./README.md \
     ./requirements-dev.txt ./requirements.txt \
     /neuropythy/
RUN cd /neuropythy && pip install requirements-dev.txt && python setup.py install

RUN mkdir -p /home/$NB_USER/data       \
             /home/$NB_USER/data/HCP   \
             /home/$NB_USER/data/cache \
             /home/$NB_USER/.jupyter

# Copy over some files...
COPY ./docker/npythyrc /home/$NB_USER/.npythyrc
COPY ./docker/jupyter_notebook_config.py /home/$NB_USER/.jupyter/

RUN pip install --upgrade setuptools
RUN mkdir -p /required_subjects

# Copy the README and license over.
COPY ./LICENSE.txt              /LICENSE.txt
COPY ./README.md                /README.md
COPY docker/required_subjects.tar.gz /

RUN cd / && tar zxvf required_subjects.tar.gz && rm /required_subjects.tar.gz

# Make sure we have the run.sh script ready:
COPY docker/main.sh /main.sh
RUN chmod 755 /main.sh
# And mark it as the entrypoint
ENTRYPOINT ["/main.sh"]
