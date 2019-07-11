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
RUN pip install --upgrade setuptools
RUN pip install 'ipyvolume>=0.5.1'

RUN mkdir /home/$NB_USER/neuropythy
COPY ./setup.py ./setup.cfg ./MANIFEST.in ./LICENSE.txt ./README.md \
     ./requirements-dev.txt ./requirements.txt \
     /home/$NB_USER/neuropythy/
COPY ./neuropythy /home/$NB_USER/neuropythy/neuropythy
RUN cd /home/$NB_USER/neuropythy && pip install -r requirements-dev.txt && python setup.py install

RUN mkdir -p /home/$NB_USER/.jupyter

# Copy over some files...
COPY ./docker/npythyrc /home/$NB_USER/.npythyrc
COPY ./docker/jupyter_notebook_config.py /home/$NB_USER/.jupyter/

# Copy the README and license over.
USER root
COPY ./LICENSE.txt              /LICENSE.txt
COPY ./README.md                /README.md
RUN apt-get install -y curl
RUN mkdir -p /required_subjects
#COPY docker/required_subjects.tar.gz /
RUN curl -L -o /required_subjects/fsaverage.tar.gz https://github.com/noahbenson/neuropythy/wiki/files/fsaverage.tar.gz && \
    curl -L -o /required_subjects/fsaverage_sym.tar.gz https://github.com/noahbenson/neuropythy/wiki/files/fsaverage_sym.tar.gz && \
    cd /required_subjects && tar zxf fsaverage.tar.gs && tar zxf fsaverage_sym.tar.gz && rm ./fsaverage.tar.gz ./fsaverage_sym.tar.gz

RUN cd / && tar zxvf required_subjects.tar.gz && rm /required_subjects.tar.gz && \
    chown -R root:root required_subjects && chmod -R 755 /required_subjects

# Make some global directories in the user's name also
RUN mkdir -p /subjects /freesurfer_subjects /hcp_subjects /cache && \
    chown $NB_USER /subjects /freesurfer_subjects /hcp_subjects /cache


# Make sure we have the run.sh script ready:
COPY docker/main.sh /main.sh
RUN chmod 755 /main.sh

USER $NB_USER

# And mark it as the entrypoint
#CMD ["/main.sh"]
ENTRYPOINT ["tini", "-g", "--", "/main.sh"]
