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

# Install some extras; first collapsible cell extensions...
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
# next, the helvetica neue font (for plotting)
RUN mkdir -p ~/.local/share/fonts/helvetica_neue_tmp
RUN curl -L -o ~/.local/share/fonts/helvetica_neue_tmp/helveticaneue.zip \
         https://github.com/noahbenson/neuropythy/wiki/files/helveticaneue.zip
RUN cd ~/.local/share/fonts/helvetica_neue_tmp \
 && unzip helveticaneue.zip \
 && mv *.ttf .. \
 && cd .. \
 && rm -r ~/.local/share/fonts/helvetica_neue_tmp
RUN fc-cache -f -v



# The root operations ...
USER root

# Copy the README and license over.
RUN apt-get update && apt-get install -y --no-install-recommends curl
COPY ./LICENSE.txt              /LICENSE.txt
COPY ./README.md                /README.md
RUN mkdir -p /data/required_subjects
RUN curl -L -o /data/required_subjects/fsaverage.tar.gz https://github.com/noahbenson/neuropythy/wiki/files/fsaverage.tar.gz && \
    curl -L -o /data/required_subjects/fsaverage_sym.tar.gz https://github.com/noahbenson/neuropythy/wiki/files/fsaverage_sym.tar.gz && \
    cd /data/required_subjects && tar zxf fsaverage.tar.gz && tar zxf fsaverage_sym.tar.gz && rm ./fsaverage.tar.gz ./fsaverage_sym.tar.gz && \
    chown -R root:root /data/required_subjects && chmod -R 755 /data/required_subjects

# Make some global directories in the user's name also
RUN mkdir -p /data/hcp && \
    chown $NB_USER /data /data/hcp && \
    chmod 755 /data /data/hcp

# Make sure we have the run.sh script ready:
COPY docker/main.sh /main.sh
COPY docker/help.txt /help.txt
RUN chmod 755 /main.sh /help.txt

USER $NB_USER

# And mark it as the entrypoint
#CMD ["/main.sh"]
ENTRYPOINT ["tini", "-g", "--", "/main.sh"]
