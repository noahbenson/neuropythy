# This Dockerfile constructs a docker image that contains an installation
# of the Neuropythy library.
#
# Example build:
#   docker build --no-cache --tag nben/neuropythy `pwd`
#
#   (but really, use docker-compose up instead).
#

# Start with the Jupyter scipy notebook docker-image.
# We tag this to a specific version so that we're assured of future success.
FROM jupyter/scipy-notebook:lab-3.4.3

# Note the Maintainer.
MAINTAINER Noah C. Benson <nben@uw.edu>

# Install some stuff...
RUN conda update --yes -n base conda && conda install --yes py4j nibabel s3fs
RUN conda install --yes -c conda-forge ipywidgets
RUN conda install --yes pip
RUN pip3 install --upgrade setuptools

# We need additional stuff for ipyvolume to work in Jupyter Labs
RUN conda install --yes -c conda-forge ipyvolume nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install ipyvolume
RUN jupyter labextension install jupyter-threejs

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

# Install collapsible cell extensions...
RUN conda install -c conda-forge jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable collapsible_headings/main \
 && jupyter nbextension enable select_keymap/main
 


# The root operations ...
USER root

# Copy the README and license over.
RUN apt-get update && apt-get install -y --no-install-recommends curl
RUN apt-get install -y default-jdk fonts-open-sans
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

# As the use (now with curl!), install the helvetica neue font (for figures)
RUN mkdir -p ~/.local/share/fonts/helvetica_neue_tmp
RUN curl -L -o ~/.local/share/fonts/helvetica_neue_tmp/helveticaneue.zip \
         https://github.com/noahbenson/neuropythy/wiki/files/helveticaneue.zip
RUN cd ~/.local/share/fonts/helvetica_neue_tmp \
 && unzip helveticaneue.zip \
 && mv *.ttf .. \
 && cd .. \
 && rm -r ~/.local/share/fonts/helvetica_neue_tmp
RUN fc-cache -f -v
RUN rm -r ~/.cache/matplotlib

# And mark it as the entrypoint
#CMD ["/main.sh"]
ENTRYPOINT ["tini", "-g", "--", "/main.sh"]
