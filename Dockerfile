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
#FROM jupyter/scipy-notebook:lab-3.4.3
FROM jupyter/scipy-notebook:python-3.9.13

# Note the Maintainer.
MAINTAINER Noah C. Benson <nben@uw.edu>

# Initial Root Operations ######################################################
USER root

# Install things that require apt.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && apt-get install -y default-jdk fonts-open-sans

# Make some global directories in the user's name also
RUN mkdir -p /data/required_subjects \
 && chown -R root:root /data/required_subjects \
 && chmod -R 775 /data/required_subjects
RUN mkdir -p /data/hcp \
 && chown $NB_USER /data /data/hcp \
 && chmod 775 /data /data/hcp

# Download the required FreeSurfer subjects.
RUN curl -L -o /data/required_subjects/fsaverage.tar.gz \
      https://github.com/noahbenson/neuropythy/wiki/files/fsaverage.tar.gz \
 && cd /data/required_subjects \
 && tar zxf fsaverage.tar.gz \
 && rm fsaverage.tar.gz
RUN curl -L -o /data/required_subjects/fsaverage_sym.tar.gz \
      https://github.com/noahbenson/neuropythy/wiki/files/fsaverage_sym.tar.gz \
 && cd /data/required_subjects \
 && tar zxf fsaverage_sym.tar.gz \
 && rm ./fsaverage_sym.tar.gz


# Initial User Operations ######################################################
USER $NB_USER

# Install our Python dependencies.
RUN eval "$(command conda shell.bash hook)" \
 && conda activate \
 && conda install --yes                py4j nibabel s3fs pip \
 && conda install --yes -c conda-forge ipywidgets widgetsnbextension \
                                       ipyvolume nodejs \
                                       jupyter_contrib_nbextensions \
 && conda install --yes -c pytorch     pytorch torchvision \
 && conda update --yes --all
RUN eval "$(command conda shell.bash hook)" \
 && conda activate \
 && pip install --upgrade setuptools

# We need to do some extra work for ipyvolume to work in jupyter-labs
# and with nbextensions.
RUN eval "$(command conda shell.bash hook)" \
 && conda activate \
 && jupyter labextension install @jupyter-widgets/jupyterlab-manager \
 && jupyter labextension install ipyvolume \
 && jupyter labextension install jupyter-threejs
RUN eval "$(command conda shell.bash hook)" \
 && conda activate \
 && jupyter nbextension enable collapsible_headings/main \
 && jupyter nbextension enable select_keymap/main \
 && jupyter nbextension enable --py --user widgetsnbextension \
 && jupyter nbextension enable --py --user pythreejs \
 && jupyter nbextension enable --py --user ipywebrtc \
 && jupyter nbextension enable --py --user ipyvolume

# Install the helvetica neue font (for figures).
RUN mkdir -p ~/.local/share/fonts/helvetica_neue_tmp
RUN curl -L -o ~/.local/share/fonts/helvetica_neue_tmp/helveticaneue.zip \
       https://github.com/noahbenson/neuropythy/wiki/files/helveticaneue.zip \
 && cd ~/.local/share/fonts/helvetica_neue_tmp \
 && unzip helveticaneue.zip \
 && mv *.ttf .. \
 && cd .. \
 && rm -r ~/.local/share/fonts/helvetica_neue_tmp \
 && fc-cache -f -v \
 && rm -r ~/.cache/matplotlib

# Install Neuropythy from the current directory.
RUN mkdir /home/$NB_USER/neuropythy
COPY ./setup.py ./setup.cfg ./MANIFEST.in ./LICENSE.txt ./README.md \
     ./requirements-dev.txt ./requirements.txt \
     /home/$NB_USER/neuropythy/
COPY ./neuropythy /home/$NB_USER/neuropythy/neuropythy
RUN eval "$(command conda shell.bash hook)" \
 && conda activate \
 && cd /home/$NB_USER/neuropythy \
 && pip3 install -r ./requirements-dev.txt \
 && python3 setup.py install


# Final Root Operations ########################################################
USER root

# Copy the README, license, help, and script files over.
COPY LICENSE.txt     /LICENSE.txt
COPY README.md       /README.md
COPY docker/main.sh  /main.sh
COPY docker/help.txt /help.txt
RUN chmod 755 /main.sh
RUN chmod 644 /help.txt /README.md /LICENSE.txt


# Final User Operations ########################################################
USER $NB_USER

# Copy over some files...
RUN mkdir -p /home/$NB_USER/.jupyter
COPY ./docker/npythyrc /home/$NB_USER/.npythyrc
COPY ./docker/jupyter_notebook_config.py /home/$NB_USER/.jupyter/

# Mark the entrypoint.
ENTRYPOINT ["tini", "-g", "--", "/main.sh"]
