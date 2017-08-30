# neuropythy #######################################################################################
A neuroscience library for Python, intended to complement the existing nibabel library.

## Author ##########################################################################################
Noah C. Benson &lt;<nben@nyu.edu>&gt;

## Installation ####################################################################################

The neuropythy library is available on [PyPI](https://pypi.python.org/pypi/neuropythy) and can be
installed via pip:

```bash
pip install neuropythy
```

The dependencies (below) should be installed auotmatically. Alternately, you can check out this
github repository and run setuptools:

```bash
# Clone the repository
git clone https://github.com/noahbenson/neuropythy
# Enter the repo directory
cd neuropythy
# setup the submodules
git submodule init && git submodule update
# Install the library
python setup.py install

```

## Dependencies ####################################################################################

The neuropythy library depends on a few other libraries, all freely available:
 * [numpy](http://numpy.scipy.org/) &ge; 1.2
 * [scipy](http://www.scipy.org/) &ge; 0.7.0
 * [nibabel](https://github.com/nipy/nibabel) &ge; 1.2
 * [pysistence](https://pythonhosted.org/pysistence/) &ge; 0.4.0
 * [py4j](https://www.py4j.org/) &ge; 0.9

These libaries should be installed automatically for you if you use pip or setuptools (see above),
and they must be found on your PYTHONPATH in order to use neuropythy.

### Optional Dependencies

 * **[python-igraph &ge; 0.7.1](http://igraph.org/python/)**. You can only create graph objects from
   cortical surface meshes if you have installed the [python-igraph](http://igraph.org/python/)
   library; it is not required otherwise.
 * **[Matplotlib &ge; 1.5.3](http://matplitlib.org/)**. A few functions for plotting cortical maps
   are defined in the neuropythy.cortex package. These are not defined if matplotlib is not imported
   successfully. The functions in question are cortex_plot, vertex_angle_color, vertex_eccen_color,
   and a few helper functions.
 * **[Java &ge; 1.8](http://www.oracle.com/technetwork/java/javase/downloads/index.html)**.
   The registration algorithm employed by the register_retinotopy command is performed by
   a Java library embedded in the neuropythy Python library. This library is the
   [nben](https://github.com/noahbenson/nben) library, and is included as a submodule of this
   GitHub repository, found in neuropythy/lib/nben; a standalone jar-file is also distributed as
   part of the PyPI neuropythy distribution. However, in order for the Py4j library, which allows
   Python to execute Java routines, to use this jar-file, you must have a working version of Java
   installed; accordingly, the register_retinotopy command is only available if you have Java
   installed and working. For help getting Java configured to work with Py4j, see the
   [Py4j installation page](https://www.py4j.org/install.html).


## Commands ########################################################################################

Currently Neuropythy is undergoing rapid development, but to get started, the neuropythy.commands
package contains functions that run command-interfaces for the various routines included.  Any of
these commands may be invoked by calling Neuropythy's main function and passing the name of the
command as the first argument followed by any additional command arguments. The argument --help may
be passed for further information about each command.

 * **surface_to_ribbon**. This command projects data on the cortical surface into a volume the same
   orientation as the subject's mri/orig.mgz file. The algorithm used tends to be much cleaner than
   that used by FreeSurfer's mri_surf2vol.
 * **benson14_retinotopy**. This command applies the anatomically-defined template of retinotopy
   described by Benson *et al.* (2014; see **References** below) to a subject. Note that the
   template applied is not actually the template shown in the paper but is a similar updated
   version.
 * **register_retinotopy**. This command fits a retinotopic model of V1, V2, and V3 to retinotopy
   data for a subject and saves the predicted retinotopic maps that result. Running this command
   requires some retinotopic measurements that have already been transferred to the subject's
   FreeSurfer surface. These files can either be specified on the command line (see the
   `register_retinotopy --help` documentation) or placed in the subject's `surf/` directory and
   named as follows:
    * lh.prf_angle.mgz (subject's LH polar angle, 0-180 degrees refers to UVM -> RHM -> LVM)
    * rh.prf_angle.mgz (subject's RH polar angle, 0-180 degrees refers to UVM -> LHM -> RVM)
    * lh.prf_eccen.mgz (subject's LH eccentricity, in degrees)
    * rh.prf_eccen.mgz (subject's RH eccentricity, in degrees)
    * lh.prf_vexpl.mgz (the varaince explained of each vertex's pRF solution for the LH; 0-1 values)
    * rh.prf_vexpl.mgz (the varaince explained of each vertex's pRF solution for the RH; 0-1 values)

   To be clear, both the left and right hemispheres' angle files should specify the polar angle in
   positive degrees; for the right hemisphere, positive refers to the left visual hemi-field; for
   the left hemisphere, positive values refer to the right visual hemi-field. In both cases, 0
   represents the upper vertical meridian and 180 represents the lower vertical meridian. Each MGZ
   file should contain a 1x1xn (or 1x1x1xn) volume where n is the number of vertices in the relevant
   hemisphere and the vertex ordering is that used by FreeSurfer.
   

If neuropythy is installed on your machine, then you can execute a command like so:

```bash
> python -m neuropythy.__main__ surface_to_ribbon --help
> python -m neuropythy.__main__ benson14_retinotopy bert
```


## Docker ##########################################################################################

There is a Docker containing Neuropythy that can be used to run the Neuropythy commands quite easily
without installing Neuropythy itself. If you have [Docker](https://www.docker.com/) installed, you
can use Neuropythy as follows:

```bash
# If your FreeSurfer subject's directory is /data/subjects and you want to
# apply the Benson2014 template to a subject bert:
docker run nben/neuropythy -ti --rm -v /data/subjects:/subjects \
           benson14_retinotopy bert
```


## References ######################################################################################

 * Benson NC, Butt OH, Brainard DH, Aguirre GK (**2014**) Correction of distortion in flattened
   representations of the cortical surface allows prediction of V1-V3 functional organization from
   anatomy. *PLoS Comput. Biol.* **10**(3):e1003538.
   doi:[10.1371/journal.pcbi.1003538](https://dx.doi.org/10.1371/journal.pcbi.1003538).
   PMC:[3967932](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3967932/).
 * Benson NC, Butt OH, Datta R, Radoeva PD, Brainard DH, Aguirre GK (**2012**) The retinotopic
   organization of striate cortex is well predicted by surface topology. *Curr. Biol.*
   **22**(21):2081-5. doi:[10.1016/j.cub.2012.09.014](https://dx.doi.org/10.1016/j.cub.2012.09.014).
   PMC:[3494819](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3494819/).


## License #########################################################################################

This README file is part of the Neuropythy library.

The Neuropythy library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
