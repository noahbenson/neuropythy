# neuropythy #######################################################################################

This Docker contains an installation of the Neuropythy library for Python. For more information
about the Neuropythy library, visit the following website:
 * <https://github.com/noahbenson/neuropythy>

## Author ##########################################################################################
Noah C. Benson &lt;<nben@nyu.edu>&gt;

## Usage ###########################################################################################

This Docker packages the Neuropythy library and can be used to run Neuropythy library commands.
These commands are described on the github repository page and in the code itself. When running a
command, the Docker will look to see if you have mounted a FreeSurfer subjects directory into the
Docker's filesystem at /subjects or /freesurfer_subjects and will automatically use these. For
example:

```bash
docker run -ti --rm -v /my/local/fs_subjects_dir:/subjects benson14_retinotopy bert
```

## License #########################################################################################

This README file is part of the Neuropythy libraray Docker.

The Neuropythy library Docker is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software Foundation, either version
3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not,
see <http://www.gnu.org/licenses/>.
