# neuropythy #######################################################################################
A neuroscience library for Python, intended to complement the existing nibabel library.

## Author ##########################################################################################
Noah C. Benson &lt;<nben@nyu.edu>&gt;

## Instalaltion ####################################################################################

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
# Install the library
python setup.py install

```

## Dependencies ####################################################################################

The neuropythy library depends on two other libraries, all freely available:
 * [numpy](http://numpy.scipy.org/) >= 1.2
 * [scipy](http://www.scipy.org/) >= 0.7.0
 * [nibabel](https://github.com/nipy/nibabel) >= 1.2
 * [pysistence](https://pythonhosted.org/pysistence/) >= 0.4.0
 * [python-igraph](http://igraph.org/python/) >= 0.7.1 (optional)
 * [py4j](https://www.py4j.org/) >= 0.9 (optional)

These libaries should be installed automatically for you if you use pip or setuptools (see above),
and they must be found on your PYTHONPATH in order to use neuropythy.

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
