####################################################################################################
# neuropythy/java/__init__.py
# The code that manages the neuropythy link to the JVM.
# By Noah C. Benson

import numpy   as np
import scipy   as sp
import numbers as num
import os, sys, gzip

from array  import array
from ..util import library_path

# Java start:
_java_port = None
_java = None

def _init_registration():
    from py4j.java_gateway import (launch_gateway, JavaGateway, GatewayParameters)
    global _java, _java_port
    if _java is not None: return
    _java_port = launch_gateway(
        classpath=os.path.join(library_path(), 'nben', 'target', 'nben-standalone.jar'),
        javaopts=['-Xmx2g'],
        die_on_exit=True)
    _java = JavaGateway(gateway_parameters=GatewayParameters(port=_java_port))

def java_link():
    if _java is None: _init_registration()
    return _java

def serialize_numpy(m, t):
    '''
    serialize_numpy(m, type) converts the numpy array m into a byte stream that can be read by the
    nben.util.Py4j Java class. The function assumes that the type of the array needn't be encoded
    in the bytearray itself. The bytearray will begin with an integer, the number of dimensions,
    followed by that number of integers (the dimension sizes themselves) then the bytes of the
    array, flattened.
    The argument type gives the type of the array to be transferred and must be 'i' for integer or
    'd' for double (or any other string accepted by array.array()).
    '''
    # Start with the header: <number of dimensions> <dim1-size> <dim2-size> ...
    header = array('i', [len(m.shape)] + list(m.shape))
    # Now, we can do the array itself, just flattened
    body = array(t, m.flatten().tolist())
    # Wrap bytes if necessary...
    if sys.byteorder != 'big':
        header.byteswap()
        body.byteswap()
    # And return the result:
    return bytearray(header.tostring() + body.tostring())

def to_java_doubles(m):
    '''
    to_java_doubles(m) yields a java array object for the vector or matrix m.
    '''
    global _java
    if _java is None: _init_registration()
    m = np.asarray(m)
    dims = len(m.shape)
    if dims > 2: raise ValueError('1D and 2D arrays supported only')
    bindat = serialize_numpy(m, 'd')
    return (_java.jvm.nben.util.Numpy.double2FromBytes(bindat) if dims == 2
            else _java.jvm.nben.util.Numpy.double1FromBytes(bindat))

def to_java_ints(m):
    '''
    to_java_ints(m) yields a java array object for the vector or matrix m.
    '''
    global _java
    if _java is None: _init_registration()
    m = np.asarray(m)
    dims = len(m.shape)
    if dims > 2: raise ValueError('1D and 2D arrays supported only')
    bindat = serialize_numpy(m, 'i')
    return (_java.jvm.nben.util.Numpy.int2FromBytes(bindat) if dims == 2
            else _java.jvm.nben.util.Numpy.int1FromBytes(bindat))

def to_java_array(m):
    '''
    to_java_array(m) yields to_java_ints(m) if m is an array of integers and to_java_doubles(m) if
    m is anything else. The numpy array m is tested via numpy.issubdtype(m.dtype, numpy.int64).
    '''
    if not hasattr(m, '__iter__'): return m
    m = np.asarray(m)
    if np.issubdtype(m.dtype, np.dtype(int).type) or all(isinstance(x, num.Integral) for x in m):
        return to_java_ints(m)
    else:
        return to_java_doubles(m)
