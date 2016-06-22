####################################################################################################
# neuropythy/registration/models.py
# Importing and interpreting of flat mesh models for registration.
# By Noah C. Benson

import numpy as np
import numpy.linalg
import scipy as sp
import scipy.spatial as space
import os, math
from pysistence import make_dict

from neuropythy.immutable import Immutable
import neuropythy.geometry as geo

class FlatMeshModel:
    '''
    FlatMeshModel is a class that handles 2D models constructed from values on triangle meshes. The
    mesh effectively represents a tesselation of some space (e.g., the cortical surface) and the 
    values can be interpolated at any point to give a continuous 2D field or set of fields.
    '''

    def __init__(self, triangles, points, fields):
        self.triangles = triangles if triangles.shape[1] == 3 else triangles.T
        self.coordinates = points if points.shape[0] == 2 else points.T
        self.vertex_count = self.coordinates.shape[1]
        self.fields = fields if fields.shape[1] == self.vertex_count else fields.T

    def __repr__(self):
        return 'FlatMeshModel(<%d triangles>, <%d points>)' % (self.triangles.shape[0],
                                                               self.vertex_count)

    def interpolate(self, data, n_jobs=1):
        '''
        '''
        return None

