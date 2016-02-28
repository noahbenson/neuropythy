####################################################################################################
# neuropythy/geometry/__init__.py
# This file defines common rotation functions that are useful with cortical mesh spheres, such as
# those produced with FreeSurfer.

import numpy as np
import math

def normalize(u):
    '''
    normalize(u) yields a vetor with the same direction as u but unit length, or, if u has zero
    length, yields u.
    '''
    unorm = np.linalg.norm(u)
    if unorm == 0:
        return u
    return np.asarray(u) / unorm

def vector_angle_cos(u, v):
    '''
    vector_angle_cos(u, v) yields the cosine of the angle between the two vectors u and v.
    '''
    return np.dot(np.asarray(u) / np.linalg.norm(u), np.asarray(v) / np.linalg.norm(v))

def vector_angle(u, v, direction=None):
    '''
    vector_angle(u, v) yields the angle between the two vectors u and v. The optional argument 
    direction is by default None, which specifies that the smallest possible angle between the
    vectors be reported; if the vectors u and v are 2D vectors and direction parameters True and
    False specify the clockwise or counter-clockwise directions, respectively; if the vectors are
    3D vectors, then direction may be a 3D point that is not in the plane containing u, v, and the
    origin, and it specifies around which direction (u x v or v x u) the the counter-clockwise angle
    from u to v should be reported (the cross product vector that has a positive dot product with
    the direction argument is used as the rotation axis).
    '''
    if direction is None:
        return np.arccos(vector_angle_cos(u, v))
    elif direction is True:
        return np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])
    elif direction is False:
        return np.arctan2(u[1], u[0]) - np.arctan2(v[1], v[0])
    else:
        axis1 = normalize(u)
        axis2 = normalize(np.cross(u, v))
        if np.dot(axis2, direction) < 0:
            axis2 = -axis2
        return np.arctan2(np.dot(axis2, v), np.dot(axis1, v))

def rotation_matrix_3D(u, th):
    """
    rotation_matrix_3D(u, t) yields a 3D numpy matrix that rotates any vector about the axis u
    t radians counter-clockwise.
    """
    # normalize the axis:
    u = normalize(u)
    # We use the Euler-Rodrigues formula;
    # see https://en.wikipedia.org/wiki/Euler-Rodrigues_formula
    a = math.cos(0.5 * th)
    s = math.sin(0.5 * th)
    (b, c, d) = -s * u
    (a2, b2, c2, d2) = (a*a, b*b, c*c, d*d)
    (bc, ad, ac, ab, bd, cd) = (b*c, a*d, a*c, a*b, b*d, c*d)
    return np.array([[a2 + b2 - c2 - d2, 2*(bc + ad),         2*(bd - ac)],
                     [2*(bc - ad),       a2 + c2 - b2 - d2,   2*(cd + ab)],
                     [2*(bd + ac),       2*(cd - ab),         a2 + d2 - b2 - c2]])

def rotation_matrix_2D(th):
    '''
    rotation_matrix_2D(th) yields a 2D numpy rotation matrix th radians counter-clockwise about the
    origin.
    '''
    s = np.sin(th)
    c = np.cos(th)
    return np.array([[c, -s], [s, c]])

# construct a rotation matrix of vector u to vector b around center
def alignment_matrix_3D(u, v):
    '''
    alignment_matrix_3D(u, v) yields a 3x3 numpy array that rotates the vector u to the vector v
    around the origin.
    '''
    # normalize both vectors:
    u = normalize(u)
    v = normalize(v)
    # get the cross product of u cross v
    w = np.cross(u, v)
    # the angle we need to rotate
    th = vector_angle(u, v)
    # we rotate around this vector by the angle between them
    return rotation_matrix_3D(w, th)

def alignment_matrix_2D(u, v):
    '''
    alignment_matrix_2D(u, v) yields a 2x2 numpy array that rotates the vector u to the vector v
    around the origin.
    '''
    return rotation_matrix_2D(vector_angle_2D(u, v, direction=True))
