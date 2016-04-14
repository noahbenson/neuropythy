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
    vector_angle_cos(u, v) yields the cosine of the angle between the two vectors u and v. If u
    or v (or both) is a (d x n) matrix of n vectors, the result will be a length n vector of the
    cosines.
    '''
    return (u * v).sum(0) / np.sqrt((np.asarray(u) ** 2).sum(0) * (np.asarray(v) ** 2).sum(0))

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

def line_intersection_2D(((x1,y1),(x2,y2)), ((x3,y3),(x4,y4))):
    '''
    line_intersection((a, b), (c, d)) yields the intersection point between the lines that pass
    through the given pairs of points.
    '''
    dx12 = (x1 - x2)
    dx34 = (x3 - x4)
    dy12 = (y1 - y2)
    dy34 = (y3 - y4)
    denom = dx12*dy34 - dy12*dx34
    q12 = (x1*y2 - y1*x2) / denom
    q34 = (x3*y4 - y3*x4) / denom
    return (q12*dx34 - q34*dx12, q12*dy34 - q34*dy12)

def triangle_area(a,b,c):
    '''
    triangle_area(a, b, c) yields the area of the triangle whose vertices are given by the points a,
    b, and c.
    '''
    if len(a) == 2:
        return 0.5*(a[0]*(b[1] - c[1]) + b[0]*(c[1] - a[1]) + c[0]*(a[1] - b[1]))
    else:
        mtx = alignment_matrix_3D(np.cross(np.asarray(b) - a, np.asarray(c) - a), [0,0,1])[0:1]
        return triangle_area(np.dot(mtx, a), np.dot(mtx, b), np.dot(mtx, c))

def triangle_address(fx, pt):
    '''
    triangle_address(FX, P) yields an address coordinate (t,r) for the point P in the triangle
    defined by the (3 x d)-sized coordinate matrix FX, in which each row of the matrix is the
    d-dimensional vector representing the respective triangle vertx for triangle [A,B,C]. The
    resulting coordinates (t,r) (0 <= t <= 1, 0 <= r <= 1) address the point P such that, if t gives
    the fraction of the angle from vector AB to vector AC that is made by the angle between vectors
    AB and AP, and r gives the fraction ||AP||/||AR|| where R is the point of intersection between
    lines AP and BC. If P is a (d x n)-sized matrix of points, then a (2 x n) matrix of addresses
    is returned.
    '''
    fx = np.asarray(fx)
    pt = np.asarray(pt)
    # The triangle vectors...
    ab = fx[1] - fx[0]
    ac = fx[2] - fx[0]
    bc = fx[2] - fx[1]
    ap = np.asarray([pt_i - a_i for (pt_i, a_i) in zip(pt, fx[0])])
    # get the unnormalized distance...
    r = np.sqrt((ap ** 2).sum(0))
    # now we can find the angle...
    unit = 1 - r.astype(bool)
    t0 = vector_angle(ab, ac)
    t = vector_angle(ap + [ab_i * unit for ab_i in ab], ab)
    sint = np.sin(t)
    sindt = np.sin(t0 - t)
    # finding r0 is tricker--we use this fancy formula based on the law of sines
    q0 = np.sqrt((bc ** 2).sum(0))          # B->C distance
    beta = vector_angle(-ab, bc)            # Angle at B
    sinGamma = np.sin(math.pi - beta - t0)
    sinBeta  = np.sin(beta)
    r0 = q0 * sinBeta * sinGamma / (sinBeta * sindt + sinGamma * sint)
    return np.asarray([t/t0, r/r0])

def triangle_unaddress(fx, tr):
    '''
    triangle_unaddress(FX, tr) yields the point P, inside the reference triangle given by the 
    (3 x d)-sized coordinate matrix FX, that is addressed by the address coordinate tr, which may
    either be a 2-d vector or a (2 x n)-sized matrix.
    '''
    fx = np.asarray(fx)
    tr = np.asarray(tr)
    # the triangle vectors...
    ab = fx[1] - fx[0]
    ac = fx[2] - fx[0]
    bc = fx[2] - fx[1]
    return np.asarray([ax + tr[1]*(abx + tr[0]*bcx) for (ax, bcx, abx) in zip(fx[0], bc, ab)])
