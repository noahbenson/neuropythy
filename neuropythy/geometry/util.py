####################################################################################################
# neuropythy/geometry/util.py
# This file defines common rotation functions that are useful with cortical mesh spheres, such as
# those produced with FreeSurfer.
# By Noah C. Benson

import numpy as np
import math

from ..util import czdivide

def normalize(u):
    '''
    normalize(u) yields a vetor with the same direction as u but unit length, or, if u has zero
    length, yields u.
    '''
    u = np.asarray(u)
    unorm = np.sqrt(np.sum(u**2, axis=0))
    z = np.isclose(unorm, 0)
    c = np.logical_not(z) / (unorm + z)
    return u * c

def vector_angle_cos(u, v):
    '''
    vector_angle_cos(u, v) yields the cosine of the angle between the two vectors u and v. If u
    or v (or both) is a (d x n) matrix of n vectors, the result will be a length n vector of the
    cosines.
    '''
    u = np.asarray(u)
    v = np.asarray(v)
    return (u * v).sum(0) / np.sqrt((u ** 2).sum(0) * (v ** 2).sum(0))

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

def spherical_distance(pt0, pt1):
    '''
    spherical_distance(a, b) yields the angular distance between points a and b, both of which
      should be expressed in spherical coordinates as (longitude, latitude).
    If a and/or b are (2 x n) matrices, then the calculation is performed over all columns.
    The spherical_distance function uses the Haversine formula; accordingly it may suffer from
    rounding errors in the case of nearly antipodal points.
    '''
    dtheta = pt1[0] - pt0[0]
    dphi   = pt1[1] - pt0[1]
    a = np.sin(dphi/2)**2 + np.cos(pt0[1]) * np.cos(pt1[1]) * np.sin(dtheta/2)**2
    return 2 * np.arcsin(np.sqrt(a))
    
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
    return rotation_matrix_2D(vector_angle(u, v, direction=True))

def point_on_line(ab, c):
    '''
    point_on_line((a,b), c) yields True if point x is on line (a,b) and False otherwise.
    '''
    (a,b) = ab
    abc = [np.asarray(u) for u in (a,b,c)]
    if any(len(u.shape) == 2 for u in abc): (a,b,c) = [np.reshape(u,(len(u),-1)) for u in abc]
    else:                                   (a,b,c) = abc
    vca = a - c
    vcb = b - c
    uba = czdivide(vba, np.sqrt(np.sum(vba**2, axis=0)))
    uca = czdivide(vca, np.sqrt(np.sum(vca**2, axis=0)))
    return (np.isclose(np.sqrt(np.sum(vca**2, axis=0)), 0) |
            np.isclose(np.sqrt(np.sum(vcb**2, axis=0)), 0) |
            np.isclose(np.abs(np.sum(uba*uca, axis=0)), 1))

def point_on_segment(ab, c):
    '''
    point_on_segment((a,b), c) yields True if point x is on segment (a,b) and False otherwise. Note
    that this differs from point_in_segment in that a point that if c is equal to a or b it is
    considered 'on' but not 'in' the segment.
    '''
    (a,c) = ac
    abc = [np.asarray(u) for u in (a,b,c)]
    if any(len(u.shape) > 1 for u in abc): (a,b,c) = [np.reshape(u,(len(u),-1)) for u in abc]
    else:                                  (a,b,c) = abc
    vab = b - a
    vbc = c - b
    vac = c - a
    dab = np.sqrt(np.sum(vab**2, axis=0))
    dbc = np.sqrt(np.sum(vbc**2, axis=0))
    dac = np.sqrt(np.sum(vac**2, axis=0))
    return np.isclose(dab + dbc, dac)
def point_in_segment(ac, b):
    '''
    point_in_segment((a,b), c) yields True if point x is in segment (a,b) and False otherwise. Note
    that this differs from point_on_segment in that a point that if c is equal to a or b it is
    considered 'on' but not 'in' the segment.
    '''
    (a,c) = ac
    abc = [np.asarray(u) for u in (a,b,c)]
    if any(len(u.shape) > 1 for u in abc): (a,b,c) = [np.reshape(u,(len(u),-1)) for u in abc]
    else:                                  (a,b,c) = abc
    vab = b - a
    vbc = c - b
    vac = c - a
    dab = np.sqrt(np.sum(vab**2, axis=0))
    dbc = np.sqrt(np.sum(vbc**2, axis=0))
    dac = np.sqrt(np.sum(vac**2, axis=0))
    return np.isclose(dab + dbc, dac) & ~np.isclose(dac,dab) & ~np.isclose(dac,dbc)

def lines_colinear(ab, cd):
    '''
    liness_colinear((a, b), (c, d)) yields True if the lines containing points (a,b) and points (c,d)
    are colinear and false otherwise. All of a, b, c, and d must be (x,y) coordinates or 2xN (x,y)
    coordinate matrices, or (x,y,z) or 3xN matrices.
    '''
    # simple check: a and b must be on (c,d)
    return point_on_line(ab, cd[0]) & point_on_line(ab, cd[1])

def segments_colinear(ab, cd):
    '''
    segments_colinear_2D((a, b), (c, d)) yields True if either a or b is on the line segment (c,d) or
    if c or d is on the line segment (a,b) and the lines are colinear; otherwise yields False. All
    of a, b, c, and d must be (x,y) coordinates or 2xN (x,y) coordinate matrices, or (x,y,z) or 3xN
    matrices.
    '''
    (a,b) = ab
    (c,d) = cd
    ss = [point_on_segment(ab, c), point_on_segment(ab, d),
          point_on_segment(cd, a), point_on_segment(cd, b)]
    return np.sum(ss, axis=0) > 1

def points_close(a,b):
    '''
    points_close(a,b) yields True if points a and b are close to each other and False otherwise.
    '''
    (a,b) = [np.asarray(u) for u in (a,b)]
    if len(a.shape) == 2 or len(b.shape) == 2: (a,b) = [np.reshape(u,(len(u),-1)) for u in (a,b)]
    return np.isclose(np.sqrt(np.sum((a - b)**2, axis=0)), 0)

def segments_overlapping(ab, cd):
    '''
    segments_overlapping((a, b), (c, d)) yields True if the line segments (a,b) and (c,d) are both
    colinear and have a non-finite overlap. If (a,b) and (c,d) touch at a single point, they are not
    considered overlapping.
    '''
    (a,b) = ab
    (c,d) = cd
    ss = [point_in_segment(ab, c), point_in_segment(ab, d),
          point_in_segment(cd, a), point_in_segment(cd, b)]
    return (~(points_close(a,b) | points_close(c,d)) & 
            ((np.sum(ss, axis=0) > 1) |
             (points_close(a,c) & points_close(b,d)) |
             (points_close(a,d) & points_close(b,c))))

def line_intersection_2D(abarg, cdarg):
    '''
    line_intersection((a, b), (c, d)) yields the intersection point between the lines that pass
    through the given pairs of points. If any lines are parallel, (numpy.nan, numpy.nan) is
    returned; note that a, b, c, and d can all be 2 x n matrices of x and y coordinate row-vectors.
    '''
    ((x1,y1),(x2,y2)) = abarg
    ((x3,y3),(x4,y4)) = cdarg
    dx12 = (x1 - x2)
    dx34 = (x3 - x4)
    dy12 = (y1 - y2)
    dy34 = (y3 - y4)
    denom = dx12*dy34 - dy12*dx34
    unit = np.isclose(denom, 0)
    if unit is True: return (np.nan, np.nan)
    denom = unit + denom
    q12 = (x1*y2 - y1*x2) / denom
    q34 = (x3*y4 - y3*x4) / denom
    xi = q12*dx34 - q34*dx12
    yi = q12*dy34 - q34*dy12
    if   unit is False: return (xi, yi)
    elif unit is True:  return (np.nan, np.nan)
    else:
        xi = np.asarray(xi)
        yi = np.asarray(yi)
        xi[unit] = np.nan
        yi[unit] = np.nan
        return (xi, yi)

def segment_intersection_2D(p12arg, p34arg):
    '''
    segment_intersection((a, b), (c, d)) yields the intersection point between the line segments
    that pass from point a to point b and from point c to point d. If there is no intersection
    point, then (numpy.nan, numpy.nan) is returned.
    '''
    (p1,p2) = p12arg
    (p3,p4) = p34arg
    pi = np.asarray(line_intersection_2D(p12arg, p34arg))
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    p4 = np.asarray(p4)
    u12 = p2 - p1
    u34 = p4 - p3
    cfn = lambda px,iis: (px if iis is None or len(px.shape) == 1 or px.shape[1] == len(iis) else
                          px[:,iis])
    dfn = lambda a,b:     a[0]*b[0] + a[1]*b[1]
    sfn = lambda a,b:     ((a-b)                 if len(a.shape) == len(b.shape) else
                           (np.transpose([a])-b) if len(a.shape) <  len(b.shape) else
                           (a - np.transpose([b])))
    fn  = lambda px,iis:  (1 - ((dfn(cfn(u12,iis), sfn(         px, cfn(p1,iis))) > 0) *
                                (dfn(cfn(u34,iis), sfn(         px, cfn(p3,iis))) > 0) *
                                (dfn(cfn(u12,iis), sfn(cfn(p2,iis),          px)) > 0) *
                                (dfn(cfn(u34,iis), sfn(cfn(p4,iis),          px)) > 0)))
    if len(pi.shape) == 1:
        if not np.isfinite(pi[0]): return (np.nan, np.nan)
        bad = fn(pi, None)
        return (np.nan, np.nan) if bad else pi
    else:
        nonpar = np.where(np.isfinite(pi[0]))[0]
        bad = fn(cfn(pi, nonpar), nonpar)
        (xi,yi) = pi
        bad = nonpar[np.where(bad)[0]]
        xi[bad] = np.nan
        yi[bad] = np.nan
        return (xi,yi)

def lines_touch_2D(ab, cd):
    '''
    lines_touch_2D((a,b), (c,d)) is equivalent to lines_colinear((a,b), (c,d)) |
    numpy.isfinite(line_intersection_2D((a,b), (c,d))[0])
    '''
    return lines_colinear(ab, cd) | np.isfinite(line_intersection_2D(ab, cd)[0])

def segments_touch_2D(ab, cd):
    '''
    segmentss_touch_2D((a,b), (c,d)) is equivalent to segments_colinear((a,b), (c,d)) |
    numpy.isfinite(segment_intersection_2D((a,b), (c,d))[0])
    '''
    return segments_colinear(ab, cd) | np.isfinite(segment_intersection_2D(ab, cd)[0])

def line_segment_intersection_2D(p12arg, p34arg):
    '''
    line_segment_intersection((a, b), (c, d)) yields the intersection point between the line
    passing through points a and b and the line segment that passes from point c to point d. If
    there is no intersection point, then (numpy.nan, numpy.nan) is returned.
    '''
    (p1,p2) = p12arg
    (p3,p4) = p34arg
    pi = np.asarray(line_intersection_2D(p12arg, p34arg))
    p3 = np.asarray(p3)
    u34 = p4 - p3
    cfn = lambda px,iis: (px if iis is None or len(px.shape) == 1 or px.shape[1] == len(iis) else
                          px[:,iis])
    dfn = lambda a,b:     a[0]*b[0] + a[1]*b[1]
    sfn = lambda a,b:     ((a-b)                 if len(a.shape) == len(b.shape) else
                           (np.transpose([a])-b) if len(a.shape) <  len(b.shape) else
                           (a - np.transpose([b])))
    fn  = lambda px,iis:  (1 - ((dfn(cfn(u34,iis), sfn(         px, cfn(p3,iis))) > 0) *
                                (dfn(cfn(u34,iis), sfn(cfn(p4,iis),          px)) > 0)))
    if len(pi.shape) == 1:
        if not np.isfinite(pi[0]): return (np.nan, np.nan)
        bad = fn(pi, None)
        return (np.nan, np.nan) if bad else pi
    else:
        nonpar = np.where(np.isfinite(pi[0]))[0]
        bad = fn(cfn(pi, nonpar), nonpar)
        (xi,yi) = pi
        bad = nonpar[np.where(bad)[0]]
        xi[bad] = np.nan
        yi[bad] = np.nan
        return (xi,yi)

def triangle_area(a,b,c):
    '''
    triangle_area(a, b, c) yields the area of the triangle whose vertices are given by the points a,
    b, and c.
    '''
    (a,b,c) = [np.asarray(x) for x in (a,b,c)]
    sides = np.sqrt(np.sum([(p1 - p2)**2 for (p1,p2) in zip([b,c,a],[c,a,b])], axis=1))
    s = 0.5 * np.sum(sides, axis=0)
    s = np.clip(s * np.prod(s - sides, axis=0), 0.0, None)
    return np.sqrt(s)

def triangle_normal(a,b,c):
    '''
    triangle_normal(a, b, c) yields the normal vector of the triangle whose vertices are given by
      the points a, b, and c. If the points are 2D points, then 3D normal vectors are still yielded,
      that are always (0,0,1) or (0,0,-1). This function auto-threads over matrices, in which case
      they must be in equivalent orientations, and the result is returned in whatever orientation
      they are given in. In some cases, the intended orientation of the matrices is ambiguous (e.g.,
      if a, b, and c are 2 x 3 matrices), in which case the matrix is always assumed to be given in
      (dims x vertices) orientation.
    '''
    (a,b,c) = [np.asarray(x) for x in (a,b,c)]
    if len(a.shape) == 1 and len(b.shape) == 1 and len(c.shape) == 1:
        return triangle_normal(*[np.transpose([x]) for x in (a,b,c)])[:,0]
    (a,b,c) = [np.transpose([x]) if len(x.shape) == 1 else x for x in (a,b,c)]
    # find a required number of dimensions, if possible
    if a.shape[0] in (2,3):
        dims = a.shape[0]
        tx = True
    else:
        dims = a.shape[1]
        (a,b,c) = [x.T for x in (a,b,c)]
        tx = False
    n = (a.shape[1] if a.shape[1] != 1 else b.shape[1] if b.shape[1] != 1 else
         c.shape[1] if c.shape[1] != 1 else 1)
    if dims == 2:
        (a,b,c) = [np.vstack((x, np.zeros((1,n)))) for x in (a,b,c)]
    ab = normalize(b - a)
    ac = normalize(c - a)
    res = np.cross(ab, ac, axisa=0, axisb=0)
    return res.T if tx else res

def cartesian_to_barycentric_3D(tri, xy):
    '''
    cartesian_to_barycentric_3D(tri,xy) is identical to cartesian_to_barycentric_2D(tri,xy) except
    it works on 3D data. Note that if tri is a 3 x 3 x n, a 3 x n x 3 or an n x 3 x 3 matrix, the
    first dimension must always be the triangle vertices and the second 3-sized dimension must be
    the (x,y,z) coordinates.
    '''
    xy = np.asarray(xy)
    tri = np.asarray(tri)
    if len(xy.shape) == 1:
        return cartesian_to_barycentric_3D(np.transpose(np.asarray([tri]), (1,2,0)),
                                           np.asarray([xy]).T)[:,0]
    xy = xy if xy.shape[0] == 3 else xy.T
    if tri.shape[0] == 3:
        tri = tri if tri.shape[1] == 3 else np.transpose(tri, (0,2,1))
    elif tri.shape[1] == 3:
        tri = tri.T if tri.shape[0] == 3 else np.transpose(tri, (1,2,0))
    elif tri.shape[2] == 3:
        tri = np.transpose(tri, (2,1,0) if tri.shape[1] == 3 else (2,0,1))
    if tri.shape[0] != 3 or tri.shape[1] != 3:
        raise ValueError('Triangle array did not have dimensions of sizes 3 and 3')
    if xy.shape[0] != 3:
        raise ValueError('coordinate matrix did not have a dimension of size 3')
    if tri.shape[2] != xy.shape[1]:
        raise ValueError('number of triangles and coordinates must match')
    # The algorithm here is borrowed from this stack-exchange post:
    # http://gamedev.stackexchange.com/questions/23743
    # in which it is attributed to Christer Ericson's book Real-Time Collision Detection.
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = xy - tri[0]
    d00 = np.sum(v0 * v0, axis=0)
    d01 = np.sum(v0 * v1, axis=0)
    d11 = np.sum(v1 * v1, axis=0)
    d20 = np.sum(v2 * v0, axis=0)
    d21 = np.sum(v2 * v1, axis=0)
    den = d00*d11 - d01*d01
    zero = np.isclose(den, 0)
    unit = 1 - zero
    den += zero
    l2 = unit * (d11 * d20 - d01 * d21) / den
    l3 = unit * (d00 * d21 - d01 * d20) / den
    return np.asarray([1.0 - l2 - l3, l2])
    
def cartesian_to_barycentric_2D(tri, xy):
    '''
    cartesian_to_barycentric_2D(tri, xy) yields a (2 x n) barycentric coordinate matrix 
    (or just a tuple if xy is a single (x, y) coordinate) of the first two barycentric coordinates
    for the triangle coordinate array tri. The array tri should be 3 (vertices) x 2 (coordinates) x
    n (triangles) unless xy is a tuple, in which case it should be a (3 x 2) matrix.
    '''
    xy = np.asarray(xy)
    tri = np.asarray(tri)
    if len(xy.shape) == 1:
        return cartesian_to_barycentric_2D(np.transpose(np.asarray([tri]), (1,2,0)),
                                           np.asarray([xy]).T)[:,0]
    xy = xy if xy.shape[0] == 2 else xy.T
    if tri.shape[0] == 3:
        tri = tri if tri.shape[1] == 2 else np.transpose(tri, (0,2,1))
    elif tri.shape[1] == 3:
        tri = tri.T if tri.shape[0] == 2 else np.transpose(tri, (1,2,0))
    elif tri.shape[2] == 3:
        tri = np.transpose(tri, (2,1,0) if tri.shape[1] == 2 else (2,0,1))
    if tri.shape[0] != 3 or tri.shape[1] != 2:
        raise ValueError('Triangle array did not have dimensions of sizes 3 and 2')
    if xy.shape[0] != 2:
        raise ValueError('coordinate matrix did not have a dimension of size 2')
    if tri.shape[2] != xy.shape[1]:
        raise ValueError('number of triangles and coordinates must match')
    # Okay, everything's the right shape...
    (x,y) = xy
    ((x1,y1), (x2,y2), (x3,y3)) = tri
    x_x3  = x  - x3
    x1_x3 = x1 - x3
    x3_x2 = x3 - x2
    y_y3  = y  - y3
    y1_y3 = y1 - y3
    y2_y3 = y2 - y3
    num1 = (y2_y3*x_x3  + x3_x2*y_y3)
    num2 = (-y1_y3*x_x3 + x1_x3*y_y3)
    den  = (y2_y3*x1_x3 + x3_x2*y1_y3)
    zero = np.isclose(den, 0)
    den += zero
    unit = 1 - zero
    l1 = unit * num1 / den
    l2 = unit * num2 / den
    return np.asarray((l1, l2))

def barycentric_to_cartesian(tri, bc):
    '''
    barycentric_to_cartesian(tri, bc) yields the d x n coordinate matrix of the given barycentric
    coordinate matrix (also d x n) bc interpolated in the n triangles given in the array tri. See
    also cartesian_to_barycentric. If tri and bc represent one triangle and coordinate, then just
    the coordinate and not a matrix is returned. The value d, dimensions, must be 2 or 3.
    '''
    bc = np.asarray(bc)
    tri = np.asarray(tri)
    if len(bc.shape) == 1:
        return barycentric_to_cartesian(np.transpose(np.asarray([tri]), (1,2,0)),
                                        np.asarray([bc]).T)[:,0]
    bc = bc if bc.shape[0] == 2 else bc.T
    if bc.shape[0] != 2: raise ValueError('barycentric matrix did not have a dimension of size 2')
    n = bc.shape[1]
    # we know how many bc's there are now; lets reorient tri to match with the last dimension as n
    if len(tri.shape) == 2:
        tri = np.transpose([tri for _ in range(n)], (1,2,0))
    # the possible orientations of tri:
    if tri.shape[0] == 3:
        if tri.shape[1] in [2,3] and tri.shape[2] == n:
            pass # default orientation
        elif tri.shape[1] == n and tri.shape[2] in [2,3]:
            tri = np.transpose(tri, (0,2,1))
        else: raise ValueError('could not deduce triangle dimensions')
    elif tri.shape[1] == 3:
        if tri.shape[0] in [2,3] and tri.shape[2] == n:
            tri = np.transpose(tri, (1,0,2))
        elif tri.shape[0] == n and tri.shape[2] in [2,3]:
            tri = np.transpose(tri, (1,2,0))
        else: raise ValueError('could not deduce triangle dimensions')
    elif tri.shape[2] == 3:
        if tri.shape[0] in [2,3] and tri.shape[1] == n:
            tri = np.transpose(tri, (2,0,1))
        elif tri.shape[0] == n and tri.shape[1] in [2,3]:
            tri = np.transpose(tri, (2,1,0))
        else: raise ValueError('could not deduce triangle dimensions')
    else: raise ValueError('At least one dimension of triangles must be 3')
    if tri.shape[0] != 3 or (tri.shape[1] not in [2,3]):
        raise ValueError('Triangle array did not have dimensions of sizes 3 and (2 or 3)')
    if tri.shape[2] != n:
        raise ValueError('number of triangles and coordinates must match')
    (l1,l2) = bc
    (p1, p2, p3) = tri
    l3 = (1 - l1 - l2)
    return np.asarray([x1*l1 + x2*l2 + x3*l3 for (x1,x2,x3) in zip(p1, p2, p3)])
    
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

def point_in_triangle(tri, pt):
    tri = np.asarray(tri)
    pt  = np.asarray(pt)
    if len(tri.shape) == 2 and len(pt.shape) == 1:
        if len(pt) == 2:
            tol = 1e-13
            v0 = tri[2] - tri[0]
            v1 = tri[1] - tri[0]
            v2 = pt - tri[0]
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d02 = np.dot(v0, v2)
            d11 = np.dot(v1, v1)
            d12 = np.dot(v1, v2)
            invDenom = (d00*d11 - d01*d01)
            if np.isclose(invDenom, 0): return False
            s = (d11*d02 - d01*d12) / invDenom
            if (s + tol) < 0 or (s - tol) >= 1: return False
            t = (d00*d12 - d01*d02) / invDenom
            return False if (t + tol) < 0 or (s + t - tol) > 1 else True
        else:
            dp1 = np.dot(pt - tri[0], np.cross(tri[0], tri[1] - tri[0]))
            dp2 = np.dot(pt - tri[1], np.cross(tri[1], tri[2] - tri[1]))
            db3 = np.dot(pt - tri[2], np.cross(tri[2], tri[0] - tri[2]))
            return ((dp1 > 0 or np.isclose(dp1, 0)) and
                    (dp2 > 0 or np.isclose(dp2, 0)) and
                    (dp3 > 0 or np.isclose(dp3, 0)))
    elif len(tri.shape) == 3 and len(pt.shape) == 2:
        if len(pt) != len(tri):
            raise ValueError('the number of triangles and points must be equal')
        if pt.shape[1] == 2:
            tol = 1e-13
            v0 = tri[:,2] - tri[:,0]
            v1 = tri[:,1] - tri[:,0]
            v2 = pt - tri[:,0]
            d00 = np.sum(v0*v0, axis=1)
            d01 = np.sum(v0*v1, axis=1)
            d02 = np.sum(v0*v2, axis=1)
            d11 = np.sum(v1*v1, axis=1)
            d12 = np.sum(v1*v2, axis=1)
            invDenom = (d00*d11 - d01*d01)
            zeros = np.isclose(invDenom, 0)
            invDenom[zeros] = 1.0
            s = (d11*d02 - d01*d12) / invDenom
            t = (d00*d12 - d01*d02) / invDenom
            return ~((s + tol < 0) | (s - tol >= 1) | (t + tol < 0) | (s + t - tol > 1) | zeros)
        else:
            x0 = np.sum((pt - tri[:,0]) * np.cross(tri[:,0], tri[:,1] - tri[:,0], axis=1), axis=1)
            x1 = np.sum((pt - tri[:,1]) * np.cross(tri[:,1], tri[:,2] - tri[:,1], axis=1), axis=1)
            x2 = np.sum((pt - tri[:,2]) * np.cross(tri[:,2], tri[:,0] - tri[:,2], axis=1), axis=1)
            return (((x0 > 0) | np.isclose(x0, 0)) &
                    ((x1 > 0) | np.isclose(x1, 0)) &
                    ((x2 > 0) | np.isclose(x2, 0)))
    elif len(tri.shape) == 3 and len(pt.shape) == 1:
        return point_in_triangle(tri, np.asarray([pt for _ in tri]))
    elif len(tri.shape) == 2 and len(pt.shape) == 1:
        return point_in_triangle(np.asarray([tri for _ in pt]), pt)
    else:
        raise ValueError('triangles and pts do not have parallel shapes')

def det4D(m):
    '''
    det4D(array) yields the determinate of the given matrix array, which may have more than 2
      dimensions, in which case the later dimensions are multiplied and added point-wise.
    '''
    # I just solved this in Mathematica, copy-pasted, and replaced the string '] m' with ']*m':
    # Mathematica code: Det@Table[m[i][j], {i, 0, 3}, {j, 0, 3}]
    return (m[0][3]*m[1][2]*m[2][1]*m[3][0] - m[0][2]*m[1][3]*m[2][1]*m[3][0] -
            m[0][3]*m[1][1]*m[2][2]*m[3][0] + m[0][1]*m[1][3]*m[2][2]*m[3][0] +
            m[0][2]*m[1][1]*m[2][3]*m[3][0] - m[0][1]*m[1][2]*m[2][3]*m[3][0] -
            m[0][3]*m[1][2]*m[2][0]*m[3][1] + m[0][2]*m[1][3]*m[2][0]*m[3][1] +
            m[0][3]*m[1][0]*m[2][2]*m[3][1] - m[0][0]*m[1][3]*m[2][2]*m[3][1] -
            m[0][2]*m[1][0]*m[2][3]*m[3][1] + m[0][0]*m[1][2]*m[2][3]*m[3][1] +
            m[0][3]*m[1][1]*m[2][0]*m[3][2] - m[0][1]*m[1][3]*m[2][0]*m[3][2] -
            m[0][3]*m[1][0]*m[2][1]*m[3][2] + m[0][0]*m[1][3]*m[2][1]*m[3][2] +
            m[0][1]*m[1][0]*m[2][3]*m[3][2] - m[0][0]*m[1][1]*m[2][3]*m[3][2] -
            m[0][2]*m[1][1]*m[2][0]*m[3][3] + m[0][1]*m[1][2]*m[2][0]*m[3][3] +
            m[0][2]*m[1][0]*m[2][1]*m[3][3] - m[0][0]*m[1][2]*m[2][1]*m[3][3] -
            m[0][1]*m[1][0]*m[2][2]*m[3][3] + m[0][0]*m[1][1]*m[2][2]*m[3][3])
def det_4x3(a,b,c,d):
    '''
    det_4x3(a,b,c,d) yields the determinate of the matrix formed the given rows, which may have
      more than 1 dimension, in which case the later dimensions are multiplied and added point-wise.
      The point's must be 3D points; the matrix is given a fourth column of 1s and the resulting
      determinant is of this matrix. 
    '''
    # I just solved this in Mathematica, copy-pasted, and replaced the string '] m' with ']*m':
    # Mathematica code: Det@Table[If[j == 3, 1, i[j]], {i, {a, b, c, d}}, {j, 0, 3}]
    return (a[1]*b[2]*c[0] + a[2]*b[0]*c[1] - a[2]*b[1]*c[0] - a[0]*b[2]*c[1] -
            a[1]*b[0]*c[2] + a[0]*b[1]*c[2] + a[2]*b[1]*d[0] - a[1]*b[2]*d[0] -
            a[2]*c[1]*d[0] + b[2]*c[1]*d[0] + a[1]*c[2]*d[0] - b[1]*c[2]*d[0] -
            a[2]*b[0]*d[1] + a[0]*b[2]*d[1] + a[2]*c[0]*d[1] - b[2]*c[0]*d[1] -
            a[0]*c[2]*d[1] + b[0]*c[2]*d[1] + a[1]*b[0]*d[2] - a[0]*b[1]*d[2] -
            a[1]*c[0]*d[2] + b[1]*c[0]*d[2] + a[0]*c[1]*d[2] - b[0]*c[1]*d[2])
    
def tetrahedral_barycentric_coordinates(tetra, pt):
    '''
    tetrahedral_barycentric_coordinates(tetrahedron, point) yields a list of weights for each vertex
      in the given tetrahedron in the same order as the vertices given. If all weights are 0, then
      the point is not inside the tetrahedron.
    '''
    # I found a description of this algorithm here (Nov. 2017):
    # http://steve.hollasch.net/cgindex/geometry/ptintet.html
    tetra = np.asarray(tetra)
    if tetra.shape[0] != 4:
        if tetra.shape[1] == 4:
            if tetra.shape[0] == 3:
                tetra = np.transpose(tetra, (1,0) if len(tetra.shape) == 2 else (1,0,2))
            else:
                tetra = np.transpose(tetra, (1,2,0))
        elif tetra.shape[1] == 3:
            tetra = np.transpose(tetra, (2,1,0))
        else:
            tetra = np.transpose(tetra, (2,0,1))
    elif tetra.shape[1] != 3:
        tetra = np.transpose(tetra, (0,2,1))
    if pt.shape[0] != 3: pt = pt.T
    # Okay, calculate the determinants...
    d_ = det_4x3(tetra[0], tetra[1], tetra[2], tetra[3])
    d0 = det_4x3(pt,       tetra[1], tetra[2], tetra[3])
    d1 = det_4x3(tetra[0], pt,       tetra[2], tetra[3])
    d2 = det_4x3(tetra[0], tetra[1], pt,       tetra[3])
    d3 = det_4x3(tetra[0], tetra[1], tetra[2], pt)
    s_ = np.sign(d_)
    z_ = np.logical_or(np.any([s_ * si == -1 for si in np.sign([d0,d1,d2,d3])], axis=0),
                       np.isclose(d_,0))
    x_ = np.logical_not(z_)
    d_inv = x_ / (x_ * d_ + z_)
    return np.asarray([d_inv * dq for dq in (d0,d1,d2,d3)])

def point_in_tetrahedron(tetra, pt):
    '''
    point_in_tetrahedron(tetrahedron, point) yields True if the given point is in the given 
      tetrahedron. If either tetrahedron or point (or both) are lists of shapes/points, then this
      calculation is automatically threaded over all the given arguments.
    '''
    bcs = tetrahedral_barycentric_coordinates(tetra, pt)
    return np.logical_not(np.all(np.isclose(bcs, 0), axis=0))

def prism_barycentric_coordinates(tri1, tri2, pt):
    '''
    prism_barycentric_coordinates(tri1, tri2, point) yields a list of weights for each vertex
      in the given tetrahedron in the same order as the vertices given. If all weights are 0, then
      the point is not inside the tetrahedron. The returned weights are (a,b,d) in a numpy array;
      the values a, b, and c are the barycentric coordinates corresponding to the three points of
      the triangles (where c = (1 - a - b) and the value d is the fractional distance (in the range
      [0,1]) of the point between tri1 (d=0) and tri2 (d=1).
    '''
    pt = np.asarray(pt)
    tri1 = np.asarray(tri1)
    tri2 = np.asarray(tri2)
    (tri1,tri2) = [
        (np.transpose(tri, (1,0) if len(tri.shape) == 2 else (2,0,1)) if tri.shape[0] != 3 else
         np.transpose(tri, (0,2,1))                                   if tri.shape[1] != 3 else
         tri)
        for tri in (tri1,tri2)]
    pt = pt.T if pt.shape[0] != 3 else pt
    # get the individual tetrahedron bc coordinates
    bcs1 = tetrahedral_barycentric_coordinates([tri1[0], tri1[1], tri1[2], tri2[0]], pt)
    bcs2 = tetrahedral_barycentric_coordinates([tri1[1], tri1[2], tri2[0], tri2[1]], pt)
    bcs3 = tetrahedral_barycentric_coordinates([tri1[2], tri2[0], tri2[1], tri2[2]], pt)
    bcs4 = tetrahedral_barycentric_coordinates([tri1[0], tri1[1], tri2[0], tri2[1]], pt)
    bcs5 = tetrahedral_barycentric_coordinates([tri1[0], tri1[2], tri2[0], tri2[2]], pt)
    bcs6 = tetrahedral_barycentric_coordinates([tri1[1], tri1[2], tri2[1], tri2[2]], pt)
    bcs = ((bcs1[0] + bcs4[0] + bcs5[0],
            bcs1[1] + bcs2[0] + bcs4[1] + bcs6[0],
            bcs1[2] + bcs2[1] + bcs3[0] + bcs5[1] + bcs6[1]),
           (bcs1[3] + bcs2[2] + bcs3[1] + bcs4[2] + bcs5[2],
            bcs2[3] + bcs3[2] + bcs4[3] + bcs6[2],
            bcs3[3] + bcs5[3] + bcs6[3]))
    # convert into (a,b,c,d) coordinates
    abc = np.sum(bcs, axis=0)
    d = np.sum(bcs[1], axis=0)
    return np.asarray((abc[0], abc[1], d))

def point_in_prism(tri1, tri2, pt):
    '''
    point_in_prism(tri1, tri2, pt) yields True if the given point is inside the prism that stretches
      between triangle 1 and triangle 2. Will automatically thread over extended dimensions. If
      multiple triangles are given, then the vertices must be an earlier dimension than the
      coordinates; e.g., a 3 x 3 x n array will be assumed to organized such that element [0,1,k] is
      the y coordinate of the first vertex of the k'th triangle.
    '''
    bcs = prism_barycentric_coordinates(tri1, tri2, pt)
    return np.logical_not(np.isclose(np.sum(bcs, axis=0), 0))
