# -*- coding: utf-8 -*-
####################################################################################################
# neuropythy/util/vset.py
# This file contains code fo the VertexSet class and related classes.

import os, sys, six, pimms
import numpy      as np
import pyrsistent as pyr
from functools import reduce

from .. import math as nym
from ..util import (is_str, is_tuple, is_list, is_map)
from .vset import (VertexSet, is_vset)

# Linear Splines ###################################################################################
@pimms.immutable
class LinearSpline(VertexSet):
    """An N-dimensional linear spline.

    Linear splines are simply paths through N-dimensional space that consist of
    discrete straight line-segments that are joined at their ends. The
    `LinearSpline` class is a `pimms` immutable class that represents these
    splines.

    The `LinearrSpline` class is a subtype of `VertexSet`, and, as such, it may
    map properties onto the vertices along the spline.

    `LinearSpline` is a `pimms.immutable` class, and it inherits from
    `VertexSet`.

    Parameters
    ----------
    *args
        Either a single `dims`x`npoints` matrix or a sequence of `x`, `y`, `z`,
        etc. values, one for each dimension of the spline. The points
        represented in this matrix are taken to be an ordered sequence of steps
        along the spline.
    periodic : boolean or None, optional
        Whether to make the resulting subspline periodic or not. If `True`,
        then the resulting subspline is always periodic, even if the last
        point must be connected to the first in order to make this
        happen. If `False`, then the resulting subspline is never periodic,
        even if the first and last points are equal. If `None`, then the
        result is a periodic spline if `f0` and f1` refer to equal points
        along the spline and false otherwise. The default is `False`.
    labels : vector of positive integers, optional
        The labels to use for the vertices. Typically this should be left to the
        default value of `None`, which results in a set of labels equivalent to
        `arange(n)`.
    properties : mapping of strings to vectors, optional
        A properties mapping, in which keys are strings (property names) and
        values are vectors of properties whose lengths are the same as the
        number of points in the spline. The default, `None`, does not include
        any properties beyond those automatically included by `VertexSet`.
    meta_data : mapping, optional
        An optional mapping of meta-data to attach to the object.

    Attributes
    ----------
    coordinates : read-only numpy matrix
        The matrix of coordinates. The first axis of the matrix always
        corresponds to the dimensionality of the points in the spline while the
        second axis always corresponds to the number of points.
    periodic : boolean
        `True` if the spline is periodic and `False` if it is not. If the spline
        has a length of 0, then `periodic` is always `None`.
    distances : read-only numpy vector
        A diff-like vector of the distances between points. `distances[i]` is
        the Euclidean distance between `coordinates[:,i]` and
        `coordinates[:,i+1]`. If `periodic` is `True`, then distances has the
        same length as there are columns in `coordinates`, with the final
        element of `distances` giving the distance from `coordinates[:,0]` and
        `coordinates[:,-1]`. If `periodic` is `False`, then distances is one
        element shorter than each row of `coordinates`.
    cumdistances : read-only numpy vector
        A vector of the cumulative distance of each point in the spline along
        the spline overall.  For periodic splines, the `cumdistances` vector is
        has a length equal to one plus the number of points, with the final
        value giving the total length of the closed spline; for aperiodic
        splines, the length is equal to the number of points.
    labels
        See `VertexSet`.
    indices
        See `VertexSet`.
    properties
        See `VertexSet`.
    meta_data
        See `VertexSet`.

    """
    def __init__(self, *args, periodic=False, labels=None, properties=None, meta_data=None):
        # What are our dimensions?
        d = len(args)
        assert d > 0, "at least one coordinate vector is required"
        if d == 1:
            # Could be a coordinate matrix or a 1D spline / vector.
            if nym.arraylike(args[0], ndim=2):
                args = args[0]
        # Make it an array / tensor.
        if labels is None:
            mtx = nym.promote(args)
            if not nym.is_numeric(mtx, ndim=2): raise ValueError("arguments must specify a matrix")
            (d,n) = args.shape
            if d == 0: raise ValueError("argument matrix must have at least one dimension")
            labels = nym.as_readonly(np.arange(n, dtype=np.int))
        else:
            n = len(labels)
        VertexSet.__init__(self, labels, properties=properties, meta_data=meta_data)
        self.coordinates = mtx
        # Detect whether this is periodic and fix the periodic value if need-be:
        if n < 2:
            periodic = None
        else:
            u0 = mtx[:,0]
            k = 0
            for k in range(n-1, -1, -1):
                if k == 0 or not nym.equal(u0, mtx[:,k]): break
            # if k == 0, then we went through all the points and they are all the same
            if k == 0: periodic = None
            elif periodic is None: periodic = (k < n-1)
        self.periodic = periodic
    @pimms.param
    def coordinates(x):
        'lspline.coordinates is the seed coordinate matrix for the given curve.'
        mtx = promote(x)
        if not nym.is_numeric(mtx, ndim=2): raise ValueError("arguments must specify a matrix")
        (d,n) = mtx.shape
        if d == 0: raise ValueError("argument matrix must have at least one dimension")
        return nym.as_readonly(mtx)
    @pimms.require
    def req_coordinates_match_vertex_count(coordinates, vertex_count):
        'Requires that the size of the coordinantes matches the vetex count.'
        return (coordinates.shape[1] == vertex_count)
    @pimms.param
    def periodic(p):
        'lspline.periodic is True if the given curve is a periodic curve and False otherwise.'
        return None if p is None else bool(p)
    @pimms.value
    def distances(coordinates, periodic):
        'lspline.distances is the specified curve-distances between points along the given spline.'
        if periodic:
            dists = nym.sqrt(nym.sum((coordinates - nym.roll(coordinates, -1, axis=1))**2, axis=0))
        else:
            dists = nym.sqrt(nym.sum((coordinatesp[:,:-1] - coordinatesp[:,1:])**2, axis=0))
        return nym.as_readonly(dists)
    @pimms.value
    def cumdistances(distances):
        '''lspline.cumdistances is the cumulative sum of the distances between points in
        the given curve.
        '''
        return nym.as_readonly(nym.cat([[0.0], nym.cumsum(distances)]))
    @pimms.value
    def length(cumdistances):
        '''lspline.length is the total length of the linear spline.'''
        return cumdistances[-1]
    @pimms.value
    def endvector_data(coordinates, periodic):
        '''lspline.endvector_data is None for periodic splines or is a pair of tuples
        `((k0,d0,u0), (k1,d1,u1))` where the `k0` and `k1` are the vertices just
        inside the front and end (respectively) of the spline such that, for
        `k0`, the vector from vertex `k0` to vertex 0 points away from the front
        of the spline, and the vector from vertex `k1` to the last vertex in the
        spline points away fron the end of the splint. The `d0` and `d1` are the
        lengths of the vectors from `k0` or `k1` to their respective ends (`d0`
        and `d1` are always greater than 0), and `u0` and `u1` are the unit
        vectors that point in this direction.
        '''
        if periodic: return None
        n = coordinates.shape[1]
        if n < 2: raise ValueError("extrapolation requires 2 or more unique spline points")
        # Find the ending unit vector first.
        u1 = coordinates[:,:,-1]
        u0 = None
        for k0 in range(n-2,-1,-1):
            if nym.equal(u1, coordinates[:,k0]): continue
            u0 = coordinates[:,k0]
            break
        if u0 is None: raise ValueError("extrapolation requires 2 or more unique spline points")
        u_end = nym.sub(u1 - u0)
        norm_end = nym.sqrt(nym.sum((u0 - u1)**2, axis=0))
        u_end /= norm_end
        # Now the starting unit vector.
        u1 = coordinates[:,0]
        u0 = None
        for k1 in range(1,n):
            if nym.equal(u, coordinates[:,k1]): continue
            u0 = coordinates[:,k1]
            break
        if u0 is None: raise ValueError("extrapolation requires 2 or more unique spline points")
        u_start = nym.sub(u1 - u0)
        norm_start = nym.sqrt(nym.sum((u0 - u1)**2, axis=0))
        u_start /= norm_start
        return ((k0, norm_start, nym.as_readonly(u0)), (k1, norm_end, nym.as_readonly(u1)))
    def __repr__(self):
        return 'LinearSpline(<%d points>, <%dD>%s)' % (
            self.coordinates.shape[1],
            self.coordinates.shape[0],
            '; periodic' if self.periodic else '')
    def _to_distances(f, unit):
        if unit is None: unit = 'fraction'
        unit = unit.lower()
        if   unit == 'spline':   return nym.mul(self.length, f)
        elif unit == 'distance': return f
        elif unit == 'index':    return self.cumdistances[f]
        else: raise ValueError(f"unrecognized unit: {unit}")
    def address(self, f, unit='fraction', extrapolate=False):
        """Find the address of a point along the spline.

        `lspline.address(f)` returns the address of the point a distance along
        the linear spline `lspline` equal to the fraction `f` times its total
        length.

        Parameters
        ----------
        f : real number or vector of real numbers
            The fraction of the distance along the total length of the linear
            spline at which to find the address. May be real number or a vector
            of points at which to produce addresses.
        unit : 'spline' or 'distance' or 'index', optional
            The unit of the value(s) of parameter `f`. For all three unit
            options, `f == 0` refers to the beginning of the spline. If
            `unit='spline'` (the default), then `f` is in units of the total
            spline length, e.g. `f == 1` indicates the end of the spline and `f
            == 0.25` refers to a point one quarter of the distance from the
            beginning to the end of the spline. If `unit='distance'` then `f`
            must be in units of Euclidean distance along the spline. If
            `unit='index'` then `f` must be integer-valued and must refer to the
            vertex indices of the spline.
        extrapolate : boolean, optional
            If `True`, then extrapolates off the end of the spline for distances
            that do not lie on the spline, otherwise, simply leaves these as
            `nan` values in the resulting address data. The default is
            `False`. This option is ignored for periodic splines. Note that when
            this value is `True`, the address data may not be technically valid
            as a set of relative coordinates due to the extrapolated points
            having weights greater than 1 or less than 0. For all extrapolated
            points, the first coordinate value in the address data will be less
            than zero.

        Returns
        -------
        mapping of address-data
            A dictionary containing address data for the points along the line
            segment.

        """
        f = promote(f)
        if not nym.is_numeric(f, '<=real', ndim=(0,1)):
            raise ValueError('argument must be a real number or a vector of reals')
        n = self.vertex_count
        d = self.dimensions
        l = self.length
        f = self._to_distances(f, unit)
        # Allocate the arrays we need...
        res = nym.empty_like(f, dtype='float', shape=(d, len(f)))
        segs = nym.empty_like(f, dtype='int', shape=(2, len(f)))
        crds = nym.empty_like(f, dtype='float', shape(2, len(f)))
        res = {'segments': segs, 'coordinates': crds}
        (ii1,ii2) = (segs[0,:], segs[1,:])
        wt1 = crds[0,:]
        wt2 = crds[1,:]
        # The calculation is different for periodic and aperiodic:
        if self.periodic:
            f = nym.mod(f, l)
            nym.searchsorted(cdist, f, out=ii1)
            nym.sub(f, cdist[ii1], out=wt2)
            nym.mod(ii1 + 1, n, out=ii2)
            nym.sub(cdist[ii2], f, out=wt1)
            wtot = wt1 + wt2
            # Anywhere that wtot is now zero indicates that two adjacent points were
            # on top of each other. In this case, we arbitrarily take the first.
            zz = (wtot == 0)
            if nym.any(zz):
                wt1[zz] = 1.0
                wtot[zz] = 1.0
            wt1 /= wtot
            wt2 /= wtot
        else:
            (gt, lt) = (nym.gt(f, l+1), nym.lt(f, -1))
            (any_gt, any_lt) = (nym.any(gt), nym.any(lt))
            if any_gt or any_lt:
                ok = np.where(~(gt|lt))[0]
                fok = f[ok]
                ii1[ok] = nym.searchsorted(cdist, fok)
            else:
                ok = slice(None)
                fok = f
                # Do the on-spline (ok) points first.
                nym.searchsorted(cdist, fok, out=ii1)
            nym.sub(fok, cdist[ii1], out=wt2[ok])
            ii2[ok] = ii1[ok] + 1
            endpt = ok[ii2[ok] == n]
            if len(endpt) > 0: ii2[endpt] = ii1[endpt]
            nym.sub(cdist[ii2], fok, out=wt2[ok])
            wtot = wt1[ok] + wt2[ok]
            zz = ok[wtot[ok] == 0]
            if len(zz) > 0:
                wt1[zz] = 1.0
                wtot[zz] = 1.0
            wt1[ok] /= wtot[ok]
            wt2[ok] /= wtot[ok]
            # Now do gt and lt elements:
            if extrapolate:
                ((k0,d0,u0), (k1,d1,u1)) = self.endvector_data
                if any_gt:
                    ii1[gt] = k1
                    ii2[gt] = n - 1
                    wt1[gt] = ((l - f[gt]) / d1)
                    wt2[gt] = 1.0 - wt1[gt]
                if any_lt:
                    ii1[lt] = k0
                    ii2[lt] = 0
                    wt1[lt] = (f[lt] / d0)
                    wt2[lt] = 1.0 - wt1[lt]
            else:
                if any_gt:
                    ii1[gt] = 0
                    ii2[gt] = 0
                    wt1[gt] = nym.nan
                    wt2[gt] = nym.nan
                if any_lt:
                    ii1[lt] = 0
                    ii2[lt] = 0
                    wt1[lt] = nym.nan
                    wt2[lt] = nym.nan
            # That's all.
            return res
    def __call__(self, f, unit='spline', extrapolate=True):
        """Returns the point(s) based on positioin along the spline.

        `lspline(f)` returns the point that is at the position along the
        `LinearSpline` object `lspline` corresponding to the fraction `f` of
        `lspline`'s total length (`lspline.length`).

        If `lspline` is periodic, then the input is modded by 1. Otherwise, if
        the input lies outside the range `[0,1]`, an error is raised.

        Parameters
        ----------
        `f` : number on `[0,1]` or vector-like of numbers on `[0,1]`
            The fraction or fractions of the spline's total distance at which to
            find the corresponding points.
        unit : 'spline' or 'distance' or 'index', optional
            The unit of the value(s) of parameter `f`. For all three unit
            options, `f == 0` refers to the beginning of the spline. If
            `unit='spline'` (the default), then `f` is in units of the total
            spline length, e.g. `f == 1` indicates the end of the spline and `f
            == 0.25` refers to a point one quarter of the distance from the
            beginning to the end of the spline. If `unit='distance'` then `f`
            must be in units of Euclidean distance along the spline. If
            `unit='index'` then `f` must be integer-valued and must refer to the
            vertex indices of the spline.
        extrapolate : boolean, optional
            If `True`, then extrapolates off the end of the spline for distances
            that do not lie on the spline, otherwise, simply leaves these as
            `nan` values in the resulting coordinates. The default is `True`,
            unlike for the `lspline.address(f)` function. This option is ignored
            for periodic splines.

        Returns
        -------
        vector or matrix
            Either a vector representing a point along the spline (if the input
            `f` was a single number) or a `d`x`n` matrix where `d` is the
            dimensionality oof the spline points and `n` is the length of `f`
            (if `f` is a vector of numbers).

        Raises
        ------
        ValueError
            If a value in the input `f` is outside the acceptable range of a
            non-periodic spline, or if `f` cannot be interpreted as a number or
            numeric vector.

        """
        addr = self.address(f, unit=unit, extrapolate=extrapolate)
        (ii1, ii2) = addr['segments']
        (wt1, wt2) = addr['coordinates']
        pts = nym.mul(self.coordinates[:,ii1], wt1) + nym.mul(self.coordinates[:,ii2], wt2)
        # That's all that is required of this function!
        return pts
    def linspace(self, n=100, tag_meta=True):
        """Returns a matrix of linearly-spaced points along the spline.

        `lspline.linspace(n)` yields `n` evenly-spaced points along the linear
        spline as a `dims`x`n` matrix. Note that this will distort the spline at
        any corner, depending on the number of output points chosen.

        Parameters
        ----------
        n : int, optional
            The number of points that the output matrix should have. The default
            value is 100.
        
        Returns
        -------
        numpy.ndarray
            A new `coordinates` matrix with the required number of points.

        Raises
        ------
        ValueError
            If the number of points requested is invalid.
        """
        if not nym.is_numeric(n, '<=int') or n < 0: raise ValueError("n must be a positive integer")
        ds = np.linspace(0.0, self.length, n)
        pts = self(ds)
        return pts
    def linsample(self, n=100, tag_meta=True):
        """Returns a resampled version of the spline with exactly `n` points.

        `lspline.linsample(n)` yields `n` evenly-spaced points along the linear
        spline. Note that this will distort the spline at any corner, depending
        on the number of output points chosen. This is essentially equivalent to
        making a new spline using the poinnts `lspline.linspace(n)`.

        Parameters
        ----------
        n : int, optional
            The number of points that the output spline should have. The default
            value is 100.
        tag_meta : boolean, optional
            If `True` (the default), then output spline's meta-data mapping will
            include the tag 'source_spline', which will point back to the
            original spline. If `False`, the no meta-data are included.
        
        Returns
        -------
        LinearSpline
            A new linear spline object with the required number of points.

        Raises
        ------
        ValueError
            If the number of points requested is invalid.
        """
        pts = self.linspace(n)
        md = {'source_spline': self} if tag_meta else None
        return LinearSpline(pts, periodic=bool(self.periodic), meta_data=md)
    @staticmethod
    def _rev_periodic(x):
        x = x[..., ::-1]
        x = nym.roll(x, 1, axis=-1)
        return nym.as_readonly(x)
    @staticmethod
    def _rev_aperiodic(x):
        return x[..., ::-1]
    def reverse(self, tag_meta=True):
        """Returns a reversed version of the spline.

        `lspline.reverse()` returns the reversed spline from `lspline`. This
        reverses not only the coordinate matrix, but also all of the properties
        and the labels.

        If a spline is aperiodic, then its coordinate matrix is simply the
        reversed matrix. If it is perioidic, however, the new spline will begin
        with the same point, but will otherwise traverse the points in the
        reverse order.

        Parameters
        ----------
        tag_meta : boolean, optional
            If `True` (the default), then output spline's meta-data mapping will
            include the tag 'source_spline', which will point back to the
            original spline. If `False`, the no meta-data are included.
        
        Returns
        -------
        LinearSpline
            A new linear spline object that has been reversed.
        """
        fixfn = LinearSpline._ref_periodic if self.peroidic else LinearSpline._rev_aperiodic
        pts = fixfn(self.coordinates)
        lbl = fixfn(self.labels)
        props = {k: curry(lambda k: fixfn(self.prop(k)), k)
                 for k in six.iterkeys(self.properties)}
        md = {'source_spline':self} if tag_meta else None
        return LinearSpline(pts, labels=lbls, properties=pimms.itable(props),
                            distance_fn=self.distance_fn, periodic=bool(self.periodic),
                            meta_data=md)
    @staticmethod
    def _interp_prop_helper(segs, crds, iis, method, props, k):
        prop = props[k]
        ((ii0_1, ii1_1), (ii0_2, ii1_2)) = segs
        ((wt0_1, wt1_1), (wt0_2, wt1_2)) = crds
        if method is None:
            if nym.is_numeric(prop, '>=real'): method = 'linear'
            else: method = 'heaviest'
        else:
            method = method.lower()
        if method == 'linear':
            p0 = prop[ii0_1]*wt0_1 + prop[ii0_2]*wt0_2
            p1 = prop[ii1_1]*wt1_1 + prop[ii1_2]*wt1_2
        else:
            p0 = prop[ii0_1] if wt0_1 > wr0_2 else prop[ii0_2]
            p1 = prop[ii1_1] if wt1_1 > wt1_2 else prop[ii1_2]
        return nym.as_readonly(nym.cat([[p0], prop[iis], [p1]]))
    def subspline(self, f0, f1, unit='spline', extrapolate=True, periodic=None,
                  interpolation=None):
        """Returns a sub-spline of an existing spline.

        `lspline.subcurve(f0, f1)` returns a `LinearSpline` object that is
        equivalent to the given spline `lspline` but that extends from
        `lspline(f0)` to `lspline(f1)` only.

        If `f0 < f1`, then the subspline is constructed in the same direction as
        `lspline` while if `f1 < f0`, the spline is constructed in the reverse
        direction. Note that this is true for both periodic splines and
        aperiodic splines, and the provided value of `f0` and `f1` are the
        values that determine the subspline direction, not their periodic
        (modded) values.

        When called on a periodic spline, the result may traverse the spline
        multiple times, depending on the values given. For example, if `s` is a
        periodic linear spline, then `s.subspline(0, 2.5)` will traverse all the
        points in `s` twice then will end half-way along the original spline.

        Parameters
        ----------
        f0 : real number on [0,1]
            The fraction of the total length of the spline at which the
            beginning of the sub-spline should occur.
        f1 : real nunmber on [0,1]
            The fraction of the total length of the spline at which the end of
            the sub-spline should occur.
        unit : 'spline' or 'distance' or 'index', optional
            The unit of the value(s) of parameter `f`. For all three unit
            options, `f == 0` refers to the beginning of the spline. If
            `unit='spline'` (the default), then `f` is in units of the total
            spline length, e.g. `f == 1` indicates the end of the spline and `f
            == 0.25` refers to a point one quarter of the distance from the
            beginning to the end of the spline. If `unit='distance'` then `f`
            must be in units of Euclidean distance along the spline. If
            `unit='index'` then `f` must be integer-valued and must refer to the
            vertex indices of the spline.
        extrapolate : boolean, optional
            If `True`, then extrapolates off the end of the spline for distances
            that do not lie on the spline, otherwise, simply leaves these as
            `nan` values in the resulting coordinates. The default is `True`,
            unlike for the `lspline.address(f)` function. This option is ignored
            for periodic splines.
        periodic : boolean or None, optional
            Whether to make the resulting subspline periodic or not. If `True`,
            then the resulting subspline is always periodic, even if the last
            point must be connected to the first in order to make this
            happen. If `False`, then the resulting subspline is never periodic,
            even if the first and last points are equal. If `None`, then the
            result is a periodic spline if `f0` and f1` refer to equal points
            along the spline and false otherwise.
        interpolation : 'heaviest' or 'linear' or None, optional
            The property values of the end-points of the subspline usually have
            to be interpolated from the nearby spline points. In order to do
            this, the interpolation method specified by `interpolation` is
            used. The default, `None` uses `'linear'` for floating-point
            properties and `'heaviest'` for all other properties. The
            `'heaviest'` method is essentially nearest-neighbor interpolation,
            but it considers only distance along the spline, not distance in
            space generally.

        Returns
        -------
        LinearSpline
            A `LinearSpline` object representing th sub-spline specified.
        """
        if f0 > f1: (f0,f1,rev) = (f1,f0,True)
        else: rev = False
        mtx = self.coordinates
        l = self.length
        n = mtx.shape[1]
        addr = self.address([f0,f1], unit=unit, extrapolate=extrapolate)
        ((ii0_1, ii1_1), (ii0_2, ii1_2)) = segs = addr['segments']
        ((wt0_1, wt1_1), (wt0_2, wt1_2)) = crds = addr['coordinates']
        p0 = nym.add(nym.mul(mtx[:,ii0_1], wt0_1), nym.mul(mtx[:,ii0_2], wt0_2))
        p1 = nym.add(nym.mul(mtx[:,ii1_1], wt1_1), nym.mul(mtx[:,ii1_2], wt1_2))
        # The rest depends on periodicity!
        if self.periodic:
            ntimes = int(nym.floor((f1 - f0) / l))
            p0_nextii = nym.mod(ii0_2+1, n) if ii0_1 == ii0_2 else ii0_2
            p1_previi = num.mod(ii1_1-1, n) if ii1_1 == ii1_2 else ii1_1
            if ntimes > 1:
                iis = nym.cat([nym.arange(p0_next, n)] +
                              [nym.arange(n)]*ntimes +
                              [nym.arange(0, p1_previi)])
            else:
                iis = slice(p0_nextii, p1_previi)
        else:
            p0_nextii = ii0_2+1 if ii0_1 == ii0_2 else ii0_2
            p1_previi = ii1_1-1 if ii1_1 == ii1_2 else ii1_1
            iis = slice(p0_nextii, p1_previi)
        pts = nym.cat([p0[:,None], mtx[:, iis], p1[:,None]])
        props = self._properties
        if props is not None and len(props) > 0:
            mtd = interpolation
            if mtd is not None:
                mtd = to_interpolation_method(mtd)
                if method != 'linear' and method != 'heaviest':
                    raise ValueError(f"unsupported spline property interpolation method: {method}")
            props = {k: curry(LinearSpline._interp_prop_helper, segs, crds, iis, mtd, props, k)
                     for k in props}
        return s if rev else s.reverse()
def to_lspline(*args, periodic=Ellipsis, properties=Ellipsis, meta_data=Ellipsis):
    """Converts the given argument to a `LinearSpline`, or raises an error.

    `to_lspline(spline)` returns the object `spline` if `spline` is
    already a linear spline object.

    `to_lspline(coord_matrix)` yields a linear spline function through the
    points in the given coordinate matrix. The matrix must have a shape of
    `ndims`x`npoints` where `ndims` is the dimensionality of the points, and
    `npoints` is the number of points in the spline.

    `to_lspline((coord_matrix, kw))` additionally uses the keyword
    dictionary `kw` as the options to `LinearSpline()`; all optional parameters
    below may appear in this dictionary.

    `to_lspline((x, y, z..., kw))` uses the coordinate matrix `[x,y,z...]`, for
    vectors `x`, `y`, `z`, etc. and the keyword dictionary `kw` for th options.

    The `to_lspline()` function returns a `LinearSpline` object `s`, which
    behaves like a function `s(f)`. `s(f)` yields the point a distance along the
    spline `s` equal to `f` times the total length of the spline
    (`s.length`). `s(d, unit='distance')` and `s(ii, unit='index')` can be used
    to obtain points along the spline at specific distances or vertex indices as
    well.

    The optional parameters below may be passed in all of the above cases and
    provide or override any existing representation of the option in the `*args`
    parameters. If a linear spline object is passed to `to_lspline()` and
    optional parameters that disagree with those of the spline, then a duplicate
    spline with the new parameters is returned.

    Parameters
    ----------
    *args
        Must be one of the following: a single `dims`x`npoints` matrix; a
        sequence of `x`, `y`, `z`, etc. vectors, one for each dimension of the
        spline; a tuple containing one of these, followed by a dictionary of any
        of the keywoord options below. The points represented in the matrix or
        vectors are taken to be an ordered sequence of steps along the
        spline. Any optional parameters explicitly passed to this function that
        are not `None` override this.
    periodic : boolean or None or Ellipsis, optional
        Whether to make the resulting subspline periodic or not. If `True`, then
        the resulting subspline is always periodic, even if the last point must
        be connected to the first in order to make this happen. If `False`, then
        the resulting subspline is never periodic, even if the first and last
        points are equal. If `None`, then the result is a periodic spline if
        `f0` and f1` refer to equal points along the spline and false
        otherwise. The default is `Ellipsis`, which instructs the functioin to
        defer to any `periodic` parameter included in `*args` and to use `False`
        of there is none.
    labels : vector of positive integers or None or Ellipsis, optional
        The labels to use for the vertices. Typically this should be left to the
        default value of `Ellipsis`, which uses whatever was provided in
        `*args`, if anything, and `None` otherwise, which results in a set of
        labels equivalent to `arange(n)`.
    properties : mapping of strings to vectors or None or Ellipsis, optional
        A properties mapping, in which keys are strings (property names) and
        values are vectors of properties whose lengths are the same as the
        number of points in the spline. The value `None` does not include any
        properties beyond those automatically included by `VertexSet`. The value
        of `Ellipsis` (the default) uses whatever properties were provided in
        `*args`, if any, and `None` otherwise.
    meta_data : mapping or Ellipsis, optional
        An optional mapping of meta-data to attach to the object. A value of
        `Ellipsis` uses whatever meta-data are found in `*args` and `None`
        otherwise.

    Returns
    -------
    LinearSpline
        Either the argument itself, if the function is passed an object that is
        already a linear spline, or the spline represeennted by the arguments if
        not.

    Raises
    ------
    ValueError
        If the arguments cannot be interpreted as
    """
    # What do we have?
    nargs = len(args)
    (ls,mtx,ops) = (None,None,None)
    if nargs == 0: raise ValueError("to_lspline requires at least one argument")
    elif nargs == 1:
        x = nargs[0]
        if isinstance(x, LinearSpline):
            ls = x
            if periodic is Ellipsis and properties is Ellipsis and meta_data is Ellipsis:
                return ls
        elif is_tuple(x):
            if len(x) == 2 and is_map(x[1]):   (mtx,ops) = x
            elif len(x) > 2 and is_map(x[-1]): (mtx,ops) = (x[:-1],x[-1])
            else: mtx = x
        else: mtx = x
    else: mtx = args
    if ops is None: ops = {}
    if periodic is not Ellipsis: ops['periodic'] = periodic
    if labels is not Ellipsis: ops['labels'] = labels
    if properties is not Ellipsis: ops['properties'] = properties
    if meta_data is not Ellipsis: ops['meta_data'] = meta_data
    if lw is None:
        return LinearSpline(mtx, *ops)
    else:
        if len(ops) == 0: return lw
        p = ops.pop('properties', Ellipsis)
        if p is not Ellipsis: lw = lw.with_prop(p)
        return lw.copy(**ops)
def is_lspline(obj):
    """Determines if the given object is a `LinearSpline` object.
    
    `is_lspline(obj)` returns `True` if `obj` is a `LinearSplin` object
    and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a linear spline is to be determined.

    Returns
    -------
    boolean
        `True` if `obj` is a `LinearSpline` and `False` otherwise.
    """
    return isinstance(obj, LinearSpline)
#TODO: Cleaning below here!
def lspline_intersections(s1, s2):
    """Finds the intersections between two linear splines.

    `lspline_intersects(s1, s2)` returns a list of the intersections between the
    two splines. This list may be empty if there are no intersections.
    Intersections may be points, which are returned as tuples `(f1, f2)` of the
    fractional distances such that `s1(f1) == s2(f2)`, or tuples of tuples
    `((f1_start, f1_end), (f2_start, f2_end))` such that `s1(f1_start*(1-p) +
    f1_end*p) == s2(f2_start*(1-p) + f2_end*p)` for `0 <= p <= 1`.

    Parameters
    ----------
    s1 : linear-spline-like
        Either a `LinearSpline` object or an object that can be converted into a
        `LinearSpline` object using the `to_linear_spline(s1)` function.
    s2 : linear-spline-like
        Either a `LinearSpline` object or an object that can be converted into a
        `LinearSpline` object using the `to_linear_spline(ss)` function.

    Returns
    -------
    numpy matrix
        A list if intersections, as described above.
    """
    (s1,s2) = (to_lspline(s1), to_lspline(s2))
    # Just compare every segment against every other segment.
    (x1,x2) = (s1.coordinates, s2.coordinates)
    d = x1.shape[0]
    if d != x2.shape[0]:
        raise ValueError("intersecting splines must have the same dimensionality")
    if d > 3:
        #TODO: implement this
        raise ValueError("intersections for dimensions > 3 are not yet supported")
    # If we're periodic, make sure the first and last points are equal
    if s1.periodic:
        if not nym.equal(x1[:,0], x1[:,-1]):
            x1 = nym.cat([x1, x1[:,[0]]], axis=1)
    if s2.periodic:
        if not nym.equal(x2[:,0], x2[:,-1]):
            x1 = nym.cat([x2, x2[:,[0]]], axis=1)
    # We make a grid version of this for the intersection functions.
    # We're going to repeat segs many times and test for intersections.
    (n1, n2) = (x1.shape[1], x2.shape[1])
    segs1 = [nym.cat([x1[:,:-1]]*n2, axis=1)
             nym.cat([x1[:,1:]]*n2, axis=1)]
    segs2 = [nym.reshape(nym.tr([x2[:,:-1]]*n1, (1, 2, 0)), (d,-1)),
             nym.reshape(nym.tr([x2[:,1:]]*n1, (1,2,0)), (d,-1))]
    
    #TODO: Finish this function and below here.
    
    
    from scipy.optimize import minimize
    from neuropythy.geometry import segment_intersection_2D
    if c1.coordinates.shape[1] > c2.coordinates.shape[1]:
        (t1,t2) = lspline_intersection(c2, c1, grid=grid)
        return (t2,t1)
    # before doing a search, see if there are literal exact intersections of the segments
    x1s  = c1.coordinates.T
    x2s  = c2.coordinates
    for (ts,te,xs,xe) in zip(c1.t[:-1], c1.t[1:], x1s[:-1], x1s[1:]):
        pts = segment_intersection_2D((xs,xe), (x2s[:,:-1], x2s[:,1:]))
        ii = np.where(np.isfinite(pts[0]))[0]
        if len(ii) > 0:
            ii = ii[0]
            def f(t): return np.sum((c1(t[0]) - c2(t[1]))**2)
            t01 = 0.5*(ts + te)
            t02 = 0.5*(c2.t[ii] + c2.t[ii+1])
            (t1,t2) = minimize(f, (t01, t02)).x
            return (t1,t2)
    if pimms.is_vector(grid): (ts1,ts2) = [c.t[0] + (c.t[-1] - c.t[0])*grid for c in (c1,c2)]
    else:                     (ts1,ts2) = [np.linspace(c.t[0], c.t[-1], grid) for c in (c1,c2)]
    (pts1,pts2) = [c(ts) for (c,ts) in zip([c1,c2],[ts1,ts2])]
    ds = np.sqrt([np.sum((pts2.T - pp)**2, axis=1) for pp in pts1.T])
    (ii,jj) = np.unravel_index(np.argmin(ds), ds.shape)
    (t01,t02) = (ts1[ii], ts2[jj])
    ttt = []
    def f(t): return np.sum((c1(t[0]) - c2(t[1]))**2)
    (t1,t2) = minimize(f, (t01, t02)).x
    return (t1,t2)
def close_curves(*crvs, **kw):
    '''
    close_curves(crv1, crv2...) yields a single curve that merges all of the given list of curves
      together. The curves must be given in order, such that the i'th curve should be connected to
      to the (i+1)'th curve circularly to form a perimeter.

    The following optional parameters may be given:
      * grid may specify the number of grid-points to use in the initial search for a start-point
        (default: 16).
      * order may specify the order of the resulting curve; by default (None) uses the lowest order
        of all curves.
      * smoothing (None) the amount to smooth the points.
      * even_out (True) whether to even out the distances along the curve.
      * meta_data (None) an optional map of meta-data to give the spline representation.
    '''
    for k in six.iterkeys(kw):
        if k not in close_curves.default_options: raise ValueError('Unrecognized option: %s' % k)
    kw = {k:(kw[k] if k in kw else v) for (k,v) in six.iteritems(close_curves.default_options)}
    (grid, order) = (kw['grid'], kw['order'])
    crvs = [(crv if is_curve_spline(crv) else to_curve_spline(crv)).even_out() for crv in crvs]
    # find all intersections:
    isects = [curve_intersection(u,v, grid=grid)
              for (u,v) in zip(crvs, np.roll(crvs,-1))]
    # subsample curves
    crds = np.hstack([crv.subcurve(s1[1], s0[0]).coordinates[:,:-1]
                      for (crv,s0,s1) in zip(crvs, isects, np.roll(isects,1,0))])
    kw['order'] = np.min([crv.order for crv in crvs]) if order is None else order
    kw = {k:v for (k,v) in six.iteritems(kw)
          if v is not None and k in ('order','smoothing','even_out','meta_data')}
    return curve_spline(crds, periodic=True, **kw)
close_curves.default_options = dict(grid=16, order=None, even_out=True,
                                    smoothing=None, meta_data=None)
