####################################################################################################
# registration.py
# Tools for registering the cortical surface to a particular potential function
# By Noah C. Benson

import numpy as np
import scipy as sp
import os, math
from cortex import CorticalMesh
from scipy.optimize import minimize

def register(mesh, terms, **args):
    '''register(mesh, [term1, term2...], options...) yields the result of registering the given 
       cortical mesh to the given potential (defined by the sum of the given potential terms). The
       registration is found by minimizing the potential function with respect to the coordinates
       of the vertices of mesh. Minimization is performed by scipy's minimize() function. All
       options that can be passed to minimize() as part of the 'options' option may be passed
       directly to register().

       Potential terms describe what the mesh is being minimized to; all potential terms should
       be objects that overload the following functions:
        * initialize(mesh, opts...): this gets called by the register() function and initializes
          the term with both the mesh being minimized and all options given to the register function
          itself. PotentialTerm objects may of course include their own constructors.
        * __call__(X): this function gets called with X, a numpy matrix that is the alternate
          coordinate matrix for the vertices in the given mesh during the minimization. This
          function should return the potential value itself at the given matrix. Note that this
          matrix is always given as a 2 or 3 x n orientation.
        * grad(X): this function should yield the gradient of the potential function at the point
          given by the coordinate matrix X.
        * hessian(X): this function should return the Hessian of the potential function at the given
          coordinate matrix X. Note that the the order of the columns and rows in the Hessian should
          be the same as in X.flatten(). Additionally, the Hessian should almost always be
          represented as a sparse matrix. This is due to the size of the Hessian, which will
          generally be on the order of 15-500 thousand rows and columns.'''
    # start by validating the terms and initializing
    if len(terms) < 1:
        raise ValueError('There must be at least one term given to register()')
    useHessian = True
    for term in terms:
        if hasattr(term, 'initialize') and hasattr(term.initialize, '__call__'):
            term.initialize(mesh, **args)
        if not hasattr(term, '__call__'):
            raise ValueError('all terms *must* have __call__ methods that yield the potential')
        if not hasattr(term, 'grad') or not hasattr(term.grad, '__call__'):
            raise ValueError('all terms *must* have callable grad methods')
        if not hasattr(term, 'hessian') or not hasattr(term.hessian, '__call__'):
            useHessian = False
    # The data we minimize over:
    dims = mesh.coordinates.shape[0]
    X0 = mesh.coordinates.flatten()
    # okay, now call minimize
    potential = lambda X: np.sum([term(np.split(X,dims)) for term in terms])
    jacobian  = lambda X: np.sum([term.grad(np.split(X,dims)) for term in terms], 0)
    if useHessian:
        hessian = lambda X: np.sum([term.hessian(np.split(X,dims)) for term in terms], 0)
        hessianp = lambda X, p: hessian(X).dot(p)
        X = minimize(potential, X0, method='Newton-CG', jac=jacobian, hessp=hessianp, options=args)
    else:
        X = minimize(potential, X0, method='BFGS', jac=jacobian, options=args)
    return X

class HarmonicEdgePotential:
    '''HarmonicEdgePotential is a class that handles the harmonic potential function represented by
       two vertices in a cortical mesh bound together by an edge. The form of the potential function
       is 0.5 k (x - x0)^2 / n where x is the distance between two vertices, x0 is the distance
       between the two vertices in the ('correct') reference mesh orientation, k is a user-supplied
       constant, and n is the total number of edges in the mesh. A harmonic edge potential object
       may be safely passed to the register() function as a potential term.
       The HarmonicEdgePotential constructor requires one argument: k.'''
    def __init__(self, k = 1):
        self.constant = k
    def initialize(self, mesh, **opts):
        # local temporaries we use here...
        X0 = mesh.coordinates
        E = mesh.edges
        n = X0.shape[1]
        m = E.shape[1]
        d = len(X0)
        # remember the mesh...
        self.mesh = mesh
        # and the overall constant
        self.coefficient = self.constant / m
        # and the initial edge lengths
        self.X0 = X0
        self.D0 = np.sqrt(np.sum((X0[:, E[0]] - X0[:, E[1]]) ** 2, 0))
        # we also need a sparse matrix for summing over edges
        self.sumMatrix = sp.sparse.dok_matrix((n, m), dtype=np.float32)
        for i in range(m):
            self.sumMatrix[E[0, i], i] = 1
            self.sumMatrix[E[1, i], i] = -1
        # finally, the hessian matrix is actually constant, so we can pre-calculate it here
        self.constant_hessian = sp.sparse.dok_matrix((d*n, d*n), dtype=np.float32)
        for i in range(m):
            for j in range(d):
                self.constant_hessian[j*n + E[0,i], j*n + E[1,i]] = self.coefficient
    def __call__(self, X):
        E = self.mesh.edges
        X0 = self.mesh.coordinates
        D0 = self.D0
        # calculate the total potential
        D = np.sqrt(np.sum((X0[:, E[0]] - X0[:, E[1]]) ** 2, 0))
        return np.sum(0.5 * self.coefficient * (D - D0) ** 2)
    def grad(self, X):
        E = self.mesh.edges
        X0 = self.mesh.coordinates
        D0 = self.D0
        # vector from edge0 to edge1 for each edge:
        dX = X0[:, E[1]] - X0[:, E[0]]
        # distance of each edge:
        D = np.sqrt(np.sum(dX ** 2, 0))
        # the gradient length, one for each edge, always pointing from e0 -> e1
        Glen = self.coefficient * (D0 - D)
        G = dX * np.array([Glen / D for i in X0])
        return np.array([self.sumMatrix.dot(dX[i]) for i in range(len(dX))]).flatten()
    def hessian(self, X):
        # nice thing about the second derivative of a harmonic...
        return self.constant_hessian

class HarmonicAnglePotential:
    '''HarmonicAnglePotential is a class that handles the harmonic potential function represented by
       the deviation of the angles of the cortical mesh faces. The form of the potential fucntion is
       0.5 k (t - t0)^2 / m where t is the angle of a particular face, t0 is the initial angle of
       the same face in the starting (reference) mesh, k is a user-supplied constant, and m is the
       number of angles in the mesh. A harmonic angle potential object may be safely passed to the
       register() function as a potential term.
       The HarmonicAnglePotential constructor requires one argument: k.'''
    def __init__(self, k = 1):
        self.constant = k
    def initialize(self, mesh, **opts):
        # Local temporaries
        X0 = mesh.coordinates
        F = mesh.faces
        d = X0.shape[0]
        n = X0.shape[1]
        m = F.shape[1] * 3
        # some relevant values we want to keep track of...
        self.mesh = mesh
        self.coefficient = self.constant / m
        self.X0 = X0
        self.T0 = 0 #here
        


