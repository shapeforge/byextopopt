import numpy
from scipy.sparse import coo_matrix
import cvxopt
import cvxopt.cholmod
from common import InterpolationType


class TopoptProblem(object):
    """
    TODO: Implement multiple load case.
    """

    def build_indices(self, nelx, nely, params):
        # FE: Build the index vectors for the for coo matrix format.
        self.KE = lk(params)
        self.edofMat = numpy.zeros((nelx * nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + elx * nely
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                self.edofMat[el, :] = numpy.array([2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 +
                                                   3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])
        # Construct the index pointers for the coo format
        self.iK = numpy.kron(self.edofMat, numpy.ones((8, 1))).flatten()
        self.jK = numpy.kron(self.edofMat, numpy.ones((1, 8))).flatten()

    def __init__(self, nelx, nely, params, bc):
        # Problem size
        self.nelx = nelx
        self.nely = nely

        # Max and min stiffness
        self.Emin = params.youngModulusMin
        self.Emax = params.youngModulusMax

        # SIMP penalty
        self.penal = params.penalty

        # Dofs
        self.ndof = 2 * (nelx + 1) * (nely + 1)

        # FE: Build the index vectors for the for coo matrix format.
        self.build_indices(nelx, nely, params)

        # self-weight parameters
        self.materialDensity = params.materialDensity
        self.gravity = params.gravity
        self.lengthSquare = params.lengthSquare
        self.thickness = params.thickness
        self.interpolationType = params.interpolationType
        self.treshPedersen = params.treshPedersen

        # BC's and support
        self.bc = bc
        self.problemOptions = params.ProblemOptions
        fixed_coords = self.bc.get_fixed_nodes(nelx, nely, self.problemOptions)
        self.fixed = numpy.unique([2 * ((nely + 1) * x + y) + c for (x, y, c) in fixed_coords])
        self.free = numpy.setdiff1d(numpy.arange(self.ndof), self.fixed)

        # Solution and RHS vectors
        self.u = numpy.zeros((self.ndof, 1))
        self.f = numpy.zeros((self.ndof, 1))
        bc.set_forces(nelx, nely, self.f, params.problemOptions)

        # Per element compliance
        self.ce = numpy.ones(nely * nelx)

    def young_modulus(self, x):
        # Compute penalized material densities
        if self.interpolationType == InterpolationType.SIMP:
            return self.Emin + (x) ** self.penal * (self.Emax - self.Emin)
        elif self.interpolationType == InterpolationType.Pedersen:
            return numpy.where(
                x >= self.treshPedersen, self.Emin + x ** self.penal * (self.Emax - self.Emin),
                self.Emin + x * self.treshPedersen ** (self.penal - 1.0) * (self.Emax - self.Emin))
        assert False, "Unrecognized configuration"
        return x

    def young_modulus_grad(self, x):
        # Compute derivative of the penalized material densities
        if self.interpolationType == InterpolationType.SIMP:
            return self.penal * x ** (self.penal - 1.0) * (self.Emax - self.Emin)
        elif self.interpolationType == InterpolationType.Pedersen:
            return numpy.where(
                x >= self.treshPedersen, self.penal * x ** (self.penal - 1.0) * (self.Emax - self.Emin),
                self.treshPedersen ** (self.penal - 1.0) * (self.Emax - self.Emin))
        assert False, "Unrecognized configuration"
        return x

    def compute_displacements(self, xPhys):
        if self.materialDensity > 0.0:
            # Self-weight case
            self.f.fill(0)
            self.bc.set_forces(self.nelx, self.nely, self.f, self.problemOptions)
            for e, __ in enumerate(xPhys):
                for k in range(1, len(self.edofMat[e]), 2):
                    self.f[self.edofMat[e][k]] -= (xPhys[e] * self.gravity * self.materialDensity * self.lengthSquare * self.lengthSquare * self.thickness) / 4.0
        # Setup and solve FE problem
        sK = ((self.KE.flatten()[numpy.newaxis]).T * (self.young_modulus(xPhys))).flatten(order='F')
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        # Remove constrained dofs from matrix and convert to coo
        K = deleterowcol(K, self.fixed, self.fixed).tocoo()
        # Solve system
        K = cvxopt.spmatrix(K.data, K.row.astype(
            numpy.int), K.col.astype(numpy.int))
        B = cvxopt.matrix(self.f[self.free, 0])
        cvxopt.cholmod.linsolve(K, B)
        self.u[self.free, 0] = numpy.array(B)[:, 0]

    def compute_compliance(self, xPhys, dc):
        # Compute compliance and its gradient
        self.ce[:] = (numpy.dot(self.u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE) *
                      self.u[self.edofMat].reshape(self.nelx * self.nely, 8)).sum(1)
        obj = ((self.young_modulus(xPhys)) * self.ce).sum()
        if dc is not None:
            if self.materialDensity > 0.0:
                # Self-weight case
                for e, __ in enumerate(xPhys):
                    u_e = self.u[self.edofMat[e]].T
                    dev_f_e = -numpy.ones(8) * (self.gravity * self.materialDensity * self.lengthSquare * self.lengthSquare * self.thickness) / 4.0
                    for i in range(8):
                        if i % 2 == 0:
                            dev_f_e[i] = 0
                    dc[e] = 2.0 * numpy.dot(u_e, dev_f_e) - self.young_modulus_grad(xPhys[e]) * numpy.dot(numpy.dot(u_e, self.KE), u_e.T)
            else:
                dc[:] = -self.young_modulus_grad(xPhys) * self.ce
        return obj


def lk(params):
    # Element stiffness matrix
    E = params.youngModulusMax
    nu = params.poissonCoeff
    k = numpy.array([1.0 / 2.0 - nu / 6.0, 1.0 / 8.0 + nu / 8.0, -1.0 / 4.0 - nu / 12.0, -1.0 / 8.0 + 3.0 * nu /
                     8.0, -1.0 / 4.0 + nu / 12.0, -1.0 / 8.0 - nu / 8.0, nu / 6.0, 1.0 / 8.0 - 3.0 * nu / 8])
    KE = E / (1 - nu**2) * numpy.array(
        [[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
         [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
         [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
         [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
         [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
         [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
         [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
         [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE


def deleterowcol(A, delrow, delcol):
    # Assumes that matrix is in symmetric csc form!
    m = A.shape[0]
    keep = numpy.delete(numpy.arange(0, m), delrow)
    A = A[keep, :]
    keep = numpy.delete(numpy.arange(0, m), delcol)
    A = A[:, keep]
    return A
