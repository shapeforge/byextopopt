class BaseProblem(object):

    @staticmethod
    def get_fixed_nodes(nelx, nely, params):
        """
        Return a list of triplets (nx, ny, c) indicating the fixed degrees of freedom.

        Parameters
        ----------
        nelx : int
            Width of the domain at the current level.
        nely : int
            Height of the domain at the current level.
        params : Struct
            Other parameters specific to this problem description.
        """
        return None

    @staticmethod
    def set_forces(nelx, nely, f, params):
        """
        Set the force vector for the problem.

        The force vector has dimension (ndof, 1), where ndof if the
        dimension * number of nodes, i.e. 2 * (nelx + 1) * (nely + 1).

        Note that each column of f is 1d vector describing nodes of the image.
        A nodal degree of freedom with coordinates (nx, ny, c) should have
        index i = 2 * ((nely + 1) * nx + ny) + c in the 1d array.

        If f has a single column, one can easily obtain a "2D" view by calling:

            fs = f.reshape((nelx + 1, nely + 1, 2), order='C')

        Parameters
        ----------
        nelx : int
            Width of the domain at the current level.
        nely : int
            Height of the domain at the current level.
        f : numpy.ndarray
            Force vector to be set, has size (ndof, 1).
        params : Struct
            Other parameters specific to this problem description.

        """
        pass

    @staticmethod
    def set_passive_elements(nelx, nely, lb, ub, params):
        """
        Set lower and upper bounds for passive elements.

        Note that lb and ub are flat (1d arrays), and an element with coordinate
        (ex, ey) should have index i = ey + ex * nely in the 1d array.

        A 2D view of each array compatible with this numbering can be obtained
        by calling:

            lbs = lb.reshape((nelx, nely), order='C')

        Parameters
        ----------
        nelx : int
            Width of the domain at the current level.
        nely : int
            Height of the domain at the current level.
        lb : numpy.ndarray
            Lower bounds to be set, given as a flat array.
        ub : numpy.ndarray
            Lower bounds to be set, given as a flat array.
        params : Struct
            Other parameters specific to this problem description.

        """
        pass
