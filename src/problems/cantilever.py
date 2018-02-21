from . import base


class BoundaryConditions(base.BaseProblem):

    @staticmethod
    def get_fixed_nodes(nelx, nely, params):
        return (
            [0, int(nely / 2), 0], [0, int(nely / 2), 1],
            [int(nelx * 2 / 3), int(nely / 2), 0], [int(nelx * 2 / 3), int(nely / 2), 1])

    @staticmethod
    def set_forces(nelx, nely, f, params):
        f[2 * int((nely + 1) * nelx + nely / 2) + 1, 0] = -1.0

    @staticmethod
    def set_passive_elements(nelx, nely, lb, ub, params):
        pass
