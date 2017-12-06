from . import base


class BoundaryConditions(base.BaseProblem):

    @staticmethod
    def get_fixed_nodes(nelx, nely, params):
        return [(0, y, 0) for y in range(0, nely + 1)] + [(nelx, nely, 1)]

    @staticmethod
    def set_forces(nelx, nely, f, params):
        f[1, 0] = -1

    @staticmethod
    def set_passive_elements(nelx, nely, lb, ub, params):
        pass
