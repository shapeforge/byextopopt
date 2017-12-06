from . import base


class BoundaryConditions(base.BaseProblem):

    @staticmethod
    def get_fixed_nodes(nelx, nely, params):
        return ([0, nely, 0], [0, nely, 1], [nelx, nely, 0], [nelx, nely, 1])

    @staticmethod
    def set_forces(nelx, nely, f, params):
        return

    @staticmethod
    def set_passive_elements(nelx, nely, lb, ub, params):
        return
