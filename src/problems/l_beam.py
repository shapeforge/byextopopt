from . import base


class BoundaryConditions(base.BaseProblem):

    @staticmethod
    def get_fixed_nodes(nelx, nely, params):
        return ([(x, 0, 0) for x in range(0, int((nelx * 2) / 5 + 1))] +
                [(x, 0, 1) for x in range(0, int((nelx * 2) / 5 + 1))])

    @staticmethod
    def set_forces(nelx, nely, f, params):
        f[2 * ((nely + 1) * nelx + (nely * 3) // 5) + 1, 0] = -1

    @staticmethod
    def set_passive_elements(nelx, nely, lb, ub, params):
        for x in range(int((nelx * 2) / 5), nelx):
            for y in range(0, int((nely * 3) / 5)):
                ub[y + x * nely] = lb[y + x * nely]
