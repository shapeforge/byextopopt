from common import ProblemType
from solver import Solver
from gui import Gui
import images


def multires(nelx, nely, params, bc):
    # Allocate design variables for the first level
    x = None
    x_comp = None

    # Dynamic parameters
    downsampling = 2**(params.numLevels - 1)
    params.exemplarDownsampling *= downsampling

    # Multires synthesis
    for level in range(params.numLevels):
        print("*** Level " + str(level))
        if x is not None:
            # Upsample previous solution
            x = images.upsample(x, nelx, nely)
            if params.problemType == ProblemType.AppearanceWithMaxCompliance:
                x_comp = images.upsample(x_comp, nelx, nely)
        gui = None
        if params.hasGui:
            gui = Gui(nelx, nely)
        if params.problemType == ProblemType.AppearanceWithMaxCompliance:
            params.complianceMax = 0
            solver = Solver(nelx, nely, params, ProblemType.Compliance, bc, gui)
            x_comp = solver.optimize(x_comp)
            min_compliance = solver.last_optimum_value()
            params.complianceMax = min_compliance * params.complianceMaxFactor
            print("")

        # Solve problem
        solver = Solver(nelx, nely, params, params.problemType, bc, gui)
        x = solver.optimize(x, enforce_constraints=(level > 0))
        if params.hasGui:
            solver.filtering.filter_variables(x, solver.x_phys)
            gui.update(solver.x_phys)

        # Go to next level
        if level < params.numLevels - 1:
            nelx *= 2
            nely *= 2
            params.exemplarDownsampling /= 2.0
            params.maxSolverStep //= 2
            params.lengthSquare /= 2.0
        print("")

    # Filter last result to obtain physical variables
    solver.filtering.filter_variables(x, solver.x_phys)
    results = {
        "last_optimum": solver.last_optimum_value(),
        "volume": sum(solver.x_phys) / len(solver.x_phys)}
    if params.problemType == ProblemType.AppearanceWithMaxCompliance:
        results["compliance_factor"] = solver.compliance_max / min_compliance
    return (solver.x_phys, nelx, nely, results)
