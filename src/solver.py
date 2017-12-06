# -*- coding: utf-8 -*-
# Third party libs
import numpy
import nlopt

# Local libs
import images
from common import ProblemType
from filter import Filter
from topopt import TopoptProblem
from appearance import AppearanceCL


class Solver(object):

    def __init__(self, nelx, nely, params, problem_type, bc, gui=None):
        """
        Allocate and initialize internal data structures.
        """
        n = nelx * nely
        self.nelx = nelx
        self.nely = nely
        self.opt = nlopt.opt(nlopt.LD_MMA, n)
        self.problem_type = problem_type

        # Alloc arrays
        self.x_phys = numpy.ones(n)
        self.x_rgb = numpy.ones((nelx, nely, 3))

        # Set bounds
        lb = numpy.array(params.densityMin * numpy.ones(n, dtype=float))
        ub = numpy.array(1.0 * numpy.ones(n, dtype=float))
        bc.set_passive_elements(nelx, nely, lb, ub, params.problemOptions)
        self.opt.set_upper_bounds(ub)
        self.opt.set_lower_bounds(lb)

        # Set stopping criteria
        self.opt.set_maxeval(params.maxSolverStep)
        self.opt.set_ftol_rel(0.0)

        # Setup topopt problem
        if problem_type.involves_compliance():
            self.problem = TopoptProblem(nelx, nely, params, bc)

        # Setup filter
        self.filtering = Filter(nelx, nely, params, problem_type)

        # Setup appearance matcher
        if problem_type.involves_appearance():
            self.exemplar_rgb = images.load_exemplar(params)
            self.appearance_cl = AppearanceCL(params.lambdaOccurrenceMap,
                                              params.exponentSimilarityMetric,
                                              params.appearanceNormWeight)

        # Setup user parameters
        self.params = params

        # Setup functional right-hand sides
        self.volume_max = self.params.volumeFracMax * n
        self.volume_min = self.params.volumeFracMin * n
        self.appearance_max = 0
        self.compliance_max = 0
        if problem_type.involves_compliance():
            if 'complianceMax' in params:
                self.compliance_max = params.complianceMax

        # Define optimization problem
        self.init_problem()

        # Set GUI callback
        self.gui = gui

    def init_problem(self):
        """
        Define objective function, constraint functions, and constraints rhs.
        """
        # Set objective and constraint functions
        self.opt.remove_inequality_constraints()
        if self.problem_type == ProblemType.Appearance:
            self.opt.set_min_objective(self.appearance_function)
        elif self.problem_type == ProblemType.Compliance:
            self.opt.set_min_objective(self.compliance_function)
            self.opt.add_inequality_constraint(self.volume_max_function)
            self.opt.add_inequality_constraint(self.volume_min_function)
        elif self.problem_type == ProblemType.AppearanceWithMaxCompliance:
            self.opt.set_min_objective(self.appearance_function)
            self.opt.add_inequality_constraint(self.compliance_function)
            self.opt.add_inequality_constraint(self.volume_max_function)
            self.opt.add_inequality_constraint(self.volume_min_function)

    def guess_initial_state(self):
        # Lower and upper bounds
        lb = self.opt.get_lower_bounds()
        ub = self.opt.get_upper_bounds()
        # Material budget
        passive = (lb == ub)
        num_passive = numpy.sum(passive)
        num_active = self.nelx * self.nely - num_passive
        budget = self.volume_max - numpy.sum(lb)
        active_frac = budget / num_active
        if active_frac < self.params.densityMin or active_frac > 1:
            assert False, "Unsatisfiable volume constraint"
        # Initial guess for non passive elements
        x = numpy.ones(self.nely * self.nelx, dtype=float) * active_frac
        return numpy.clip(x, lb, ub)

    def enforce_volume_constraint(self, x, volume_function):
        """
        Given a problem instance with a violated volume constraint g_v, an objective f,
        and possibly other constraints g_i, tries to minimize g_v under the constraint that
        the other functionals (excluding appearance) should not increase.
        """
        # If there is no volume constraint, or if it is satisfied already, do nothing
        if not self.problem_type.has_volume_constraint() or volume_function(x) <= 0:
            return x

        print("Enforce volume constraint")

        # Otherwise, define a new subproblem to minimize
        self.opt.remove_inequality_constraints()
        old_stopval = self.opt.get_stopval()
        self.opt.set_stopval(0)

        # Objective: volume
        self.opt.set_min_objective(volume_function)

        # Constraint: compliance
        old_compliance_max = self.compliance_max
        if self.problem_type.involves_compliance():
            self.compliance_max = 0
            self.compliance_max = self.compliance_function(x)
            self.opt.add_inequality_constraint(self.compliance_function)

        # Solve subproblem
        x = self.opt.optimize(x)

        # Restore previous state
        self.compliance_max = old_compliance_max
        self.opt.set_stopval(old_stopval)
        self.init_problem()
        return x

    def enforce_compliance_constraint(self, x):
        """
        Given a problem instance with a violated compliance constraint g_c, an objective f,
        and possibly other constraints g_i, tries to minimize g_c under the constraint that
        the other functionals (excluding appearance) should not increase.
        """

        # If there is no volume constraint, or if it is satisfied already, do nothing
        if not self.problem_type.has_compliance_constraint() or self.compliance_function(x) <= 0:
            return x

        print("Enforce compliance constraint")

        # Otherwise, define a new subproblem to minimize
        self.opt.remove_inequality_constraints()
        old_stopval = self.opt.get_stopval()
        self.opt.set_stopval(0)

        # Objective: compliance
        self.opt.set_min_objective(self.compliance_function)

        # Constraint: volume
        old_volume_max = self.volume_max
        old_volume_min = self.volume_min
        if self.problem_type.involves_volume():
            self.volume_max = 0
            self.volume_max = self.volume_max_function(x)
            self.opt.add_inequality_constraint(self.volume_max_function)
            self.volume_min = 0
            self.volume_min = self.volume_min_function(x)
            self.opt.add_inequality_constraint(self.volume_min_function)

        # Solve subproblem
        x = self.opt.optimize(x)

        # Restore previous state
        self.volume_max = old_volume_max
        self.volume_min = old_volume_min
        self.opt.set_stopval(old_stopval)
        self.init_problem()
        return x

    def optimize(self, x, enforce_constraints=True):
        print("* " + str(self.problem_type))
        lb = self.opt.get_lower_bounds()
        ub = self.opt.get_upper_bounds()

        # If first attempt, guess initial state
        if x is None:
            x = self.guess_initial_state()

        # Make sure bounds are strictly satisfied (avoid Nlopt crash)
        x = numpy.clip(x, lb, ub)
        print("* Initial volume = " + str(sum(x) / float(len(x))))

        # Enforce violated constraint by solving alternative subproblems
        if enforce_constraints:
            self.enforce_volume_constraint(x, self.volume_max_function)
            self.enforce_volume_constraint(x, self.volume_min_function)
            x = self.enforce_compliance_constraint(x)
            print("Enforcing constraints: done")

        # Launch optimization
        x = self.opt.optimize(x)
        print("* Last optimum value = " + str(self.opt.last_optimum_value()))
        return x

    def last_optimum_value(self):
        return self.opt.last_optimum_value()

    def compliance_function(self, x, dc=None):
        """
        Compliance function. When used as a constraint, corresponds to the inequality:
        >>> u.f - c_max ≤ 0
        """
        # Filter design variables
        self.filtering.filter_variables(x, self.x_phys)

        # Display physical variables
        if self.gui:
            self.gui.update(self.x_phys)

        # Setup and solve FE problem
        self.problem.compute_displacements(self.x_phys)

        # Compliance and sensitivity
        obj = self.problem.compute_compliance(self.x_phys, dc)

        # Sensitivity filtering
        if dc is not None:
            self.filtering.filter_compliance_sensitivities(self.x_phys, dc)

        print("- Compliance = %.3f" % (obj))

        return obj - self.compliance_max

    def volume_max_function(self, x, dv=None):
        """
        Maximum volume function. When used as a constraint, corresponds to the inequality:
        >>> volume(x_phys) - v_max ≤ 0
        """
        # Filter design variables
        self.filtering.filter_variables(x, self.x_phys)

        if dv is not None:
            # Volume sensitivities
            dv[:] = 1.0

            # Sensitivity filtering
            self.filtering.filter_volume_sensitivities(self.x_phys, dv)

        vol = sum(self.x_phys)

        print("- Volume = %.3f %%" % (vol * 100.0 / float(len(x))))

        return vol - self.volume_max

    def volume_min_function(self, x, dv=None):
        """
        Minimum volume function. When used as a constraint, corresponds to the inequality:
        >>> -volume(x_phys) + v_min ≤ 0
        """
        # Filter design variables
        self.filtering.filter_variables(x, self.x_phys)

        if dv is not None:
            # Volume sensitivities
            dv[:] = -1.0

            # Sensitivity filtering
            self.filtering.filter_volume_sensitivities(self.x_phys, dv)

        vol = sum(self.x_phys)

        print("- Volume = %.3f %%" % (vol * 100.0 / float(len(x))))

        return -vol + self.volume_min

    def appearance_function(self, x, da=None):
        """
        Appearance function. When used as a constraint, corresponds to the inequality:
        >>> app(x_phys) - a_max ≤ 0
        """
        # Filter design variables
        self.filtering.filter_variables(x, self.x_phys)

        # Display physical variables
        if self.gui:
            self.gui.update(self.x_phys)

        # Appearance and its gradient
        self.x_rgb[:] = numpy.reshape(
            numpy.repeat(self.x_phys.reshape((self.nelx, self.nely)), 3, axis=1), (self.nelx, self.nely, 3))
        patch_size = (self.params.neighborhoodSize, self.params.neighborhoodSize)
        pm_iter = self.params.patchMatchIter
        if da is not None:
            grad_reshape = da.reshape((self.nelx, self.nely))
        else:
            grad_reshape = None
        sim = self.appearance_cl.compute(self.x_rgb, self.exemplar_rgb, grad_reshape, patch_size, pm_iter)

        if da is not None:
            # Sensitivity filtering
            self.filtering.filter_appearance_sensitivities(self.x_phys, da)

        print("- Appearance = %.3f" % (sim))
        return sim - self.appearance_max
