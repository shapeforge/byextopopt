"""
Parameter class for topology optimization.
"""

# System libs
import json

# Local libs
import utils.json
from utils.types import Struct
from common import FilterType, InterpolationType, ProblemType


class Parameters(Struct):
    """
    Parameter class for topology optimization.

    Attributes
    ----------
    appearanceNormWeight : float
        Multiplication factor in order to normalize the appearance energy term.
    complianceMaxFactor : float
        Maximum deviation factor from the optimized compliance (e.g. 1.2 means 20% more compliance allowed).
    densityMin : int
        Minimum allowed density. Note that this concerns the design variables
        only (not the physical densities).
    exemplarDownsampling : float
        Exemplar image scaling.
    exemplarPath : str
        Path to the exemplar image file.
    exponentSimilarityMetric : float
        Exponent of the appearance energy.
    filterRadius : float
        Radius of the filter.
    filterType : Enum
        Type of the filter.
    hasGui : bool
        Show/hide the GUI.
    interpolationType : Enum
        Type of the interpolation scheme.
    lambdaOccurrenceMap : int
        Amount of enforced spatial uniformity patchmatch (see paper for details).
    maxSolverStep : int
        Maximum number of solver steps.
    neighborhoodSize : int
        Texture synthesis neighborhood size.
    numLevels : int
        Number of multiresolution levels.
    patchMatchIter : int
        Number of patchmatch iterations.
    penalty : float
        Penalty in the SIMP scheme.
    poissonCoeff : float
        Poisson's coefficient of the base material.
    problemModule : str
        Boundary conditions filename (inside the problems folder).
    problemOptions : dict
        Additional parameters passed to the problem.
    problemType : Enum
        Minimize appearance only, minimize compliance only, or minimize appearance
        with a compliance constraint.
    treshPedersen : float
        Value of the threshold in the Pedersen interpolation scheme.
    volumeFracMax : float
        Maximum volume fraction (incl. passive elements).
    volumeFracMin : int
        Minimum volume fraction (incl. passive elements).
    youngModulusMax : float
        Maximum Young's modulus of the base material.
    youngModulusMin : float
        Minimum Young's modulus of the base material.
    materialDensity : float
        Density of the material (g / mm^3). Set greater than 0.0 for self-weight problems.
    gravity : float
        Gravity (mm / s^2)
    lengthSquare : float
        Length of a square element in the coarser level (mm).
    thickness : float
        Thickness of the 2D domain (mm)
    """

    def __init__(self, *args, **kwargs):
        """
        Create an instance with defaults arguments.
        """

        # Set default parameters
        self.densityMin = 0.0
        self.youngModulusMin = 1e-9
        self.youngModulusMax = 1.0
        self.poissonCoeff = 0.35
        self.penalty = 3.0
        self.volumeFracMin = 0
        self.volumeFracMax = 0.5
        self.treshPedersen = 0.3  # Pedersen treshold value
        self.filterRadius = 1.5
        self.maxSolverStep = 40
        self.neighborhoodSize = 20
        self.lambdaOccurrenceMap = 20
        self.exponentSimilarityMetric = 1.2
        self.patchMatchIter = 30
        self.exemplarPath = ""
        self.exemplarDownsampling = 1.0
        self.numLevels = 3
        self.complianceMaxFactor = 1.2
        self.interpolationType = InterpolationType.SIMP
        self.filterType = FilterType.Density
        self.problemType = ProblemType.Compliance
        self.problemModule = "half_mbb_beam"
        self.problemOptions = Struct()
        self.hasGui = True
        self.appearanceNormWeight = 0.0  # Experimental
        self.materialDensity = 0.0
        self.gravity = 9.8
        self.lengthSquare = 1.0
        self.thickness = 1.0

        # Call parent class and update with user-given parameters
        super(Parameters, self).__init__(*args, **kwargs)

    def str2enum(self):
        """
        Convert top-level members from str to enum when its key ends with "Type".
        """
        for key, value in self.items():
            if key.endswith("Type"):
                name, member = value.split(".")
                e = getattr(globals()[name], member)
                setattr(self, key, e)

    def __str__(self):
        return json.dumps(self, cls=utils.json.SimpleEnumEncoder, sort_keys=True, indent=4)

    def save(self, filename):
        """
        Save parameters in a json file.
        """
        with open(filename, 'w') as json_file:
            encoder = utils.json.SimpleEnumEncoder
            json_file.write(json.dumps(self, cls=encoder, sort_keys=True, indent=4))

    @staticmethod
    def loads(filename):
        """
        Load parameters from a json file.
        """
        with open(filename, 'r') as json_file:
            res = Parameters(**json.load(json_file))
            res.str2enum()
            return res


if __name__ == "__main__":
    p = Parameters.loads('input/l_beam.json')
    print(p)
    p.save('temp.json')
