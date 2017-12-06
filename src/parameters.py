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
        Description / TODO
    complianceMaxFactor : float
        Description
    densityMin : int
        Minimum allowed density. Note that this concerns the design variables
        only (not the physical densities).
    exemplarDownsampling : float
        Description / TODO
    exemplarPath : str
        Description / TODO
    exponentSimilarityMetric : float
        Description / TODO
    filterRadius : float
        Radius of the filter.
    filterType : Enum
        Type of the filter.
    hasGui : bool
        Description / TODO
    interpolationType : Enum
        Type of the interpolation scheme.
    lambdaOccurrenceMap : int
        Description / TODO
    maxSolverStep : int
        Description / TODO
    neighborhoodSize : int
        Description / TODO
    numLevels : int
        Description / TODO
    patchMatchIter : int
        Description / TODO
    penalty : float
        Penalty in the SIMP scheme.
    poissonCoeff : float
        Poisson's coefficient of the base material.
    problemModule : str
        Description / TODO
    problemOptions : dict
        Description / TODO
    problemType : TYPE
        Description / TODO
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
