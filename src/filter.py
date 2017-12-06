import numpy
from scipy.sparse import coo_matrix
from common import FilterType


class Filter(object):

    def prepare_filter(self, nelx, nely, rmin):
        """
        Build (and assemble) the index+data vectors for the coo matrix format.
        """
        nfilter = int(nelx * nely * ((2 * (numpy.ceil(rmin) - 1) + 1)**2))
        iH = numpy.zeros(nfilter)
        jH = numpy.zeros(nfilter)
        sH = numpy.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j
                kk1 = int(numpy.maximum(i - (numpy.ceil(rmin) - 1), 0))
                kk2 = int(numpy.minimum(i + numpy.ceil(rmin), nelx))
                ll1 = int(numpy.maximum(j - (numpy.ceil(rmin) - 1), 0))
                ll2 = int(numpy.minimum(j + numpy.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * nely + l
                        fac = rmin - numpy.sqrt(((i - k) * (i - k) + (j - l) * (j - l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = numpy.maximum(0.0, fac)
                        cc = cc + 1
        # Finalize assembly and convert to csc format
        self.H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
        self.Hs = self.H.sum(1)

    def __init__(self, nelx, nely, params, problem_type):
        """
        Create a new filter engine.
        """
        self.params = params
        if params.filterType != FilterType.NoFilter:
            if params.filterType == FilterType.Density or problem_type.involves_compliance():
                self.prepare_filter(nelx, nely, params.filterRadius)

    def filter_variables(self, x, x_phys):
        """
        Filter design variables.
        """
        if self.params.filterType in (FilterType.NoFilter, FilterType.Sensitivity):
            x_phys[:] = x
        elif self.params.filterType == FilterType.Density:
            x_phys[:] = numpy.asarray(self.H * x[numpy.newaxis].T / self.Hs)[:, 0]

    def filter_compliance_sensitivities(self, x_phys, dc):
        """
        Filter gradient of the compliance.
        """
        if self.params.filterType == FilterType.NoFilter:
            pass
        elif self.params.filterType == FilterType.Sensitivity:
            dc[:] = (numpy.asarray((self.H * (x_phys * dc))[numpy.newaxis].T / self.Hs)[:, 0] /
                     numpy.maximum(0.001, x_phys))
        elif self.params.filterType == FilterType.Density:
            dc[:] = numpy.asarray(self.H * (dc[numpy.newaxis].T / self.Hs))[:, 0]

    def filter_volume_sensitivities(self, dummy_x_phys, dv):
        """
        Filter gradient of the volume.
        """
        if self.params.filterType in (FilterType.NoFilter, FilterType.Sensitivity):
            pass
        elif self.params.filterType == FilterType.Density:
            dv[:] = numpy.asarray(self.H * (dv[numpy.newaxis].T / self.Hs))[:, 0]

    def filter_appearance_sensitivities(self, dummy_x_phys, da):
        """
        Filter gradient of the appearance energy.
        """
        if self.params.filterType in (FilterType.NoFilter, FilterType.Sensitivity):
            pass
        elif self.params.filterType == FilterType.Density:
            da[:] = numpy.asarray(self.H * (da[numpy.newaxis].T / self.Hs))[:, 0]
