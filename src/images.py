import scipy.ndimage
from scipy import misc


def upsample(x, telx, tely):
    inter = 1  # 0 -> nearest, 1-> bilinear
    x_up = scipy.ndimage.zoom(x.reshape(telx // 2, tely // 2), (2.0, 2.0), order=inter, mode='nearest').flatten()
    return x_up


def downsample(rgb, scaling):
    inter = 1  # 0 -> nearest, 1-> bilinear
    rgb_down = scipy.ndimage.zoom(rgb, (1.0 / scaling, 1.0 / scaling, 1), order=inter, mode='nearest')
    return rgb_down


def load_exemplar(params):
    exemplar_img = misc.imread(params.exemplarPath)
    if params.exemplarDownsampling > 1:
        exemplar_img = downsample(exemplar_img, params.exemplarDownsampling)
    lower_bound = params.densityMin
    upper_bound = 1.0
    exemplar_rgb = lower_bound + (upper_bound - lower_bound) * \
        (1.0 - exemplar_img.transpose((1, 0, 2)).copy() / 255.0)
    return exemplar_rgb
