"""COR 2 & COR 1"""

import numpy as np
from .utils import FITS

from . import __version__

# class downloader():


"""level 0.5"""


def level_05(src_path, trg_path):
    img, header = FITS(src_path)

    return img, header


"""level 1"""


def level_1(src_path, trg_path):
    return src_path
