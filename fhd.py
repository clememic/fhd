"""FHD descriptors module."""

import os

import numpy as np


class FHD(object):

    """FHistogram Decomposition descriptor."""

    LIB_FHISTOGRAMS = os.path.join(os.path.dirname(__file__),
                                   'libfhistograms_raster.so')

    def __init__(self):
        pass

    @classmethod
    def compute_fhistogram(cls, A, B, num_dirs=180, force_type=0.0):
        """Compute an FHistogram between two layers A and B."""
        # Load C shared library
        import ctypes
        clib = ctypes.cdll.LoadLibrary(cls.LIB_FHISTOGRAMS)
        # Compute FHistogram between A and B
        fhistogram = np.ndarray(num_dirs)
        height, width = A.shape
        clib.FRHistogram_CrispRaster(
            fhistogram.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(num_dirs),
            ctypes.c_double(force_type),
            A.ctypes.data_as(ctypes.c_char_p),
            B.ctypes.data_as(ctypes.c_char_p),
            ctypes.c_int(width),
            ctypes.c_int(height))
        return fhistogram
