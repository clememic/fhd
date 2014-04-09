"""FHD descriptors module."""

import os

import numpy as np


class FHD(object):

    """FHistogram Decomposition descriptor."""

    LIB_FHISTOGRAMS = os.path.join(os.path.dirname(__file__),
                                   'libfhistograms_raster.so')

    def __init__(self, fhistograms, shape_force, spatial_force):
        """Create an FHD descriptor."""
        self.N = fhistograms.shape[0]
        self.num_dirs = fhistograms.shape[2]
        self.fhistograms = fhistograms
        self.shape_force = shape_force
        self.spatial_force = spatial_force

    def __getitem__(self, index):
        """Return FHistogram by index."""
        return self.fhistograms[index]

    @classmethod
    def compute_fhd(cls, layers, num_dirs=180, shape_force=0.0,
                    spatial_force=0.0):
        """Compute an FHD descriptor for given layers."""
        N = layers.shape[0]
        fhistograms = np.ndarray((N, N, num_dirs))
        for i in range(N):
            for j in range(i, N):
                fhistograms[i, j] = cls.compute_fhistogram(
                    layers[i], layers[j], num_dirs,
                    shape_force if i == j else spatial_force)
        return cls(fhistograms, shape_force, spatial_force)

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
