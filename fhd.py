"""FHD descriptors module."""

import math
import os

import numpy as np

def L1(A, B):
    """Manhattan distance."""
    return np.sum(np.abs(A - B))

def L2(A, B):
    """Euclidean distance."""
    return np.sqrt(np.sum((A - B) ** 2))

def CHI2(A, B):
    """Chi-squared distance."""
    return np.sum(np.nan_to_num(((A - B) ** 2) / (A + B)))

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

    def dump(self, filename):
        """Dump FHD descriptor to file."""
        np.savetxt(filename, self.fhistograms[np.triu_indices(self.N)])

    @classmethod
    def load(cls, filename, N, shape_force, spatial_force):
        fhistograms_from_file = np.loadtxt(filename)
        num_dirs = fhistograms_from_file.shape[-1]
        fhistograms = np.ndarray((N, N, num_dirs))
        fhistograms[np.triu_indices(N)] = fhistograms_from_file
        return cls(fhistograms, shape_force, spatial_force)

    @classmethod
    def compute_fhd(cls, layers, num_dirs=180, shape_force=0.0,
                    spatial_force=0.0):
        """Compute an FHD descriptor for given layers."""
        N = len(layers)
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

    @classmethod
    def distance(cls, A, B, alpha=None):
        """Distance between two FHD descriptors."""
        N = A.N
        if alpha is None:
            alpha = 1 - (2 / (N + 1))
        shape_dist = np.sum(L2(A[i, i], B[i, i]) for i in range(A.N))
        spatial_dist = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                spatial_dist += L2(A[i, j], B[i, j])
        return (alpha * shape_dist) + ((1 - alpha) * spatial_dist)


def meanshift(image, spatial_radius, range_radius, min_density):
    """Segment an image with meanshift."""
    import pymeanshift as pyms
    segm, labels, num_modes = pyms.segment(image, spatial_radius, range_radius,
                                           min_density, pyms.SPEEDUP_MEDIUM)
    return segm, num_modes


def kmeans(samples, num_clusters):
    """Perform kmeans clustering on given samples."""
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(samples)
    clusters = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_
    for index in range(samples.shape[0]):
        samples[index] = clusters[labels[index]]
    return samples, clusters

def binary_layers(segm, clusters):
    """Split a segmented image into binary layers."""
    N = clusters.shape[0]
    layers = [np.zeros((segm.shape[0], segm.shape[1]), np.uint8)
              for i in range(N)]
    from scipy.ndimage import binary_erosion
    for index, cluster in enumerate(clusters):
        layers[index][np.where((segm == cluster).all(segm.ndim - 1))] = 255
        layers[index] = binary_erosion(layers[index], np.ones((3, 3)))
    return layers
