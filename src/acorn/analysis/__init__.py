"""
acorn.analysis — quantitative analysis utilities for segmented particles.
"""

from acorn.analysis.surface_area import (
    # single-particle estimation (2D projection methods)
    estimate_surface_area,
    batch_surface_area,
    SurfaceAreaResult,
    # 3D volume methods
    estimate_surface_area_3d,
    # population statistics
    compute_specific_surface_area,
    # diagnostic visualization
    plot_sa_diagnostics,
    METHOD_COLORS,
    # shape metric helpers
    compute_circularity,
    compute_convexity,
    compute_fractal_dimension,
    fit_ellipse_to_mask,
    contour_to_radial_signal,
    detect_spikes,
    # SA formula helpers
    ellipsoid_sa,
    cauchy_sa,
    fourier_sa,
    fourier_spiky_sa,
    # particle-type detection
    detect_hollow,
    detect_aggregate,
    # GPU diagnostics
    gpu_stats,
)
from acorn.analysis.tracking import (
    track_annotations,
    track_statistics,
    extract_centroids,
)

__all__ = [
    "estimate_surface_area",
    "batch_surface_area",
    "SurfaceAreaResult",
    "estimate_surface_area_3d",
    "compute_specific_surface_area",
    "plot_sa_diagnostics",
    "METHOD_COLORS",
    "compute_circularity",
    "compute_convexity",
    "compute_fractal_dimension",
    "fit_ellipse_to_mask",
    "contour_to_radial_signal",
    "detect_spikes",
    "ellipsoid_sa",
    "cauchy_sa",
    "fourier_sa",
    "fourier_spiky_sa",
    "detect_hollow",
    "detect_aggregate",
    "gpu_stats",
    "track_annotations",
    "track_statistics",
    "extract_centroids",
]
