"""Measurement engine — pure calculation, no rendering or GUI imports."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from acorn.core.annotations import (
    DistanceMeasurement,
    AngleMeasurement,
    ROIAnnotation,
)


@dataclass
class LineProfileResult:
    """Intensity profile along a line."""
    distances_nm: np.ndarray    # x-axis: distance from p1 in nm
    intensities: np.ndarray     # y-axis: normalised intensity values
    p1: tuple[float, float]
    p2: tuple[float, float]
    length_nm: float
    pixel_size: float           # nm/px used for calculation


class MeasurementEngine:
    """
    Stateless measurement calculator. All methods take image-pixel coordinates
    and use ``pixel_size`` (nm/px) for physical unit conversion.

    Parameters
    ----------
    pixel_size : float
        Calibrated pixel size in nm/px (from DM4Image.pixel_size).
    """

    def __init__(self, pixel_size: float = 1.0) -> None:
        self.pixel_size = pixel_size

    # ── distance ──────────────────────────────────────────────────────────────

    def distance(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        color: str = "#00FF88",
        calibrated: bool = True,
    ) -> DistanceMeasurement:
        """
        Euclidean distance between two image-pixel points, converted to nm.

        Returns a DistanceMeasurement that can be added to an AnnotationStore.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist_px = math.hypot(dx, dy)
        dist_nm = dist_px * self.pixel_size
        return DistanceMeasurement(
            p1=p1, p2=p2, distance_nm=dist_nm, distance_px=dist_px,
            calibrated=calibrated, color=color,
        )

    # ── angle ─────────────────────────────────────────────────────────────────

    def angle(
        self,
        p1: tuple[float, float],
        vertex: tuple[float, float],
        p2: tuple[float, float],
        color: str = "#00FF88",
    ) -> AngleMeasurement:
        """
        Angle at ``vertex`` formed by rays to ``p1`` and ``p2`` (degrees).

        Uses the law of cosines: cos θ = (v1·v2) / (|v1| |v2|).
        """
        v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
        v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 == 0 or mag2 == 0:
            deg = 0.0
        else:
            cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            deg = math.degrees(math.acos(cos_val))
        return AngleMeasurement(
            p1=p1, vertex=vertex, p2=p2, angle_deg=deg, color=color
        )

    # ── area / ROI statistics ─────────────────────────────────────────────────

    def roi_stats(
        self,
        vertices: list[tuple[float, float]],
        image: np.ndarray,
        color: str = "#00AAFF",
    ) -> ROIAnnotation:
        """
        Compute area and intensity statistics inside a polygon ROI.

        Parameters
        ----------
        vertices : list of (x, y) pixel coords defining the polygon
        image    : 2-D float array (normalised, same shape as displayed image)

        Returns
        -------
        ROIAnnotation with area_nm2 and stats = {mean, std, min, max, n_pixels}
        """
        from skimage.draw import polygon as sk_polygon

        h, w = image.shape[:2]
        xs = np.array([v[0] for v in vertices])
        ys = np.array([v[1] for v in vertices])

        # skimage polygon: (row, col) = (y, x)
        rr, cc = sk_polygon(ys, xs, shape=(h, w))

        if len(rr) == 0:
            stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_pixels": 0}
            area_nm2 = 0.0
        else:
            vals = image[rr, cc].astype(np.float64)
            stats = {
                "mean": float(vals.mean()),
                "std":  float(vals.std()),
                "min":  float(vals.min()),
                "max":  float(vals.max()),
                "n_pixels": int(len(vals)),
            }
            # Shoelace formula for polygon area in pixels, then convert to nm²
            n = len(xs)
            area_px = abs(
                sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))
            ) / 2.0
            area_nm2 = area_px * (self.pixel_size ** 2)

        return ROIAnnotation(
            vertices=list(vertices),
            area_nm2=area_nm2,
            stats=stats,
            color=color,
        )

    # ── line profile ──────────────────────────────────────────────────────────

    def line_profile(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        image: np.ndarray,
        linewidth: int = 1,
    ) -> LineProfileResult:
        """
        Intensity profile along the line from p1 to p2, in physical nm units.

        Uses ``skimage.measure.profile_line`` for sub-pixel accuracy with
        bilinear interpolation.

        Parameters
        ----------
        p1, p2    : (x, y) pixel coords of the line endpoints
        image     : 2-D float array (normalised)
        linewidth : averaging width perpendicular to the line (pixels)
        """
        from skimage.measure import profile_line

        # profile_line expects (row, col) = (y, x)
        src = (p1[1], p1[0])
        dst = (p2[1], p2[0])

        intensities = profile_line(
            image, src, dst,
            linewidth=linewidth,
            mode="reflect",
            order=1,            # bilinear
        ).astype(np.float64)

        n = len(intensities)
        length_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        length_nm = length_px * self.pixel_size
        distances_nm = np.linspace(0.0, length_nm, n)

        return LineProfileResult(
            distances_nm=distances_nm,
            intensities=intensities,
            p1=p1,
            p2=p2,
            length_nm=length_nm,
            pixel_size=self.pixel_size,
        )
