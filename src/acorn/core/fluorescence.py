"""ImageJ-style threshold and watershed for bright fluorescence objects."""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max


def segment_bright(image, *, sigma=1.0, min_area=20, min_distance=5):
    """Return editable polygon vertices and the settings used for a 2D image."""
    image = np.asarray(image)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite, two-dimensional image")
    smooth = filters.gaussian(image, sigma=sigma, preserve_range=True)
    if smooth.max() == smooth.min():
        return [], {"threshold": None, "sigma": sigma, "min_area_px": min_area,
                    "min_distance_px": min_distance}
    threshold = float(filters.threshold_otsu(smooth))
    mask = morphology.remove_small_objects(smooth > threshold, min_size=min_area)
    distance = ndi.distance_transform_edt(mask)
    peaks = peak_local_max(distance, min_distance=min_distance, labels=mask,
                           exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = segmentation.watershed(-distance, markers, mask=mask)
    polygons = []
    for region in measure.regionprops(labels):
        if region.area < min_area:
            continue
        contours = measure.find_contours(np.pad(region.image, 1), .5)
        if contours:
            contour = max(contours, key=len) - 1
            contour += np.array(region.bbox[:2])
            vertices = [(float(x), float(y)) for y, x in contour]
            if len(vertices) >= 3:
                polygons.append(vertices)
    return polygons, {"threshold": threshold, "sigma": sigma,
                      "min_area_px": min_area, "min_distance_px": min_distance}
