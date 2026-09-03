from __future__ import annotations

import h5py
import pytest
import numpy as np

from acorn.core.dm4_loader import ALL_EXTS, DM4Image, write_hdf5_image


def test_emd_extension_is_supported() -> None:
    assert ".emd" in ALL_EXTS
    assert ".h5" in ALL_EXTS
    assert ".hdf5" in ALL_EXTS


def test_loads_velox_like_emd_hdf5(tmp_path) -> None:
    path = tmp_path / "velox_like.emd"
    data = np.arange(3 * 32 * 48, dtype=np.float32).reshape(3, 32, 48)
    with h5py.File(path, "w") as h5:
        ds = h5.create_dataset("/Data/Image/0/Data", data=data)
        ds.attrs["pixel_size"] = 0.42
        ds.attrs["pixel_unit"] = "nm"

    img = DM4Image.from_file(path)

    assert img.raw.shape == (32, 48)
    assert img.frames is not None
    assert img.frames.shape == (3, 32, 48)
    assert img.pixel_size == 0.42
    assert img.meta.pixel_size_from_header is True
    assert img.meta.all_tags["hdf5_dataset"] == "/Data/Image/0/Data"


def test_write_hdf5_image_round_trips(tmp_path) -> None:
    path = write_hdf5_image(tmp_path / "image.h5", np.ones((16, 20), dtype=np.float32), pixel_size_nm=1.25)

    img = DM4Image.from_file(path)

    assert img.raw.shape == (16, 20)
    assert img.pixel_size == 1.25


# ── vendor pixel size ────────────────────────────────────────────────────────

def _zeiss_tif(path, px_nm, size=(64, 64)):
    """A TIFF shaped like a Zeiss SEM export: real scale only in CZ_SEM."""
    import tifffile
    import numpy as np
    tifffile.imwrite(
        path, np.zeros(size, np.uint8),
        extratags=[(34118, "s", 0,
                    f"AP_IMAGE_PIXEL_SIZE\nImage Pixel Size = {px_nm} nm\n", True)],
        resolution=(1, 1),
    )


def test_zeiss_pixel_size_is_read_from_the_vendor_tag(tmp_path):
    """The failure this prevents.

    A Zeiss SEM writes XResolution (1,1) and ResolutionUnit 1 -- "no absolute
    unit" -- so a reader that understands only ImageJ/OME sees no calibration
    and falls back to whatever was last typed in. A real 26-image dataset was
    annotated that way, every sidecar carrying 1.867 nm/px while the true
    scales ranged over 3.6-19.3, which made every measurement in physical units
    wrong by a different factor per image.
    """
    from acorn.core.dm4_loader import _vendor_pixel_size_nm
    import tifffile

    # The parser is what matters; drive it directly with the tag structure
    # tifffile produces for a Zeiss file.
    class _Tag:
        def __init__(self, name, value): self.name, self.value = name, value

    class _Page:
        def __init__(self, tags): self.tags = type("T", (), {"values": lambda s: tags})()

    class _TF:
        def __init__(self, tags): self.pages = [_Page(tags)]

    cz = {"ap_image_pixel_size": ("Image Pixel Size", 19.29, "nm")}
    assert _vendor_pixel_size_nm(_TF([_Tag("CZ_SEM", cz)])) == pytest.approx(19.29)

    # micrometre units must be converted, not taken at face value
    cz_um = {"ap_image_pixel_size": ("Image Pixel Size", 1.5, "µm")}
    assert _vendor_pixel_size_nm(_TF([_Tag("CZ_SEM", cz_um)])) == pytest.approx(1500.0)

    # the unnamed numeric block is metres per pixel
    cz_raw = {"": (0, 0, 0, 1.928529e-08, 5789.0)}
    assert _vendor_pixel_size_nm(_TF([_Tag("CZ_SEM", cz_raw)])) == pytest.approx(19.28529)

    # Thermo/FEI writes metres under Scan/PixelWidth
    fei = {"Scan": {"PixelWidth": 1.34896e-08}}
    assert _vendor_pixel_size_nm(_TF([_Tag("FEI_HELIOS", fei)])) == pytest.approx(13.4896)

    # nothing to read is None, not a guess
    assert _vendor_pixel_size_nm(_TF([_Tag("XResolution", (1, 1))])) is None


def test_a_vendor_scale_is_not_overwritten_by_an_empty_resolution_field():
    """Zeiss files carry BOTH a real vendor scale and a blank XResolution.
    Reading them in the wrong order throws the real value away."""
    from acorn.core import dm4_loader
    src = dm4_loader.__file__
    text = open(src).read()
    i_vendor = text.index("_vendor_pixel_size_nm(tf)")
    i_imagej = text.index("Then ImageJ / OME-TIFF metadata")
    assert i_vendor < i_imagej, "vendor tags must be read before ImageJ metadata"
    assert "not self.meta.pixel_size_from_header" in text, \
        "the ImageJ branch must defer to a vendor value"
