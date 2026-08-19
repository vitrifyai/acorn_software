from __future__ import annotations

import h5py
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
