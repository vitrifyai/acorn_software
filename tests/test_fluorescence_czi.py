import numpy as np
import pytest

from acorn.core.dm4_loader import DM4Image
from acorn.core.fluorescence import segment_bright


def test_bright_objects_are_separate_and_parameters_recorded():
    y, x = np.ogrid[:100, :100]
    image = (((x - 25) ** 2 + (y - 50) ** 2 < 12 ** 2) |
             ((x - 75) ** 2 + (y - 50) ** 2 < 12 ** 2)).astype(float)
    polygons, settings = segment_bright(image)
    assert len(polygons) == 2
    assert all(len(p) >= 3 for p in polygons)
    assert settings["threshold"] is not None


def test_flat_and_invalid_images():
    assert segment_bright(np.zeros((10, 10)))[0] == []
    with pytest.raises(ValueError):
        segment_bright(np.zeros((2, 3, 4)))


def test_czi_selects_plane_and_preserves_calibration(monkeypatch, tmp_path):
    from xml.etree import ElementTree
    import aicspylibczi

    class FakeCzi:
        meta = ElementTree.fromstring('''<Root><Scaling><Items>
            <Distance Id="X"><Value>7e-8</Value></Distance>
            <Distance Id="Y"><Value>7e-8</Value></Distance>
            </Items></Scaling><Information><Image><Dimensions><Channels>
            <Channel Name="transmitted"/><Channel Name="EGFP"/>
            </Channels></Dimensions></Image></Information></Root>''')

        def __init__(self, path):
            pass

        def get_dims_shape(self):
            return [{"C": (0, 2), "Z": (0, 3), "Y": (0, 4), "X": (0, 5)}]

        def read_image(self, **selection):
            assert selection in ({"C": 0, "Z": 0}, {"C": 1, "Z": 0})
            return np.ones((1, 1, 4, 5)), [("C", 1), ("Z", 1), ("Y", 4), ("X", 5)]

    monkeypatch.setattr(aicspylibczi, "CziFile", FakeCzi)
    image = DM4Image.from_file(tmp_path / "image.czi")
    assert image.raw.shape == (4, 5)
    assert image.meta.pixel_size == pytest.approx(70)
    assert image.meta.pixel_size_from_header
    assert image.meta.all_tags["czi_selection"] == {"C": 1, "Z": 0}
    assert image.meta.all_tags["czi_channels"] == ["transmitted", "EGFP"]
    selected = DM4Image.from_file(tmp_path / "image.czi", czi_channel=0)
    assert selected.meta.all_tags["czi_selection"]["C"] == 0
    with pytest.raises(ValueError, match="out of range"):
        DM4Image.from_file(tmp_path / "image.czi", czi_channel=3)
