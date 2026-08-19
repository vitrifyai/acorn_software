from __future__ import annotations

import tifffile

from acorn_fib_sim.io import generate_fib_dataset
from acorn_fib_sim.plugin import _fresh_run_dir, _open_paths_in_acorn, _params_from_clu
from acorn_fib_sim.simulator import ImagingConfig, MillingConfig, SimConfig, simulate


def test_fib_simulator_is_seeded() -> None:
    cfg = SimConfig(imaging=ImagingConfig(shape=(48, 64), seed=3))

    a = simulate(cfg)
    b = simulate(cfg)

    assert a.image.shape == (48, 64)
    assert a.truth.shape == (48, 64)
    assert (a.image == b.image).all()
    assert a.image.std() > 0


def test_generate_fib_dataset_writes_layers(tmp_path) -> None:
    cfg = SimConfig(
        sample="material",
        imaging=ImagingConfig(shape=(48, 64), seed=5),
        milling=MillingConfig(curtain_strength=0.2),
    )

    paths = generate_fib_dataset(tmp_path, 1, cfg)

    assert len(paths) == 1
    assert paths[0].exists()
    assert tifffile.imread(paths[0]).shape == (48, 64)
    assert (tmp_path / "layers" / "fibsim_00000_truth.tif").exists()
    assert (tmp_path / "metadata.json").exists()


def test_clu_params_map_to_fib_sim_request(tmp_path) -> None:
    params = _params_from_clu(
        {
            "output_dir": str(tmp_path),
            "sample": "materials",
            "count": 3,
            "preset": "low dose liftout",
        }
    )

    assert params["sample"] == "material"
    assert params["count"] == 3
    assert params["liftout"] is True
    assert params["needle"] is True
    assert params["electrons_per_pixel"] == 80.0


def test_fib_plugin_uses_fresh_run_directories(tmp_path) -> None:
    first = _fresh_run_dir(tmp_path, "fib_run")
    first.mkdir(parents=True)
    second = _fresh_run_dir(tmp_path, "fib_run")

    assert first != second
    assert first.parent == tmp_path
    assert second.parent == tmp_path


def test_fib_auto_open_reports_missing_outputs(tmp_path) -> None:
    class Context:
        def _w(self):
            raise AssertionError("window should not be used without files")

    count, message = _open_paths_in_acorn(Context(), [str(tmp_path / "missing.tif")])

    assert count == 0
    assert "no output TIFF files" in message
