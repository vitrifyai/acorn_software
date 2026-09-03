from __future__ import annotations

import tifffile

from acorn_tem_sim.io import generate_tem_dataset
from acorn_tem_sim.plugin import _params_from_clu
from acorn_sim_common import fresh_run_dir, open_paths_in_acorn
from acorn_tem_sim.simulator import Specimen, make_potential, resolve, simulate_micrograph


def test_tem_simulator_is_seeded() -> None:
    cfg = resolve({"pixel_size_a": 1.5, "image_size_px": 96, "total_dose_e_per_a2": 20})
    specimen = Specimen(n_particles=6, diameter_nm_mean=18, diameter_nm_sd=3, seed=4)

    a = simulate_micrograph(cfg, specimen, defocus_um=-1.5, seed=7)
    b = simulate_micrograph(cfg, specimen, defocus_um=-1.5, seed=7)

    assert a.image.shape == (96, 96)
    assert a.potential.shape == (96, 96)
    assert (a.image == b.image).all()
    assert a.label.max() > 0


def test_generate_tem_dataset_writes_layers(tmp_path) -> None:
    cfg = resolve({"pixel_size_a": 1.5, "image_size_px": 64, "total_dose_e_per_a2": 20})
    specimen = Specimen(n_particles=4, diameter_nm_mean=16, diameter_nm_sd=2, seed=5)

    paths = generate_tem_dataset(tmp_path, 1, cfg, specimen, seed=5)

    assert len(paths) == 1
    assert paths[0].exists()
    assert tifffile.imread(paths[0]).shape == (64, 64)
    assert (tmp_path / "layers" / "temsim_00000_potential.tif").exists()
    assert (tmp_path / "metadata.json").exists()


def test_clu_params_map_to_tem_sim_request(tmp_path) -> None:
    params = _params_from_clu(
        {
            "output_dir": str(tmp_path),
            "count": 3,
            "preset": "low dose",
            "particles": 12,
            "diameter_nm": 25,
        }
    )

    assert params["count"] == 3
    assert params["total_dose_e_per_a2"] == 8.0
    assert params["n_particles"] == 12
    assert params["diameter_nm_mean"] == 25.0


def test_clu_params_map_tem_simulation_path() -> None:
    assert _params_from_clu({"backend": "physics"})["simulation_path"] == "multislice"
    assert _params_from_clu({"simulation_path": "custom"})["simulation_path"] == "custom"
    assert _params_from_clu({})["simulation_path"] == "fast"


def test_clu_params_map_tem_specimen_kind() -> None:
    assert _params_from_clu({"sample": "single lipid vesicle"})["specimen_kind"] == "lipid_single"
    assert _params_from_clu({"sample": "multilamellar vesicle"})["specimen_kind"] == "lipid_multi"
    assert _params_from_clu({"sample": "bacteria"})["specimen_kind"] == "bacteria"
    assert _params_from_clu({"sample": "protein from pdb"})["specimen_kind"] == "pdb"


def test_tem_specimen_kinds_generate_nonblank_potentials() -> None:
    for kind in ("plga", "lipid_single", "lipid_multi", "protein", "bacteria"):
        specimen = Specimen(kind=kind, n_particles=3, diameter_nm_mean=24, diameter_nm_sd=2, seed=9)
        result = make_potential((96, 96), 1.5, specimen)

        assert result.v_proj.shape == (96, 96)
        assert result.label.shape == (96, 96)
        assert result.v_proj.std() > 0


def test_tem_pdb_projection_uses_local_atom_model(tmp_path) -> None:
    pdb = tmp_path / "mini.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLY A   1      -6.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  GLY A   1       0.000   4.000   1.000  1.00 20.00           C",
                "ATOM      3  C   GLY A   1       6.000   0.000  -1.000  1.00 20.00           C",
                "ATOM      4  O   GLY A   1       0.000  -4.000   0.500  1.00 20.00           O",
            ]
        )
    )
    specimen = Specimen(
        kind="pdb",
        n_particles=2,
        pdb_path=str(pdb),
        oligomer_count=3,
        solvent_noise=0,
        seed=12,
    )

    result = make_potential((96, 96), 1.5, specimen)

    assert result.v_proj.std() > 0
    assert result.label.max() > 0


def test_tem_multislice_path_generates_image(tmp_path) -> None:
    cfg = resolve(
        {
            "simulation_path": "multislice",
            "pixel_size_a": 2.0,
            "image_size_px": 64,
            "total_dose_e_per_a2": 20,
            "slice_thickness_a": 5.0,
        }
    )
    specimen = Specimen(kind="plga", n_particles=3, diameter_nm_mean=16, diameter_nm_sd=2, seed=5)

    image = simulate_micrograph(cfg, specimen, seed=6)

    assert image.image.shape == (64, 64)
    assert image.potential.shape == (64, 64)
    assert image.image.std() > 0


def test_tem_custom_script_path_generates_dataset(tmp_path) -> None:
    script = tmp_path / "custom_tem.py"
    script.write_text(
        "import numpy as np\n"
        "def generate_image(cfg, specimen, seed, index):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    return rng.random((32, 32), dtype=np.float32)\n"
    )
    cfg = resolve(
        {
            "simulation_path": "custom",
            "custom_script_path": str(script),
            "image_size_px": 32,
        }
    )

    paths = generate_tem_dataset(tmp_path / "out", 2, cfg, Specimen(), seed=4)

    assert len(paths) == 2
    assert all(path.exists() for path in paths)


def test_tem_plugin_uses_fresh_run_directories(tmp_path) -> None:
    first = fresh_run_dir(tmp_path, "tem_run")
    first.mkdir(parents=True)
    second = fresh_run_dir(tmp_path, "tem_run")

    assert first != second
    assert first.parent == tmp_path
    assert second.parent == tmp_path


def test_tem_auto_open_reports_missing_outputs(tmp_path) -> None:
    class Context:
        def _w(self):
            raise AssertionError("window should not be used without files")

    count, message = open_paths_in_acorn(Context(), [str(tmp_path / "missing.tif")])

    assert count == 0
    assert "no output TIFF files" in message


# ── debris ───────────────────────────────────────────────────────────────────

def test_debris_is_chain_like_and_never_a_single_round_mass():
    """The failure this prevents, measured on real data.

    Debris built from compact particle-sized blobs is indistinguishable from a
    small particle, and a detector trained against it learns to suppress
    genuine small particles rather than to reject debris. Across three
    successively more faithful debris models the smallest particle the detector
    would find on real PLGA micrographs rose from 24 nm to 44 nm to 54 nm.
    Debris must be recognisably elongated and many-lobed.
    """
    import numpy as np
    from acorn_tem_sim.engine.scene import Debris

    d = Debris(n=40, extent_nm=120.0, grain_nm=7.0)
    d.prepare((512, 512), 18.0, 600.0, np.random.default_rng(0))
    assert len(d._blobs) == 40

    elongations, lobes = [], []
    for specks, _z0, _t in d._blobs:
        lobes.append(len(specks))
        pts = np.array([(x, y) for (y, x, _r) in specks], float)
        pts = pts - pts.mean(0)
        _, sv, _ = np.linalg.svd(pts, full_matrices=False)
        elongations.append(sv[0] / max(sv[1], 1e-6))

    assert min(lobes) >= 6, f"aggregates too simple: {min(lobes)} lobes"
    assert np.median(elongations) > 1.6, (
        f"aggregates are compact (median elongation {np.median(elongations):.2f}); "
        "a round debris clump reads as a small particle")


def test_debris_is_labelled_so_a_detector_can_discriminate():
    """Debris carries a label of its own.

    It was first emitted unlabelled, so it would read as background. That made
    real-world performance worse in every test, because the detector learned to
    suppress anything that looked like a debris speck -- and a speck looks like
    a small particle. Given its own class instead, the share of real detections
    that are actually particle-shaped rose from 70% to 91%.
    """
    import numpy as np
    from acorn_tem_sim.engine.scene import Debris

    d = Debris(n=5)
    d.prepare((256, 256), 18.0, 600.0, np.random.default_rng(1))
    fps = d.footprints(18.0)
    assert len(fps) == 5
    assert {label for label, _v in fps} == {"debris"}
    for _label, verts in fps:
        assert len(verts) >= 3


def test_debris_and_particles_are_separate_ground_truth_classes():
    """A scene with both must not merge them into one label."""
    import numpy as np
    from acorn_tem_sim.engine.scene import (Scene, Nanoparticles, Debris,
                                            scene_annotations)

    scene = Scene(ice_thickness_nm=60.0, seed=2,
                  components=[Nanoparticles(n=12, diameter_nm_mean=40.0),
                              Debris(n=6)])
    labels = [lab for lab, _v in scene_annotations((256, 256), 18.0, scene)]
    assert labels.count("nanoparticle") == 12
    assert labels.count("debris") == 6
