"""Physics-based cryo-TEM micrograph simulator.

Forward model:

    specimen -> projected potential -> CTF -> dose/detector

Specimen models are simplified projected-potential phantoms in vitreous ice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

_H = 6.62607015e-34
_ME = 9.1093837015e-31
_E = 1.602176634e-19
_C = 299792458.0


def wavelength_pm(voltage_kv: float) -> float:
    v = voltage_kv * 1e3
    lam_m = _H / math.sqrt(2 * _ME * _E * v * (1 + _E * v / (2 * _ME * _C**2)))
    return lam_m * 1e12


def sigma_rad_per_va(voltage_kv: float) -> float:
    v = voltage_kv * 1e3
    gamma = 1 + _E * v / (_ME * _C**2)
    m = gamma * _ME
    lam_m = wavelength_pm(voltage_kv) * 1e-12
    return 2 * math.pi * m * _E * lam_m / _H**2 * 1e-10


PRESETS = {
    "krios-k3": {"voltage_kv": "300", "cs_mm": 2.7, "detector_model": "K3"},
    "glacios-falcon4": {"voltage_kv": "200", "cs_mm": 2.7, "detector_model": "Falcon4"},
    "talos-ceta": {"voltage_kv": "200", "cs_mm": 2.7, "detector_model": "Ceta"},
    "cs-corrected": {"voltage_kv": "300", "cs_mm": 0.01, "detector_model": "K3"},
}


DEFAULTS = {
    "simulation_path": "fast",
    "custom_script_path": "",
    "slice_thickness_a": 5.0,
    "pixel_size_a": 1.5,
    "voltage_kv": "300",
    "cs_mm": 2.7,
    "amplitude_contrast": 0.10,
    "total_dose_e_per_a2": 40.0,
    "cc_mm": 2.7,
    "energy_spread_ev": 0.9,
    "convergence_mrad": 0.10,
    "detector_model": "K3",
    "image_size_px": 512,
    "defocus_min_um": -1.0,
    "defocus_max_um": -2.5,
    "phase_plate": False,
    "energy_filter_ev": 0.0,
}

_MTF_SIGMA = {"K3": 0.5, "Falcon4": 0.5, "K2": 0.7, "Ceta": 1.2}
MIP_VITREOUS_ICE = 4.5
MIP_PLGA = 8.0
MIP_LIPID = 9.0
MIP_PROTEIN = 7.5
MIP_BACTERIA = 6.5
_ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "FE": 26,
    "ZN": 30,
    "SE": 34,
    "I": 53,
}


def resolve(answers: dict | None = None, preset: str = "krios-k3") -> dict:
    values = {**DEFAULTS, **PRESETS.get(preset, {}), **(answers or {})}
    values["image_size_px"] = int(values["image_size_px"])
    values["simulation_path"] = str(values.get("simulation_path", "fast")).strip().lower()
    values["custom_script_path"] = str(values.get("custom_script_path", ""))
    values["slice_thickness_a"] = float(values["slice_thickness_a"])
    values["pixel_size_a"] = float(values["pixel_size_a"])
    values["cs_mm"] = float(values["cs_mm"])
    values["amplitude_contrast"] = float(values["amplitude_contrast"])
    values["total_dose_e_per_a2"] = float(values["total_dose_e_per_a2"])
    values["cc_mm"] = float(values["cc_mm"])
    values["energy_spread_ev"] = float(values["energy_spread_ev"])
    values["convergence_mrad"] = float(values["convergence_mrad"])
    values["defocus_min_um"] = float(values["defocus_min_um"])
    values["defocus_max_um"] = float(values["defocus_max_um"])
    values["energy_filter_ev"] = float(values["energy_filter_ev"])
    values["phase_plate"] = _as_bool(values["phase_plate"])
    values["voltage_kv"] = str(int(float(values["voltage_kv"])))
    if values["defocus_max_um"] > values["defocus_min_um"]:
        values["defocus_min_um"], values["defocus_max_um"] = values["defocus_max_um"], values["defocus_min_um"]
    kv = float(values["voltage_kv"])
    values["_derived"] = {
        "wavelength_pm": round(wavelength_pm(kv), 4),
        "sigma_rad_per_VA": sigma_rad_per_va(kv),
        "preset": preset,
    }
    return values


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@dataclass(frozen=True)
class Specimen:
    kind: str = "plga"
    ice_thickness_nm: float = 40.0
    n_particles: int = 40
    diameter_nm_mean: float = 30.0
    diameter_nm_sd: float = 8.0
    plga_mip_v: float = MIP_PLGA
    ice_mip_v: float = MIP_VITREOUS_ICE
    lipid_mip_v: float = MIP_LIPID
    protein_mip_v: float = MIP_PROTEIN
    bacteria_mip_v: float = MIP_BACTERIA
    membrane_thickness_nm: float = 4.0
    oligomer_count: int = 1
    pdb_path: str = ""
    solvent_noise: float = 6.0
    solvent_corr_a: float = 3.5
    allow_overlap: bool = False
    seed: int = 0


@dataclass(frozen=True)
class PotentialResult:
    v_proj: np.ndarray
    label: np.ndarray
    specimen: Specimen


@dataclass(frozen=True)
class Micrograph:
    image: np.ndarray
    counts: np.ndarray
    ideal: np.ndarray
    potential: np.ndarray
    label: np.ndarray
    defocus_um: float
    config: dict = field(default_factory=dict)


def _sphere_chord(shape: tuple[int, int], cy: float, cx: float, r_px: float):
    h, w = shape
    y0, y1 = max(0, int(cy - r_px - 1)), min(h, int(cy + r_px + 2))
    x0, x1 = max(0, int(cx - r_px - 1)), min(w, int(cx + r_px + 2))
    if y0 >= y1 or x0 >= x1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    inside = r2 < r_px**2
    chord = np.zeros_like(r2)
    chord[inside] = 2.0 * np.sqrt(r_px**2 - r2[inside])
    return (y0, y1, x0, x1), chord


def _capsule_mask(shape: tuple[int, int], cy: float, cx: float, length_px: float, radius_px: float, angle_rad: float):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = xx - cx
    y = yy - cy
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    u = x * ca + y * sa
    v = -x * sa + y * ca
    half = max(radius_px, length_px * 0.5 - radius_px)
    core = (np.abs(u) <= half) & (np.abs(v) <= radius_px)
    cap1 = (u + half) ** 2 + v**2 <= radius_px**2
    cap2 = (u - half) ** 2 + v**2 <= radius_px**2
    return core | cap1 | cap2


def _place_solid_particles(v_proj: np.ndarray, label: np.ndarray, pixel_size_a: float, specimen: Specimen, rng) -> None:
    d_excess = specimen.plga_mip_v - specimen.ice_mip_v
    placed: list[tuple[float, float, float]] = []
    tries = 0
    while len(placed) < specimen.n_particles and tries < max(1, specimen.n_particles * 50):
        tries += 1
        d_nm = rng.normal(specimen.diameter_nm_mean, specimen.diameter_nm_sd)
        if d_nm <= 2:
            continue
        r_px = (d_nm * 10.0 / 2.0) / pixel_size_a
        cy, cx = rng.uniform(0, v_proj.shape[0]), rng.uniform(0, v_proj.shape[1])
        if not specimen.allow_overlap and any(np.hypot(cy - py, cx - px) < (r_px + pr) for py, px, pr in placed):
            continue
        res = _sphere_chord(v_proj.shape, cy, cx, r_px)
        if res is None:
            continue
        (y0, y1, x0, x1), chord_px = res
        chord_a = chord_px * pixel_size_a
        v_proj[y0:y1, x0:x1] += (d_excess * chord_a).astype(np.float32)
        label[y0:y1, x0:x1] = np.maximum(label[y0:y1, x0:x1], chord_a)
        placed.append((cy, cx, r_px))


def _place_lipid_vesicles(v_proj: np.ndarray, label: np.ndarray, pixel_size_a: float, specimen: Specimen, rng, layers: int) -> None:
    d_excess = specimen.lipid_mip_v - specimen.ice_mip_v
    thickness_px = max(1.0, specimen.membrane_thickness_nm * 10.0 / pixel_size_a)
    spacing_px = max(thickness_px * 2.5, 7.0 * 10.0 / pixel_size_a)
    yy, xx = np.mgrid[0:v_proj.shape[0], 0:v_proj.shape[1]].astype(np.float32)
    placed: list[tuple[float, float, float]] = []
    tries = 0
    while len(placed) < specimen.n_particles and tries < max(1, specimen.n_particles * 60):
        tries += 1
        d_nm = rng.normal(specimen.diameter_nm_mean, specimen.diameter_nm_sd)
        if d_nm <= 4:
            continue
        r_outer = (d_nm * 10.0 / 2.0) / pixel_size_a
        cy, cx = rng.uniform(0, v_proj.shape[0]), rng.uniform(0, v_proj.shape[1])
        if not specimen.allow_overlap and any(np.hypot(cy - py, cx - px) < (r_outer + pr) for py, px, pr in placed):
            continue
        for layer in range(layers):
            r1 = r_outer - layer * spacing_px
            r0 = r1 - thickness_px
            if r0 <= 2:
                break
            dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
            outer = np.zeros(v_proj.shape, dtype=np.float32)
            inner = np.zeros(v_proj.shape, dtype=np.float32)
            outer_mask = dist2 < r1**2
            inner_mask = dist2 < r0**2
            outer[outer_mask] = 2.0 * np.sqrt(r1**2 - dist2[outer_mask])
            inner[inner_mask] = 2.0 * np.sqrt(r0**2 - dist2[inner_mask])
            shell = np.maximum(outer - inner, 0)
            shell_a = shell * pixel_size_a
            v_proj += (d_excess * shell_a).astype(np.float32)
            label[:] = np.maximum(label, shell_a)
        placed.append((cy, cx, r_outer))


def _place_bacteria(v_proj: np.ndarray, label: np.ndarray, pixel_size_a: float, specimen: Specimen, rng) -> None:
    d_excess = specimen.bacteria_mip_v - specimen.ice_mip_v
    for _ in range(specimen.n_particles):
        width_px = max(6.0, specimen.diameter_nm_mean * 10.0 / pixel_size_a)
        length_px = width_px * rng.uniform(2.2, 5.0)
        cy, cx = rng.uniform(0, v_proj.shape[0]), rng.uniform(0, v_proj.shape[1])
        angle = rng.uniform(0, np.pi)
        mask = _capsule_mask(v_proj.shape, cy, cx, length_px, width_px * 0.5, angle)
        wall = mask.astype(np.float32) - gaussian_filter(mask.astype(np.float32), sigma=max(1.0, width_px * 0.07))
        interior = gaussian_filter(mask.astype(np.float32), sigma=max(1.0, width_px * 0.12))
        texture = 0.15 * gaussian_filter(rng.standard_normal(v_proj.shape).astype(np.float32), sigma=2.0)
        v_proj += d_excess * specimen.ice_thickness_nm * (0.35 * interior + 0.8 * np.clip(wall, 0, 1) + texture * mask)
        label[mask] = np.maximum(label[mask], specimen.ice_thickness_nm * 10.0)


def _place_filaments(v_proj: np.ndarray, label: np.ndarray, pixel_size_a: float, specimen: Specimen, rng) -> None:
    """Thin, curved filaments (the natural target of ridge detection): a persistent
    random walk traces each centerline, dilated to a small constant width. Uses
    diameter_nm_mean as the (small) filament diameter."""
    from scipy.ndimage import binary_dilation

    h, w = v_proj.shape
    d_excess = specimen.bacteria_mip_v - specimen.ice_mip_v
    width_px = max(2.0, specimen.diameter_nm_mean * 10.0 / pixel_size_a)
    radius = max(1, int(round(width_px * 0.5)))
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (xx ** 2 + yy ** 2) <= radius ** 2
    for _ in range(specimen.n_particles):
        centerline = np.zeros((h, w), dtype=bool)
        cy, cx = rng.uniform(0.08 * h, 0.92 * h), rng.uniform(0.08 * w, 0.92 * w)
        ang = rng.uniform(0, 2 * np.pi)
        n_steps = int(rng.uniform(200, 500))
        for _ in range(n_steps):
            ang += rng.normal(0.0, 0.09)          # smooth curvature
            cy += np.sin(ang) * 1.5
            cx += np.cos(ang) * 1.5
            iy, ix = int(round(cy)), int(round(cx))
            if 0 <= iy < h and 0 <= ix < w:
                centerline[iy, ix] = True
            else:
                break
        if not centerline.any():
            continue
        tube = binary_dilation(centerline, structure=disk)
        soft = gaussian_filter(tube.astype(np.float32), sigma=1.0)
        v_proj += d_excess * specimen.ice_thickness_nm * 0.7 * soft
        label[tube] = np.maximum(label[tube], specimen.ice_thickness_nm * 10.0)


def _element_from_pdb_line(line: str) -> str:
    element = line[76:78].strip().upper()
    if element:
        return element
    name = line[12:16].strip().upper()
    letters = "".join(ch for ch in name if ch.isalpha())
    if not letters:
        return "C"
    if len(letters) >= 2 and letters[:2] in _ATOMIC_NUMBERS:
        return letters[:2]
    return letters[:1]


def _parse_pdb_atoms(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not path:
        return None
    coords: list[tuple[float, float, float]] = []
    weights: list[float] = []
    try:
        for line in Path(path).read_text(errors="ignore").splitlines():
            if line.startswith(("ATOM", "HETATM")):
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                    weights.append(float(_ATOMIC_NUMBERS.get(_element_from_pdb_line(line), 6)))
                except ValueError:
                    continue
    except OSError:
        return None
    if not coords:
        return None
    pts = np.asarray(coords, dtype=np.float32)
    pts -= pts.mean(axis=0, keepdims=True)
    w = np.asarray(weights, dtype=np.float32)
    w /= max(float(w.mean()), 1.0)
    return pts, w


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random 3D rotation from a unit quaternion."""
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array(
        [
            [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)],
        ],
        dtype=np.float32,
    )


def _cyclic_oligomer_atoms(points: np.ndarray, weights: np.ndarray, oligomer_count: int) -> tuple[np.ndarray, np.ndarray]:
    copies = []
    all_weights = []
    n = max(1, int(oligomer_count))
    for i in range(n):
        theta = 2 * np.pi * i / n
        ca, sa = np.cos(theta), np.sin(theta)
        rot = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float32)
        copies.append(points @ rot.T)
        all_weights.append(weights)
    return np.vstack(copies), np.concatenate(all_weights)


def _splat_projected_atoms(
    shape: tuple[int, int],
    points_a: np.ndarray,
    weights: np.ndarray,
    center_yx: tuple[float, float],
    pixel_size_a: float,
) -> np.ndarray:
    """Project oriented atoms along z into a weighted 2D projected-potential proxy."""
    h, w = shape
    cy, cx = center_yx
    y = points_a[:, 1] / pixel_size_a + cy
    x = points_a[:, 0] / pixel_size_a + cx
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    dx = x - x0
    dy = y - y0
    img = np.zeros(shape, dtype=np.float32)
    for ox, oy, coeff in ((0, 0, (1 - dx) * (1 - dy)), (1, 0, dx * (1 - dy)), (0, 1, (1 - dx) * dy), (1, 1, dx * dy)):
        xx = x0 + ox
        yy = y0 + oy
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        if np.any(valid):
            np.add.at(img, (yy[valid], xx[valid]), weights[valid] * coeff[valid])
    return img


def _splat_atom_slice(
    volume: np.ndarray,
    points_a: np.ndarray,
    weights: np.ndarray,
    pixel_size_a: float,
    slice_thickness_a: float,
    center_yx: tuple[float, float] | None = None,
) -> None:
    nz, h, w = volume.shape
    cy, cx = center_yx if center_yx is not None else (h * 0.5, w * 0.5)
    z = points_a[:, 2] / slice_thickness_a + nz * 0.5
    y = points_a[:, 1] / pixel_size_a + cy
    x = points_a[:, 0] / pixel_size_a + cx
    zi = np.rint(z).astype(np.int32)
    yi = np.rint(y).astype(np.int32)
    xi = np.rint(x).astype(np.int32)
    valid = (zi >= 0) & (zi < nz) & (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
    if np.any(valid):
        np.add.at(volume, (zi[valid], yi[valid], xi[valid]), weights[valid])


def _protein_volume(shape: tuple[int, int], pixel_size_a: float, specimen: Specimen, rng) -> tuple[np.ndarray, np.ndarray] | None:
    pdb_atoms = _parse_pdb_atoms(specimen.pdb_path)
    if pdb_atoms is None:
        return None
    points, weights = pdb_atoms
    points, weights = _cyclic_oligomer_atoms(points, weights, specimen.oligomer_count)

    span_a = np.ptp(points, axis=0)
    thickness_a = max(float(span_a[2]) + 24.0, specimen.diameter_nm_mean * 10.0)
    slice_thickness_a = max(1.0, specimen.membrane_thickness_nm)
    nz = int(np.clip(np.ceil(thickness_a / slice_thickness_a), 4, 192))
    volume = np.zeros((nz, shape[0], shape[1]), dtype=np.float32)
    radius_px = max(4.0, 0.5 * max(float(span_a[0]), float(span_a[1]), specimen.diameter_nm_mean * 10.0) / pixel_size_a)
    placed: list[tuple[float, float, float]] = []
    tries = 0
    n_particles = max(1, int(specimen.n_particles))
    while len(placed) < n_particles and tries < n_particles * 60:
        tries += 1
        cy, cx = rng.uniform(0, shape[0]), rng.uniform(0, shape[1])
        if not specimen.allow_overlap and any(np.hypot(cy - py, cx - px) < (radius_px + pr) for py, px, pr in placed):
            continue
        oriented = points @ _random_rotation_matrix(rng).T
        _splat_atom_slice(volume, oriented, weights, pixel_size_a, slice_thickness_a, (cy, cx))
        placed.append((cy, cx, radius_px))
    sigma_xy = max(0.45, 1.1 / pixel_size_a)
    sigma_z = max(0.35, 1.1 / slice_thickness_a)
    volume = gaussian_filter(volume, sigma=(sigma_z, sigma_xy, sigma_xy))
    if volume.max() > 0:
        volume /= volume.max()
    label = volume.max(axis=0)
    scale = (specimen.protein_mip_v - specimen.ice_mip_v) * slice_thickness_a * 2.0
    return volume * scale, label


def _projection_volume(shape: tuple[int, int], pixel_size_a: float, specimen: Specimen, rng) -> tuple[np.ndarray, np.ndarray]:
    projected = np.zeros(shape, dtype=np.float32)
    label = np.zeros(shape, dtype=np.float32)
    kind = specimen.kind.lower()
    if kind in {"plga", "solid", "solid_particles"}:
        _place_solid_particles(projected, label, pixel_size_a, specimen, rng)
    elif kind in {"lipid_single", "single_lipid", "vesicle"}:
        _place_lipid_vesicles(projected, label, pixel_size_a, specimen, rng, layers=1)
    elif kind in {"lipid_multi", "multilamellar", "multi_lipid"}:
        _place_lipid_vesicles(projected, label, pixel_size_a, specimen, rng, layers=3)
    elif kind in {"protein", "pdb"}:
        _place_proteins(projected, label, pixel_size_a, specimen, rng)
    elif kind in {"bacteria", "cell"}:
        _place_bacteria(projected, label, pixel_size_a, specimen, rng)
    else:
        raise ValueError(f"unknown TEM specimen kind: {specimen.kind!r}")
    nz = 8
    window = np.hanning(nz + 2)[1:-1].astype(np.float32)
    window /= window.sum() + 1e-8
    return projected[None, :, :] * window[:, None, None], label


def _fresnel_propagator(shape: tuple[int, int], pixel_size_a: float, voltage_kv: float, dz_a: float) -> np.ndarray:
    lam_a = wavelength_pm(voltage_kv) * 1e-2
    ky = np.fft.fftfreq(shape[0], d=pixel_size_a)
    kx = np.fft.fftfreq(shape[1], d=pixel_size_a)
    kxx, kyy = np.meshgrid(kx, ky)
    k2 = kxx**2 + kyy**2
    return np.exp(-1j * np.pi * lam_a * dz_a * k2).astype(np.complex64)


def _objective_phase(shape: tuple[int, int], pixel_size_a: float, cfg: dict, defocus_um: float, bfactor: float) -> np.ndarray:
    lam_a = cfg["_derived"]["wavelength_pm"] * 1e-2
    cs_a = cfg["cs_mm"] * 1e7
    df_a = -defocus_um * 1e4
    k = _freq_grid(shape, pixel_size_a)
    k2 = k**2
    chi = np.pi * lam_a * df_a * k2 - 0.5 * np.pi * cs_a * lam_a**3 * k2**2
    envelope = np.exp(-bfactor * k2 / 4.0)
    return (np.exp(-1j * chi) * envelope).astype(np.complex64)


def multislice_exit_wave(
    volume: np.ndarray,
    cfg: dict,
    *,
    slice_thickness_a: float,
) -> np.ndarray:
    shape = volume.shape[1:]
    sigma = sigma_rad_per_va(float(cfg["voltage_kv"]))
    prop = _fresnel_propagator(shape, cfg["pixel_size_a"], float(cfg["voltage_kv"]), slice_thickness_a)
    wave = np.ones(shape, dtype=np.complex64)
    for slc in volume:
        wave *= np.exp(1j * sigma * slc).astype(np.complex64)
        wave = np.fft.ifft2(np.fft.fft2(wave) * prop).astype(np.complex64)
    return wave


def _place_proteins(v_proj: np.ndarray, label: np.ndarray, pixel_size_a: float, specimen: Specimen, rng) -> None:
    d_excess = specimen.protein_mip_v - specimen.ice_mip_v
    pdb_atoms = _parse_pdb_atoms(specimen.pdb_path)
    atom_sigma_px = max(0.45, 1.4 / pixel_size_a)
    for _ in range(specimen.n_particles):
        cy, cx = rng.uniform(0, v_proj.shape[0]), rng.uniform(0, v_proj.shape[1])
        if pdb_atoms is None:
            d_nm = max(2.0, rng.normal(specimen.diameter_nm_mean, specimen.diameter_nm_sd))
            r_px = (d_nm * 10.0 / 2.0) / pixel_size_a
            res = _sphere_chord(v_proj.shape, cy, cx, r_px)
            if res is None:
                continue
            (y0, y1, x0, x1), chord_px = res
            chord_a = chord_px * pixel_size_a
            v_proj[y0:y1, x0:x1] += (d_excess * chord_a).astype(np.float32)
            label[y0:y1, x0:x1] = np.maximum(label[y0:y1, x0:x1], chord_a)
            continue
        points, weights = pdb_atoms
        points, weights = _cyclic_oligomer_atoms(points, weights, specimen.oligomer_count)
        points = points @ _random_rotation_matrix(rng).T
        atom_img = _splat_projected_atoms(v_proj.shape, points, weights, (cy, cx), pixel_size_a)
        atom_img = gaussian_filter(atom_img, sigma=atom_sigma_px)
        if atom_img.max() <= 0:
            continue
        # Z-weighted, atom-number-weighted projection proxy in V Angstrom units.
        v_proj += d_excess * atom_img * pixel_size_a * 3.0
        mask = atom_img > atom_img.max() * 0.08
        label[mask] = np.maximum(label[mask], atom_img[mask] / atom_img.max())


def make_potential(shape: tuple[int, int], pixel_size_a: float, specimen: Specimen) -> PotentialResult:
    rng = np.random.default_rng(specimen.seed)
    v_proj = np.zeros(shape, dtype=np.float32)
    label = np.zeros(shape, dtype=np.float32)
    kind = specimen.kind.lower()
    if kind in {"plga", "solid", "solid_particles"}:
        _place_solid_particles(v_proj, label, pixel_size_a, specimen, rng)
    elif kind in {"lipid_single", "single_lipid", "vesicle"}:
        _place_lipid_vesicles(v_proj, label, pixel_size_a, specimen, rng, layers=1)
    elif kind in {"lipid_multi", "multilamellar", "multi_lipid"}:
        _place_lipid_vesicles(v_proj, label, pixel_size_a, specimen, rng, layers=3)
    elif kind in {"protein", "pdb"}:
        _place_proteins(v_proj, label, pixel_size_a, specimen, rng)
    elif kind in {"bacteria", "cell"}:
        _place_bacteria(v_proj, label, pixel_size_a, specimen, rng)
    elif kind in {"filament", "fiber", "fibre", "actin"}:
        _place_filaments(v_proj, label, pixel_size_a, specimen, rng)
    else:
        raise ValueError(f"unknown TEM specimen kind: {specimen.kind!r}")

    if specimen.solvent_noise > 0:
        white = rng.standard_normal(shape).astype(np.float32)
        corr_px = max(0.5, specimen.solvent_corr_a / pixel_size_a)
        f_ = gaussian_filter(white, sigma=corr_px)
        f_ /= f_.std() + 1e-8
        amp = specimen.solvent_noise * np.sqrt(specimen.ice_thickness_nm) * pixel_size_a
        v_proj = v_proj + amp * f_

    v_proj -= v_proj.mean()
    return PotentialResult(v_proj=v_proj, label=label, specimen=specimen)


def make_volume(shape: tuple[int, int], pixel_size_a: float, specimen: Specimen) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(specimen.seed)
    if specimen.kind.lower() in {"pdb", "protein"} and specimen.pdb_path:
        vol = _protein_volume(shape, pixel_size_a, specimen, rng)
        if vol is not None:
            return vol
    return _projection_volume(shape, pixel_size_a, specimen, rng)


def _freq_grid(shape: tuple[int, int], pixel_size_a: float) -> np.ndarray:
    ky = np.fft.fftfreq(shape[0], d=pixel_size_a)
    kx = np.fft.fftfreq(shape[1], d=pixel_size_a)
    kxx, kyy = np.meshgrid(kx, ky)
    return np.sqrt(kxx**2 + kyy**2).astype(np.float32)


def ctf(shape: tuple[int, int], pixel_size_a: float, cfg: dict, defocus_um: float, bfactor: float = 40.0) -> np.ndarray:
    lam = cfg["_derived"]["wavelength_pm"] * 1e-2
    cs_a = cfg["cs_mm"] * 1e7
    q = cfg["amplitude_contrast"]
    df_a = -defocus_um * 1e4

    k = _freq_grid(shape, pixel_size_a)
    k2 = k**2
    chi = np.pi * lam * df_a * k2 - 0.5 * np.pi * cs_a * lam**3 * k2**2
    ctf_ = -(np.sqrt(max(0.0, 1 - q**2)) * np.sin(chi) + q * np.cos(chi))

    cc_a = cfg["cc_mm"] * 1e7
    e_acc = float(cfg["voltage_kv"]) * 1e3
    d_spread = cc_a * (cfg["energy_spread_ev"] / e_acc)
    e_t = np.exp(-0.5 * (np.pi * lam * d_spread * k2) ** 2)

    alpha = cfg["convergence_mrad"] * 1e-3
    grad = cs_a * lam**3 * k2 * k - df_a * lam * k
    e_s = np.exp(-(np.pi * alpha / lam) ** 2 * grad**2)
    e_b = np.exp(-bfactor * k2 / 4.0)
    return (ctf_ * e_t * e_s * e_b).astype(np.float32)


def apply_ctf(sigma_vproj: np.ndarray, ctf_2d: np.ndarray) -> np.ndarray:
    contrast = np.fft.ifft2(np.fft.fft2(sigma_vproj) * ctf_2d).real
    return (1.0 + 2.0 * contrast).astype(np.float32)


def expose(i_ideal: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    dose = cfg["total_dose_e_per_a2"]
    px_area = cfg["pixel_size_a"] ** 2
    lam = np.clip(i_ideal, 0, None) * dose * px_area
    counts = rng.poisson(lam).astype(np.float32)
    sigma = _MTF_SIGMA.get(cfg["detector_model"], 0.6)
    if sigma > 0:
        counts = gaussian_filter(counts, sigma=sigma)
    return counts


def to_display(counts: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(counts, [0.5, 99.5])
    return np.clip((counts - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def _pick_defocus(cfg: dict, rng: np.random.Generator, defocus_um: float | None) -> float:
    if defocus_um is not None:
        return float(defocus_um)
    lo, hi = cfg["defocus_min_um"], cfg["defocus_max_um"]
    return float(rng.uniform(min(lo, hi), max(lo, hi)))


def simulate_micrograph(
    cfg: dict,
    specimen: Specimen | None = None,
    defocus_um: float | None = None,
    bfactor: float = 40.0,
    seed: int = 0,
) -> Micrograph:
    rng = np.random.default_rng(seed)
    n = int(cfg["image_size_px"])
    shape = (n, n)
    px = cfg["pixel_size_a"]
    specimen = specimen or Specimen(seed=seed)
    df = _pick_defocus(cfg, rng, defocus_um)
    if cfg.get("phase_plate"):
        df = df if abs(df) > 0.05 else -0.05
    if str(cfg.get("simulation_path", "fast")).lower() in {"multislice", "physics", "advanced"}:
        volume, label = make_volume(shape, px, specimen)
        slice_thickness_a = max(1.0, float(cfg.get("slice_thickness_a", 5.0)))
        wave = multislice_exit_wave(volume, cfg, slice_thickness_a=slice_thickness_a)
        objective = _objective_phase(shape, px, cfg, df, bfactor)
        exit_image = np.abs(np.fft.ifft2(np.fft.fft2(wave) * objective)) ** 2
        ideal = exit_image.astype(np.float32)
        potential = volume.sum(axis=0)
    else:
        pot = make_potential(shape, px, specimen)
        sigma = sigma_rad_per_va(float(cfg["voltage_kv"]))
        ideal = apply_ctf(sigma * pot.v_proj, ctf(shape, px, cfg, df, bfactor=bfactor))
        potential = pot.v_proj
        label = pot.label
    counts = expose(ideal, cfg, rng)
    return Micrograph(
        image=to_display(counts),
        counts=counts,
        ideal=ideal,
        potential=potential,
        label=label,
        defocus_um=df,
        config=cfg,
    )
