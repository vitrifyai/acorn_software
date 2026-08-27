"""Specimen charging: computed tendency, modelled appearance.

An uncoated biological specimen does not conduct. Electrons delivered by the
beam accumulate, the accumulated charge pushes back on the beam that follows,
and the image acquires the artefacts anyone who has imaged uncoated organics
will recognise: bright flaring on isolated raised features, streaks trailing
along the scan direction, and drift that shifts the picture as it is written.

This module is deliberately split into two halves that make different claims,
because conflating them would let a drawing pass for a derivation.

    COMPUTED      Whether a material charges, in which direction, and how
                  strongly, comes from the transport model. A surface is in
                  balance when it emits as many electrons as it receives, which
                  is delta + eta = 1. Below that it accumulates negative charge;
                  above it, positive. The crossover energy is found by tracing
                  electrons, not by looking it up -- and for biology it lands
                  near 0.7 keV, which is why uncoated organics get imaged around
                  1 kV and not at 5.

    MODELLED      What that accumulation does to the picture is drawn, not
                  derived. Solving it properly means iterating a surface
                  potential against the beam it deflects, which is a different
                  and much larger piece of physics. The artefacts below are
                  shaped to resemble the real ones; they are not predictions of
                  them, and nothing here should be cited as though it were.

Charging is off unless asked for. An artefact model that arrives by default is
one that ends up in a figure without anyone deciding it should.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import Material
from .transport import trace

# Materials that carry charge away faster than the beam delivers it. A metal
# does not charge however low its yield, so no artefact is applied to one.
CONDUCTORS = frozenset({
    "aluminium", "titanium", "iron", "copper", "silver", "gold", "platinum",
    "gallium", "carbon",       # amorphous carbon coats exist precisely for this
    "silicon",                 # a doped Si stub is THE standard grounded mount
})
# Deliberately absent, because they genuinely charge: silica, alumina, titania,
# ice, resin, biology. Note that silicon is here but silica is not -- the wafer
# grounds, its oxide does not, and mixing the two up would exempt every
# insulating specimen mounted on a stub from an artefact it really suffers.

# Energies at which the crossover is searched. Log-spaced because the useful
# range spans a decade and the curve turns sharply at the bottom of it.
_SCAN_KV = np.geomspace(0.15, 8.0, 22)


@dataclass(frozen=True)
class ChargeState:
    """How a material behaves at one beam energy. Computed, not drawn."""

    material: str
    kv: float
    total_yield: float          # delta + eta
    conducts: bool

    @property
    def balance(self) -> float:
        """Distance from equilibrium. Negative means the surface charges negative."""
        return self.total_yield - 1.0

    @property
    def charges(self) -> bool:
        return not self.conducts and abs(self.balance) > 0.05

    @property
    def sign(self) -> str:
        if not self.charges:
            return "none"
        return "negative" if self.balance < 0 else "positive"

    def describe(self) -> str:
        if self.conducts:
            return f"{self.material} conducts — no charging at any energy."
        if not self.charges:
            return (f"{self.material} at {self.kv:g} kV is close to balance "
                    f"(yield {self.total_yield:.2f}) — little charging.")
        return (f"{self.material} at {self.kv:g} kV charges {self.sign} "
                f"(yield {self.total_yield:.2f}, balance is 1.00).")


def charge_state(material: Material, kv: float, n_electrons: int = 8000,
                 seed: int = 0) -> ChargeState:
    """Total yield at `kv`, from the transport model."""
    if material.name in CONDUCTORS or material.rho <= 0:
        return ChargeState(material.name, float(kv), float("nan"), True)
    r = trace(material, float(kv), n_electrons=n_electrons, seed=seed)
    return ChargeState(material.name, float(kv), float(r.delta + r.eta), False)


def crossover_kv(material: Material, n_electrons: int = 8000,
                 seed: int = 0) -> float | None:
    """The upper energy at which total yield falls back through 1, or None.

    Above this a non-conducting surface charges negative; below it, positive.
    The curve is not monotonic -- yield rises from a lower crossover, peaks, then
    falls -- so this scans rather than bisecting, which would need a bracket the
    caller cannot know.
    """
    if material.name in CONDUCTORS or material.rho <= 0:
        return None
    ys = [charge_state(material, kv, n_electrons, seed).total_yield
          for kv in _SCAN_KV]
    above = [i for i, y in enumerate(ys) if y >= 1.0]
    if not above:
        return None
    i = above[-1]
    if i == len(_SCAN_KV) - 1:
        return float(_SCAN_KV[-1])
    x0, x1, y0, y1 = _SCAN_KV[i], _SCAN_KV[i + 1], ys[i], ys[i + 1]
    return float(x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0))


def _accumulation(insulating: np.ndarray, height_nm: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Where charge collects, as a 0..1 map.

    What concentrates charge is a poor path to ground, and that has two very
    different geometries depending on whether anything in the field conducts.

    With a conductor present, distance from it is the right driver: charge drains
    to the nearest grounded neighbour.

    With nothing conducting -- spores on a membrane, the case that charges worst
    and the one being simulated here -- distance-to-ground has no meaning inside
    the frame. Using it anyway measures distance from the image border, which is
    a property of the simulation window rather than the specimen, and produces a
    centre-bright dome instead of the discrete flares a real image shows. So the
    driver becomes relief plus isolation: raised material charges, and a feature
    with few neighbours has less to drain through, so it flares hardest. In a
    crowded field the isolation term falls away almost everywhere -- correctly,
    since nothing there is isolated -- and relief carries the result.
    """
    from scipy.ndimage import (
        center_of_mass,
        distance_transform_edt,
        gaussian_filter,
        label as cc_label,
    )

    if not insulating.any():
        return np.zeros(insulating.shape, np.float32)

    relief = height_nm.astype(np.float32)
    span = float(relief.max() - relief.min())
    relief = (relief - relief.min()) / span if span > 0 else np.zeros_like(relief)

    has_conductor = bool((~insulating).any())
    if has_conductor:
        drain = distance_transform_edt(insulating).astype(np.float32)
        if drain.max() > 0:
            drain /= drain.max()
    else:
        # Isolation, measured between features rather than around them. The
        # obvious version -- smooth the raised mask and call the result local
        # crowding -- is wrong, because a feature contributes to its own
        # neighbourhood and so suppresses its own score. A lone object then
        # comes out as the LEAST isolated thing in the frame, which is exactly
        # backwards. Distance to the nearest OTHER feature has no such flaw.
        raised = relief > 0.35
        if not raised.any():
            # Nothing clears the threshold -- a low-contrast or nearly flat
            # scene. Fall back to relief rather than leaving zeros, which would
            # silently disable the artefact while still reporting it applied.
            raised = insulating.copy()
            drain = relief.copy()
        else:
            lab, n = cc_label(raised)
            drain = np.zeros_like(relief)
            if n:
                centres = np.array(center_of_mass(raised, lab, range(1, n + 1)))
                if n == 1:
                    isolation = np.array([1.0])
                else:
                    d = np.linalg.norm(
                        centres[:, None, :] - centres[None, :, :], axis=-1)
                    np.fill_diagonal(d, np.inf)
                    nearest = d.min(axis=1)
                    hi = float(nearest.max())
                    isolation = nearest / hi if hi > 0 else np.ones(n)
                # Some spores flare and their neighbours do not; isolation sets
                # the tendency, this sets which ones actually go.
                jitter = 0.55 + 0.45 * rng.random(n).astype(np.float32)
                weights = np.concatenate(
                    [[0.0], (isolation * jitter).astype(np.float32)])
                drain = weights[lab]

    # A uniform floor over everything insulating. A featureless insulating
    # surface still charges -- it just charges evenly, shifting the whole
    # picture rather than flaring parts of it. Without this the model reports
    # charging as applied while leaving a flat scene untouched, which is a
    # metadata claim that nothing backs up. Kept small deliberately: a large
    # floor raises an intensity threshold along with the signal, which cancels
    # the differential flare that is the visible part of the artefact.
    acc = np.where(insulating,
                   0.08 + 0.57 * drain + 0.35 * relief * drain.max(),
                   0.0).astype(np.float32)
    # Real charge patches are irregular rather than following the geometry
    # exactly; a little correlated noise keeps them from looking stencilled.
    acc *= (1.0 + 0.25 * gaussian_filter(
        rng.standard_normal(acc.shape).astype(np.float32), 8.0))
    return np.clip(acc, 0.0, None)


def apply_charging(signal: np.ndarray, insulating: np.ndarray,
                   height_nm: np.ndarray, balance: float, strength: float = 1.0,
                   scan_axis: int = 1, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Render charging artefacts onto a signal. MODELLED, not derived.

    `balance` is the computed distance from equilibrium and sets the direction
    and scale; everything after that is a drawing. Three artefacts, each chosen
    because it is what an uncoated specimen actually looks like:

        flaring   isolated raised features go bright and bloom
        streaking charge trails along the fast scan direction, because a line is
                  written left to right and what charged early affects what is
                  written after it -- never before, so the filter is causal
        drift     the picture shifts progressively down the slow axis

    Returns the modified signal and a record of what was applied.
    """
    from scipy.ndimage import gaussian_filter, gaussian_filter1d

    out = signal.astype(np.float32, copy=True)
    if strength <= 0 or not insulating.any() or abs(balance) < 0.05:
        return out, {"charging": "none"}

    rng = np.random.default_rng(seed)
    severity = float(strength) * min(abs(float(balance)), 1.0)
    acc = _accumulation(insulating, height_nm, rng)
    if acc.max() > 0:
        acc = acc / acc.max()

    # Flaring. Negative charging drives the local yield back up toward balance,
    # which reads as brightening; positive charging retards escaping secondaries
    # and reads as darkening.
    direction = 1.0 if balance < 0 else -1.0
    flare = gaussian_filter(acc, 2.0)
    out *= (1.0 + direction * 1.6 * severity * flare)

    # Streaking along the fast scan axis. Genuinely one-sided: charge laid down
    # at a point affects what the beam writes AFTER it and cannot reach back to
    # what was already written. A symmetric blur would be wrong in a way that is
    # easy to miss, because it still looks like smearing -- so the kernel is
    # built explicitly one-sided rather than being a Gaussian with a nudge.
    tail = max(2.0, 26.0 * severity)
    n = max(3, int(tail * 4))
    k = np.exp(-np.arange(n, dtype=np.float32) / tail)
    k /= k.sum()
    pad = [(0, 0), (0, 0)]
    pad[scan_axis] = (n - 1, 0)
    padded = np.pad(acc, pad, mode="edge")
    if scan_axis == 1:
        trail = np.apply_along_axis(
            lambda r: np.convolve(r, k, mode="valid"), 1, padded)
    else:
        trail = np.apply_along_axis(
            lambda c: np.convolve(c, k, mode="valid"), 0, padded)
    out *= (1.0 + direction * 0.9 * severity * trail.astype(np.float32))

    # Drift: the surface potential builds through the frame, so the deflection
    # it causes grows as the scan proceeds.
    rows = out.shape[1 - scan_axis]
    max_shift = severity * 0.02 * out.shape[scan_axis]
    shifts = (np.linspace(0.0, 1.0, rows) ** 1.5) * max_shift
    shifted = np.empty_like(out)
    for i in range(rows):
        line = out[i] if scan_axis == 1 else out[:, i]
        s = int(round(shifts[i]))
        moved = np.roll(line, s)
        if s > 0:
            moved[:s] = line[0]
        if scan_axis == 1:
            shifted[i] = moved
        else:
            shifted[:, i] = moved
    out = shifted

    return np.clip(out, 0.0, None), {
        "charging": "applied",
        "sign": "negative" if balance < 0 else "positive",
        "balance": float(balance),
        "severity": float(severity),
        "max_drift_px": float(max_shift),
        "model": "phenomenological — tendency computed, appearance drawn",
    }
