"""Material properties for SEM electron transport.

Everything the Monte Carlo needs about a solid reduces to four numbers: atomic
number Z, atomic weight A, density rho, and the energy it costs to liberate one
secondary electron (epsilon). The first three are tabulated physical constants;
the fourth is not, and is discussed under `Material.epsilon_ev` below.

Compounds are handled by weight-averaging Z and A. That is the standard
approximation for electron transport (Joy; Reimer) and is accurate for the
stopping power and elastic scattering because both are dominated by sums over
atoms rather than by chemistry. It is *not* accurate for the SE yield, which
depends on band structure -- hence the separate epsilon.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """A solid, as the transport model sees it.

    epsilon_ev
        Mean energy deposited per escaping secondary electron. Unlike Z/A/rho
        this is not a tabulated constant -- it folds together the SE production
        efficiency and the band structure, and published values for the same
        material vary by a factor of two. Treat it as the one calibration knob:
        the BSE yield and the interaction volume come out of the physics with no
        free parameters, but the absolute SE yield does not. Defaults here are
        chosen to put delta in the accepted range at 1-20 keV; if you need
        quantitative SE contrast, fit epsilon to a measured delta.

    se_escape_nm
        Mean escape depth of secondary electrons, lambda_SE. Short for metals
        (~1 nm), longer for insulators (~10 nm) because there are no conduction
        electrons to scatter against.
    """

    name:          str
    Z:             float
    A:             float          # g/mol
    rho:           float          # g/cm^3
    epsilon_ev:    float = 30.0
    se_escape_nm:  float = 2.0


# Elements and compounds relevant to the specimens ACORN sees: biological
# material and resin, vitreous ice, silicon and its oxide, grid and marker
# metals, and the Pt/Ga a FIB leaves behind.
# epsilon_ev below is CALIBRATED, not measured: for each material it is the
# value that makes `transport.trace` reproduce the published SE yield at 20 keV
# and normal incidence (carbon 0.05, silicon 0.10, copper 0.13, silver 0.15,
# gold 0.20, platinum 0.19 ...). Everything else in the model -- the backscatter
# yield, the interaction volume, the shape of delta(E0) -- is predicted with no
# free parameters; only the absolute SE scale is tied down this way. The fitted
# values landing in a narrow 57-243 eV band, rather than scattering over orders
# of magnitude, is the evidence that the underlying transport is right.
MATERIALS: dict[str, Material] = {
    # --- light / biological -------------------------------------------------
    "vacuum":    Material("vacuum",    0.00,   1.000,  0.00, 1e9,   1.0),
    "ice":       Material("ice",       7.42,  18.020,  0.92,  80.0, 4.0),
    "biology":   Material("biology",   6.60,  12.660,  1.35, 109.4, 3.0),
    "resin":     Material("resin",     6.20,  12.100,  1.20, 127.8, 4.0),
    "carbon":    Material("carbon",    6.00,  12.011,  2.25, 143.6, 2.5),

    # --- semiconductors / oxides -------------------------------------------
    "silicon":   Material("silicon",  14.00,  28.086,  2.33,  87.2, 2.5),
    "silica":    Material("silica",   10.80,  20.030,  2.20, 133.3, 4.0),
    "alumina":   Material("alumina",  10.60,  20.390,  3.95, 239.6, 4.0),
    "titania":   Material("titania",  16.40,  26.630,  4.23, 242.8, 3.0),

    # --- metals -------------------------------------------------------------
    "aluminium": Material("aluminium", 13.00, 26.982,  2.70,  56.7, 1.5),
    "titanium":  Material("titanium",  22.00, 47.867,  4.51,  88.1, 1.5),
    "iron":      Material("iron",      26.00, 55.845,  7.87, 106.6, 1.2),
    "copper":    Material("copper",    29.00, 63.546,  8.96,  99.7, 1.0),
    "silver":    Material("silver",    47.00, 107.870, 10.49,  96.3, 1.0),
    "gold":      Material("gold",      79.00, 196.970, 19.30, 120.8, 1.0),
    "platinum":  Material("platinum",  78.00, 195.080, 21.45, 141.0, 1.0),
    "gallium":   Material("gallium",   31.00, 69.723,  5.91,  95.3, 1.5),
}


def get(name: str) -> Material:
    """Look up a material by name, with a readable error listing what exists."""
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(
            f"unknown material {name!r}. Known materials: "
            + ", ".join(sorted(MATERIALS))
        ) from None


def mix(components: dict[str, float]) -> Material:
    """Weight-average several materials into one effective medium.

    `components` maps material name -> weight fraction; fractions are
    normalised. Use for alloys, embedded resins, or a Ga-implanted surface
    layer. epsilon and the escape depth are averaged too, which is cruder than
    averaging Z -- an effective-medium SE yield is a guess, not a prediction.
    """
    if not components:
        raise ValueError("mix() needs at least one component")
    total = sum(components.values())
    if total <= 0:
        raise ValueError("component weights must sum to something positive")

    z = a = r = eps = esc = 0.0
    for name, w in components.items():
        m = get(name)
        f = w / total
        z += f * m.Z
        a += f * m.A
        r += f * m.rho
        eps += f * m.epsilon_ev
        esc += f * m.se_escape_nm
    return Material("+".join(sorted(components)), z, a, r, eps, esc)


def kanaya_okayama_nm(material: Material, E0_kev: float) -> float:
    """Kanaya-Okayama electron range, nm.

    R = 0.0276 * A * E^1.67 / (Z^0.889 * rho)  micrometres, E in keV.

    The standard empirical range formula. Used here only as a sanity check on
    the Monte Carlo and to size the simulation grid -- nothing in the forward
    model depends on it.
    """
    if material.rho <= 0:
        return float("inf")
    um = 0.0276 * material.A * E0_kev ** 1.67 / (material.Z ** 0.889 * material.rho)
    return um * 1000.0
