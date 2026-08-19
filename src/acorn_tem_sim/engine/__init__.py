"""cryotem: physics-based cryo-TEM image simulator (setup phase + forward model)."""
from .setup import (
    run_setup, resolve, probe_metadata, wavelength_pm, sigma_rad_per_VA,
    FIELDS, PRESETS,
)
from .potential import Specimen, make_potential
from .simulate import simulate_micrograph, Micrograph
from .multislice import simulate_multislice, MultisliceResult
from .bacteria import Bacterium, simulate_bacterium, SPECIES
from .fib_lamella import simulate_fib_lamella, mill_slabs
from .stem import stem_4d, virtual_detectors, make_probe, FourDSTEM
from .scene import Scene, Cell, Nanoparticles, Contamination, simulate_scene

__all__ = [
    "run_setup", "resolve", "probe_metadata",
    "wavelength_pm", "sigma_rad_per_VA", "FIELDS", "PRESETS",
    "Specimen", "make_potential", "simulate_micrograph", "Micrograph",
    "simulate_multislice", "MultisliceResult",
    "Bacterium", "simulate_bacterium", "SPECIES",
    "simulate_fib_lamella", "mill_slabs",
    "stem_4d", "virtual_detectors", "make_probe", "FourDSTEM",
    "Scene", "Cell", "Nanoparticles", "Contamination", "simulate_scene",
]
