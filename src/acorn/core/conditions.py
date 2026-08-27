"""What is currently altering a result, and what it can alter.

Every defect worth finding in this codebase has had the same shape: something is
in force, the output looks plausible, and nothing says so. Binning changed the
physical size CryoBLOB searched for. Denoising pulls object boundaries inward
before they are measured. A manual pixel size overrode a calibrated header. In
each case the numbers stayed reasonable, which is exactly why nobody noticed.

So this module answers one question -- *what is in force right now?* -- from a
single declaration, and that declaration drives two things at once:

    the UI          a persistent line naming every active condition, so a
                    setting cannot be quietly on
    the tests       `CONDITIONS` is enumerable, so a test can assert that every
                    condition claiming to affect measurement has an invariant
                    proving a known object still measures correctly under it

The second is the part that keeps this honest. Adding a setting means adding it
here, and the test suite then demands evidence that it does not corrupt a
measurement. A setting nobody declared is the failure mode; a setting declared
without evidence fails the build.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MeasurementClaim(str, Enum):
    """What a condition claims about numbers in physical units.

    The distinction that matters is between ALTERS and INVARIANT. Both are
    claims requiring evidence, and the second is the easier one to get wrong:
    binning asserts that a diameter in nanometres is unchanged because the pixel
    size was rescaled alongside the array, and if that rescale were ever dropped
    every measurement would be wrong by the bin factor while still looking
    reasonable. A claim of invariance is exactly the kind that has to be proven.
    """

    ALTERS = "alters"          # changes a measured number, by design
    INVARIANT = "invariant"    # changes the pixel grid but NOT the measurement
    UNRELATED = "unrelated"    # changes what is found or shown, not the units


class Affects(str, Enum):
    """What a condition can change about a result.

    MEASUREMENT is the serious one: a number in physical units that ends up in a
    figure or a paper. DETECTION changes what is found without changing the units
    of what is reported. APPEARANCE changes only the picture on screen.
    """

    MEASUREMENT = "measurement"
    DETECTION = "detection"
    APPEARANCE = "appearance"


@dataclass(frozen=True)
class Condition:
    """A setting that can change a result when it is not at its default."""

    key: str
    label: str
    affects: tuple[Affects, ...]
    note: str
    claim: MeasurementClaim = MeasurementClaim.UNRELATED

    @property
    def alters_numbers(self) -> bool:
        return Affects.MEASUREMENT in self.affects

    @property
    def needs_evidence(self) -> bool:
        """True when the claim about measurements has to be demonstrated."""
        return self.claim in (MeasurementClaim.ALTERS, MeasurementClaim.INVARIANT)


@dataclass(frozen=True)
class ActiveCondition:
    """A condition that is currently in force, with its value."""

    condition: Condition
    detail: str

    @property
    def alters_numbers(self) -> bool:
        return self.condition.alters_numbers


# The single declaration. Anything that can change a result belongs here --
# including things that only change appearance, because "only appearance" is a
# claim worth stating explicitly rather than leaving to be assumed.
CONDITIONS: tuple[Condition, ...] = (
    Condition(
        key="bin_factor",
        label="Analysis binning",
        affects=(Affects.DETECTION, Affects.APPEARANCE),
        note=("Detection and measurement use the binned pixels. Sizes stay in "
              "real units because the pixel size is rescaled with them, but "
              "any parameter expressed in PIXELS -- CryoBLOB's blob size, for "
              "instance -- now means a different physical size."),
        claim=MeasurementClaim.INVARIANT,
    ),
    Condition(
        key="pixel_size_override",
        label="Manual pixel size",
        affects=(Affects.MEASUREMENT,),
        note=("Every distance, area and diameter is computed from this rather "
              "than from the file's own calibration."),
        claim=MeasurementClaim.ALTERS,
    ),
    Condition(
        key="denoise",
        label="Denoising",
        affects=(Affects.MEASUREMENT, Affects.DETECTION, Affects.APPEARANCE),
        note=("Denoising pulls object boundaries inward, so sizes measured from "
              "a denoised image are biased low by an amount that depends on the "
              "method and its strength."),
        claim=MeasurementClaim.ALTERS,
    ),
    Condition(
        key="crop_region",
        label="Crop region",
        affects=(Affects.DETECTION,),
        note="Detection sees only the cropped area; objects outside it are absent.",
        claim=MeasurementClaim.UNRELATED,
    ),
    Condition(
        key="exclude_zones",
        label="Exclusion zones",
        affects=(Affects.DETECTION,),
        note="Detections inside these zones are discarded.",
        claim=MeasurementClaim.UNRELATED,
    ),
    Condition(
        key="charging",
        label="Simulated charging",
        affects=(Affects.MEASUREMENT, Affects.DETECTION, Affects.APPEARANCE),
        note=("A simulated artefact, not a property of the specimen, and it "
              "moves numbers by a lot. Measured on a 29-spore field: severe "
              "charging grew mean object area 2.2x and changed the object "
              "count from 40 to 28. Note the direction -- the count moved "
              "TOWARD the truth, because flaring fills the dark ridged "
              "interiors that otherwise fragment one spore into several. So "
              "charging can flatter a segmentation as easily as it spoils "
              "one, and a method tuned on charged simulations may be tuned to "
              "the artefact. Which way it goes depends on the method."),
        claim=MeasurementClaim.ALTERS,
    ),
    Condition(
        key="contrast",
        label="Contrast method",
        affects=(Affects.APPEARANCE,),
        note=("Display only. Detectors receive the stored pixels, not the "
              "contrast-adjusted view."),
        claim=MeasurementClaim.UNRELATED,
    ),
)

_BY_KEY = {c.key: c for c in CONDITIONS}


def get(key: str) -> Condition:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown condition {key!r}. Declared: "
                       + ", ".join(sorted(_BY_KEY))) from None


def evaluate(*, bin_factor: int = 1,
             pixel_size_override: float | None = None,
             header_pixel_size: float | None = None,
             denoise_method: str = "none",
             denoise_strength: float = 0.0,
             crop_region=None,
             exclude_zones=None,
             contrast_method: str = "percentile",
             default_contrast: str = "percentile",
             charging: float = 0.0) -> list[ActiveCondition]:
    """Everything currently in force, most consequential first.

    Takes plain values rather than reaching into the window, so it is testable
    without a running application and cannot be broken by a UI refactor.

    Two arguments exist to keep this from crying wolf. `header_pixel_size` is
    what the file itself says, so a calibration restored from a sidecar that
    agrees with the header is not reported as a manual override -- it is simply
    the file's own number. `default_contrast` is the default for this file type,
    because electron-microscopy formats default to bandpass and flagging that as
    "in force" would light the indicator on every image ever opened. An
    indicator that is always lit is one nobody reads.
    """
    active: list[ActiveCondition] = []

    if int(bin_factor) > 1:
        active.append(ActiveCondition(
            get("bin_factor"), f"{int(bin_factor)}x{int(bin_factor)}"))

    if pixel_size_override is not None and float(pixel_size_override) > 0:
        override = float(pixel_size_override)
        header = float(header_pixel_size) if header_pixel_size else None
        # Only a value that DISAGREES with the file is worth reporting. A
        # sidecar restoring the header's own calibration is not an override.
        differs = header is None or abs(override - header) > 1e-9 * max(header, 1.0)
        if differs:
            detail = f"{override:.4g} nm/px"
            if header:
                detail += f" (file says {header:.4g})"
            active.append(ActiveCondition(get("pixel_size_override"), detail))

    if denoise_method and denoise_method not in ("none", ""):
        active.append(ActiveCondition(
            get("denoise"), f"{denoise_method} (strength {denoise_strength:.2g})"))

    if crop_region is not None:
        active.append(ActiveCondition(get("crop_region"), "active"))

    if exclude_zones:
        n = len(exclude_zones) if hasattr(exclude_zones, "__len__") else 1
        active.append(ActiveCondition(
            get("exclude_zones"), f"{n} zone(s)"))

    if float(charging) > 0:
        active.append(ActiveCondition(
            get("charging"), f"strength {float(charging):.2g}"))

    if (contrast_method and contrast_method not in ("", "none")
            and contrast_method != default_contrast):
        active.append(ActiveCondition(get("contrast"), str(contrast_method)))

    # Measurement-altering conditions first: those are the ones that change a
    # number someone may publish.
    order = {Affects.MEASUREMENT: 0, Affects.DETECTION: 1, Affects.APPEARANCE: 2}
    active.sort(key=lambda a: min(order[x] for x in a.condition.affects))
    return active


def summarise(active: list[ActiveCondition]) -> str:
    """One line for the UI. Empty when nothing is in force."""
    if not active:
        return ""
    numeric = [a for a in active if a.alters_numbers]
    parts = ", ".join(f"{a.condition.label} ({a.detail})" for a in active)
    lead = "Affecting measurements" if numeric else "In force"
    return f"{lead}: {parts}"


def explain(active: list[ActiveCondition]) -> str:
    """The full account, for a tooltip or a report."""
    if not active:
        return "Nothing is altering results — measurements come from the stored pixels."
    return "\n\n".join(
        f"{a.condition.label} — {a.detail}\n{a.condition.note}" for a in active)
