"""
SEM physics model for shape-from-shading surface reconstruction.

Forward model maps a height field h(x,y) to an SEM image using the
secondary-electron (SE) yield approximation and a generalised detector
geometry.

Physics model
-------------
SE yield (first-order approximation):
    eta(x,y) = eta0 / cos(theta) = eta0 * sqrt(1 + p^2 + q^2)

where p = dh/dx, q = dh/dy and theta is the local surface tilt from
the beam axis (vertical).

Image intensity:
    I(x,y) = I_bg + eta0 * sqrt(1 + p^2 + q^2) * (1 + lam * max(0, n . d))

    n = (-p, -q, 1) / |(-p, -q, 1)|   surface normal
    d = (sin(alpha)*cos(phi), sin(alpha)*sin(phi), cos(alpha))  detector unit vector
    lam in [0, 1]  detector channelling / asymmetry factor

Unknown parameters: I_bg, eta0, lam, alpha (elevation), phi (azimuth).

Typical ET detector values: alpha ~ 20-35 deg, phi ~ 0-90 deg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Detector parameters
# ---------------------------------------------------------------------------

@dataclass
class DetectorParams:
    """SEM detector + contrast parameters.  All angles in degrees."""
    I_bg:      float = 0.0     # background level (mean of flat substrate)
    eta0:      float = 1.0     # contrast scale
    lam:       float = 0.30    # detector asymmetry weight [0, 1]
    alpha_deg: float = 25.0    # detector elevation angle
    phi_deg:   float = 0.0     # detector azimuth angle

    # copy / override helper
    def replace(self, **kwargs) -> "DetectorParams":
        from dataclasses import replace as _r
        return _r(self, **kwargs)


# ---------------------------------------------------------------------------
# Differentiable forward model (NumPy — no PyTorch dependency required)
# ---------------------------------------------------------------------------

def _grad2d(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (p, q) = (dh/dx, dh/dy) via central differences, float32."""
    p = np.gradient(h.astype(np.float32), axis=1)
    q = np.gradient(h.astype(np.float32), axis=0)
    return p, q


def render(h: np.ndarray, params: DetectorParams) -> np.ndarray:
    """Forward model: height field -> SEM image intensity (float32)."""
    p, q = _grad2d(h)

    # SE yield factor = sec(theta) = sqrt(1 + p^2 + q^2)
    sec_theta = np.sqrt(1.0 + p ** 2 + q ** 2, dtype=np.float32)

    # Surface normal
    denom = sec_theta  # already computed
    nx = (-p / denom).astype(np.float32)
    ny = (-q / denom).astype(np.float32)
    nz = (1.0 / denom).astype(np.float32)

    # Detector unit vector
    a = math.radians(params.alpha_deg)
    f = math.radians(params.phi_deg)
    dx = math.sin(a) * math.cos(f)
    dy = math.sin(a) * math.sin(f)
    dz = math.cos(a)

    ndotd = np.clip(nx * dx + ny * dy + nz * dz, 0.0, None)

    I = (params.I_bg
         + params.eta0 * sec_theta * (1.0 + params.lam * ndotd))
    return I.astype(np.float32)


# ---------------------------------------------------------------------------
# Parameter auto-estimation
# ---------------------------------------------------------------------------

def estimate_params_from_image(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    flat_region_mask: Optional[np.ndarray] = None,
) -> DetectorParams:
    """
    Estimate I_bg and eta0 from image statistics.

    If flat_region_mask is provided (user-drawn rectangle on bare substrate),
    I_bg = median of that region, eta0 = std * 3 (intensity range from flat).
    Otherwise fall back to global percentiles.

    alpha and phi are NOT estimable without 3D ground truth; defaults are
    returned and should be supplied by the user from instrument documentation.
    """
    img = image.astype(np.float32)
    if flat_region_mask is not None and flat_region_mask.any():
        pixels = img[flat_region_mask.astype(bool)]
        I_bg  = float(np.median(pixels))
        eta0  = float(np.std(pixels) * 3.0 + 1e-6)
    elif mask is not None and mask.any():
        pixels = img[mask.astype(bool)]
        I_bg  = float(np.percentile(pixels, 5))
        eta0  = float(np.percentile(pixels, 95) - I_bg + 1e-6)
    else:
        I_bg  = float(np.percentile(img, 5))
        eta0  = float(np.percentile(img, 95) - I_bg + 1e-6)

    return DetectorParams(I_bg=I_bg, eta0=eta0)


# ---------------------------------------------------------------------------
# Shape-from-shading (variational, PyTorch-based for GPU + autodiff)
# ---------------------------------------------------------------------------

def shape_from_shading(
    I_obs: np.ndarray,
    mask: np.ndarray,
    params: DetectorParams,
    *,
    n_iters: int = 300,
    lr: float = 5e-3,
    smoothness_weight: float = 0.10,
    learn_detector: bool = False,
    device: str = "cpu",
    progress_cb: Callable[[int, int], None] | None = None,
    stop_flag: list | None = None,
) -> np.ndarray:
    """
    Recover height field h(x,y) from observed SEM intensity via variational
    optimisation.

    Minimises:
        L_data      = ||I_render(h) - I_obs||^2  (inside mask)
        L_smooth    = smoothness_weight * ||Laplacian(h)||^2
        L_boundary  = ||h at mask edge||^2  (h=0 on substrate)

    If learn_detector=True, I_bg, eta0, lam, alpha, phi are also optimised
    (useful when detector geometry is unknown; requires complex particle shape
    for identifiability — flag this mode as experimental).

    Returns h as float32 numpy array, same shape as I_obs.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for SEM shape-from-shading:\n"
            "  pip install torch\n"
        ) from exc

    dev = torch.device(device if device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))

    I_t = torch.as_tensor(I_obs.astype(np.float32), device=dev)
    mask_t = torch.as_tensor(mask.astype(np.float32), device=dev)

    # Linear initialisation: first-order approximation (flat-field)
    # h_init ~ (I - I_bg) / eta0 - 1  (ignores detector asymmetry)
    h_init = ((I_t - params.I_bg) / (params.eta0 + 1e-8) - 1.0).clamp(min=0.0)
    h_init = h_init * mask_t
    h = h_init.clone().detach().requires_grad_(True)

    opt_params = [h]

    # Learnable detector params (experimental mode)
    if learn_detector:
        log_eta0   = torch.tensor(math.log(max(params.eta0, 1e-6)), device=dev, requires_grad=True)
        I_bg_t     = torch.tensor(params.I_bg,   device=dev, requires_grad=True)
        logit_lam  = torch.tensor(math.log(params.lam / (1 - params.lam + 1e-8) + 1e-8),
                                  device=dev, requires_grad=True)
        alpha_t    = torch.tensor(math.radians(params.alpha_deg), device=dev, requires_grad=True)
        phi_t      = torch.tensor(math.radians(params.phi_deg),   device=dev, requires_grad=True)
        opt_params += [log_eta0, I_bg_t, logit_lam, alpha_t, phi_t]

    optimizer = torch.optim.Adam(opt_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=lr * 0.1)

    a_fixed = math.radians(params.alpha_deg)
    f_fixed = math.radians(params.phi_deg)

    for it in range(n_iters):
        if stop_flag and stop_flag[0]:
            break

        optimizer.zero_grad()

        # Enforce boundary: zero outside mask after each step
        h_masked = h * mask_t

        # Compute gradients (central differences via conv)
        ph = F.pad(h_masked.unsqueeze(0).unsqueeze(0), (1, 1, 0, 0), mode='replicate')
        p_t = (ph[..., 2:] - ph[..., :-2]) / 2.0
        p_t = p_t.squeeze()

        qh = F.pad(h_masked.unsqueeze(0).unsqueeze(0), (0, 0, 1, 1), mode='replicate')
        q_t = (qh[..., 2:, :] - qh[..., :-2, :]) / 2.0
        q_t = q_t.squeeze()

        sec_theta = torch.sqrt(1.0 + p_t ** 2 + q_t ** 2)
        denom = sec_theta
        nx = -p_t / denom
        ny = -q_t / denom
        nz = 1.0 / denom

        if learn_detector:
            eta0_v  = torch.exp(log_eta0)
            I_bg_v  = I_bg_t
            lam_v   = torch.sigmoid(logit_lam)
            dx = torch.sin(alpha_t) * torch.cos(phi_t)
            dy = torch.sin(alpha_t) * torch.sin(phi_t)
            dz = torch.cos(alpha_t)
        else:
            eta0_v = params.eta0
            I_bg_v = params.I_bg
            lam_v  = params.lam
            dx = math.sin(a_fixed) * math.cos(f_fixed)
            dy = math.sin(a_fixed) * math.sin(f_fixed)
            dz = math.cos(a_fixed)

        ndotd = torch.clamp(nx * dx + ny * dy + nz * dz, min=0.0)
        I_pred = I_bg_v + eta0_v * sec_theta * (1.0 + lam_v * ndotd)

        # Data loss (inside mask only, erode by 1 pixel to avoid edge effects)
        kern = torch.ones(1, 1, 3, 3, device=dev)
        mask_eroded = (F.conv2d(mask_t.unsqueeze(0).unsqueeze(0),
                                kern, padding=1).squeeze() >= 9.0).float()
        diff = (I_pred - I_t) * mask_eroded
        L_data = (diff ** 2).mean()

        # Laplacian smoothness loss
        lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                              dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
        lap = F.conv2d(h_masked.unsqueeze(0).unsqueeze(0),
                       lap_k, padding=1).squeeze()
        L_smooth = smoothness_weight * (lap ** 2).mean()

        # Boundary penalty (h must be 0 on mask edge)
        boundary = mask_t - mask_eroded
        L_boundary = (h_masked * boundary).pow(2).mean() * 10.0

        loss = L_data + L_smooth + L_boundary

        if learn_detector:
            # Regularise angles toward priors
            L_prior = (0.5 * (alpha_t - math.radians(params.alpha_deg)) ** 2
                       + 0.5 * (phi_t   - math.radians(params.phi_deg))   ** 2)
            loss = loss + L_prior

        loss.backward()
        optimizer.step()
        scheduler.step()

        if progress_cb and (it % 20 == 0 or it == n_iters - 1):
            progress_cb(it + 1, n_iters)

    with torch.no_grad():
        h_out = (h * mask_t).cpu().numpy().astype(np.float32)

    return h_out


# ---------------------------------------------------------------------------
# Surface area from height field
# ---------------------------------------------------------------------------

def surface_area_from_height(
    h: np.ndarray,
    mask: np.ndarray,
    pixel_size_nm: float,
) -> float:
    """
    True 3D surface area of the visible (top) hemisphere:
        SA = sum_pixels sqrt(1 + p^2 + q^2) * px^2   (within mask)

    Returns area in nm^2.
    """
    p, q = _grad2d(h)
    integrand = np.sqrt(1.0 + p ** 2 + q ** 2, dtype=np.float32)
    return float(np.sum(integrand[mask.astype(bool)]) * pixel_size_nm ** 2)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SEMParticleResult:
    particle_id:       int   = 0
    label:             str   = ""
    image_name:        str   = ""
    SA_sem_nm2:        float = 0.0   # physics (+ NN if available)
    SA_2d_nm2:         float = 0.0   # fallback: projected area × 4 (sphere approx)
    roughness_rms:     float = 0.0   # RMS height variation within mask
    detector_alpha:    float = 0.0
    detector_phi:      float = 0.0
    method:            str   = "sem_physics"
    flagged:           bool  = False
    flag_reason:       str   = ""
