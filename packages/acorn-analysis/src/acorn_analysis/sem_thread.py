"""Background thread for SEM physics-based 3D surface area estimation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal


class SEMAnalysisThread(QThread):
    """
    Per-particle SEM surface area estimation.

    For each item (vertices, label, image_name, image_path, pixel_size_nm):
      1. Build binary mask from polygon vertices.
      2. Extract raw image crop from the source file.
      3. Auto-estimate I_bg / eta0 from the crop if not supplied.
      4. Run shape-from-shading (Adam optimisation) to recover h(x,y).
      5. Optionally apply U-Net residual correction.
      6. Compute SA = integral sqrt(1 + p^2 + q^2) * px^2 within mask.
      7. Compute fallback 2D-projected SA for comparison.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, str)   # (DataFrame, out_dir_str)
    error    = pyqtSignal(str)

    def __init__(
        self,
        items: list[dict],
        detector_config: dict,
        sfs_config: dict,
        nn_config: dict,
        output_dir: str,
        device: str = "auto",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._items       = items
        self._det         = detector_config   # alpha_deg, phi_deg, I_bg, eta0, lam, learn_detector
        self._sfs         = sfs_config        # n_iters, smoothness, lr
        self._nn          = nn_config         # use_nn, checkpoint
        self._output_dir  = output_dir
        self._device      = device
        self._stop        = [False]

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

    def _run(self) -> None:
        import numpy as np
        import pandas as pd

        try:
            import cv2
        except ImportError as exc:
            self.error.emit(
                f"opencv-python is required:\n  pip install opencv-python\n\n{exc}"
            )
            return

        try:
            import torch
        except ImportError as exc:
            self.error.emit(
                f"PyTorch is required for SEM shape-from-shading:\n  pip install torch\n\n{exc}"
            )
            return

        from acorn.analysis.sem_physics import (
            DetectorParams, estimate_params_from_image,
            shape_from_shading, surface_area_from_height, SEMParticleResult,
        )
        from acorn.core.dm4_loader import DM4Image

        # Resolve device
        if self._device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = self._device

        params = DetectorParams(
            I_bg      = self._det.get("I_bg", 0.0),
            eta0      = self._det.get("eta0", 1.0),
            lam       = self._det.get("lam", 0.30),
            alpha_deg = self._det.get("alpha_deg", 25.0),
            phi_deg   = self._det.get("phi_deg",   0.0),
        )
        learn_det = self._det.get("learn_detector", False)

        use_nn   = self._nn.get("use_nn", False)
        ckpt     = self._nn.get("checkpoint", "")
        nn_model_loaded = False
        if use_nn and ckpt and Path(ckpt).exists():
            nn_model_loaded = True

        n = len(self._items)
        rows: list[dict] = []

        _cached_path: str = ""
        _cached_raw: Optional[np.ndarray] = None

        for i, item in enumerate(self._items):
            if self._stop[0]:
                break

            vertices  = item.get("vertices", [])
            label     = item.get("label", "")
            img_name  = item.get("image_name", "")
            img_path  = item.get("image_path", "")
            px_nm     = float(item.get("pixel_size_nm", 1.0))

            if not vertices or len(vertices) < 3:
                continue

            # Build mask
            pts   = np.array(vertices, dtype=np.float32)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            pad   = 8   # extra pixels around mask for boundary handling
            w     = int(x_max - x_min) + 2 * pad + 1
            h     = int(y_max - y_min) + 2 * pad + 1
            shift = np.array([x_min - pad, y_min - pad])
            shifted = pts - shift
            mask_u8 = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_u8, [shifted.astype(np.int32)], 1)
            mask = mask_u8.astype(bool)

            # Load raw image crop
            raw_crop: Optional[np.ndarray] = None
            if img_path:
                if img_path != _cached_path:
                    try:
                        _cached_raw = DM4Image.from_file(img_path).raw.astype(np.float32)
                        _cached_path = img_path
                    except Exception:
                        _cached_raw = None
                if _cached_raw is not None:
                    ry0 = max(0, int(y_min) - pad)
                    ry1 = min(_cached_raw.shape[0], ry0 + h)
                    rx0 = max(0, int(x_min) - pad)
                    rx1 = min(_cached_raw.shape[1], rx0 + w)
                    rc  = _cached_raw[ry0:ry1, rx0:rx1]
                    # Pad to (h, w) if needed due to image boundary
                    if rc.shape != (h, w):
                        padded = np.zeros((h, w), dtype=np.float32)
                        padded[:rc.shape[0], :rc.shape[1]] = rc
                        raw_crop = padded
                    else:
                        raw_crop = rc.copy()

            pct = 5 + int(85 * i / max(n, 1))
            self.progress.emit(pct, f"SEM SA {i + 1}/{n}  [{label}] — shape-from-shading…")

            # If I_bg/eta0 are at default zero/one, auto-estimate from this crop
            use_params = params
            if raw_crop is not None and (params.I_bg == 0.0 or params.eta0 == 1.0):
                auto = estimate_params_from_image(raw_crop, mask)
                use_params = params.replace(
                    I_bg  = auto.I_bg  if params.I_bg  == 0.0 else params.I_bg,
                    eta0  = auto.eta0  if params.eta0  == 1.0 else params.eta0,
                )

            # Shape-from-shading
            if raw_crop is not None:
                try:
                    h_field = shape_from_shading(
                        raw_crop,
                        mask,
                        use_params,
                        n_iters        = self._sfs.get("n_iters", 300),
                        lr             = self._sfs.get("lr", 5e-3),
                        smoothness_weight = self._sfs.get("smoothness", 0.10),
                        learn_detector = learn_det,
                        device         = dev,
                        stop_flag      = self._stop,
                    )
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning("SFS failed for %s: %s", img_name, exc)
                    h_field = np.zeros((h, w), dtype=np.float32)
            else:
                h_field = np.zeros((h, w), dtype=np.float32)

            # Optional U-Net residual correction
            method_used = "sem_sfs"
            if nn_model_loaded and raw_crop is not None:
                try:
                    from acorn.analysis.sem_unet import apply_nn_correction
                    h_field = apply_nn_correction(
                        raw_crop, h_field, mask.astype(np.float32),
                        ckpt, device=dev,
                    )
                    method_used = "sem_sfs+nn"
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning("NN correction failed: %s", exc)

            # Surface area
            SA_sem = surface_area_from_height(h_field, mask, px_nm)

            # Fallback 2D-based SA (projected area * 4 for sphere approximation)
            projected_area_nm2 = float(mask.sum()) * px_nm ** 2
            SA_2d = projected_area_nm2 * 4.0   # sphere: SA = 4 * pi_r^2, proj = pi_r^2

            # Roughness: RMS of height field within mask
            h_vals = h_field[mask]
            roughness = float(np.std(h_vals)) if len(h_vals) > 0 else 0.0

            flagged = raw_crop is None
            result = SEMParticleResult(
                particle_id   = i,
                label         = label,
                image_name    = img_name,
                SA_sem_nm2    = SA_sem,
                SA_2d_nm2     = SA_2d,
                roughness_rms = roughness,
                detector_alpha = use_params.alpha_deg,
                detector_phi   = use_params.phi_deg,
                method         = method_used,
                flagged        = flagged,
                flag_reason    = "no raw image" if flagged else "",
            )
            row = {k: v for k, v in vars(result).items()}
            rows.append(row)

        if not rows:
            self.error.emit(
                "No valid ROI annotations found for the selected labels.\n"
                "Make sure annotations are ROI polygons."
            )
            return

        df = pd.DataFrame(rows)
        self.progress.emit(95, "Saving results…")

        out_dir = self._output_dir or ""
        if out_dir:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(str(out_path / "sem_sa_per_particle.csv"), index=False)

        self.progress.emit(100, "Done.")
        self.finished.emit(df, out_dir)

    def stop(self) -> None:
        self._stop[0] = True


class SEMTrainThread(QThread):
    """Train the SEM U-Net on synthetic data in the background."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)   # checkpoint path
    error    = pyqtSignal(str)

    def __init__(self, config: dict, detector_params=None, parent=None) -> None:
        super().__init__(parent)
        self._config  = config
        self._dp      = detector_params

    def run(self) -> None:
        try:
            from acorn.analysis.sem_unet import train_sem_unet
            ckpt = train_sem_unet(
                output_dir      = self._config.get("output_dir", "."),
                n_samples       = self._config.get("n_samples", 2000),
                epochs          = self._config.get("epochs", 50),
                batch_size      = self._config.get("batch_size", 16),
                image_size      = self._config.get("image_size", 128),
                detector_params = self._dp,
                shape_types     = self._config.get("shape_types") or None,
                device          = "auto",
                log_cb         = lambda msg: self.progress.emit(0, msg),
                progress_cb    = lambda ep, tot: self.progress.emit(
                    int(100 * ep / tot), f"Epoch {ep}/{tot}"
                ),
            )
            self.finished.emit(str(ckpt))
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")
