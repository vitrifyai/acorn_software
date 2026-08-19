"""Headless CryoBLOB detection runner — calls the plugin detector directly.

Usage: run_cryoblob_headless.py <detection_mode> <use_watershed 0|1> <out_csv> <img1> [img2 ...]
Writes an ACORN-format CryoBLOB CSV so validate_cryoblob.py can score it.
"""
import sys, csv
sys.path.insert(0, "/home/vnw/acorn-cryoblob-plugin/src")
from acorn_cryoblob.thread import _process_single_file

detection_mode = sys.argv[1]
use_watershed = bool(int(sys.argv[2]))
out_csv = sys.argv[3]
images = sys.argv[4:]

PARAMS = dict(
    pixel_size_nm=0.5, run_mode="final",
    blob_downscale=4.0, min_sigma=8.0, max_sigma=47.0, blob_step=1.0,
    threshold_rel=0.15, max_detections=100, refine_sizes=True, size_scale=1.0,
    ridge_threshold=0.006, ridge_scales=20, min_marker_distance=4.0,
    use_ridge_detection=False, stream_large_files=False,
    exponential=False, logarizer=False, gblur=2, background=0, apply_filter=0,
    cache_results=False,
)

all_records = []
for img in images:
    recs = _process_single_file(img, detection_mode=detection_mode,
                                use_watershed=use_watershed, **PARAMS)
    print(f"{img}: {len(recs)} detections", flush=True)
    all_records.extend(recs)

if all_records:
    cols = list(all_records[0].keys())
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_records)
    print(f"wrote {len(all_records)} rows -> {out_csv}")
else:
    print("no detections")
