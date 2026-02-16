#!/usr/bin/env python3
"""
maya_scan.py

One-file Maya LiDAR pipeline:
- LAZ/LAS -> DTM (PDAL) (GeoTIFF)
- DTM -> multi-scale Local Relief Model (LRM) (multi-sigma, pixel-based)
- LRM -> candidate structures (connected components + terrain filters)
- candidates -> density raster + scores
- candidates -> clustering (DBSCAN; eps can be auto-chosen) **in meters** (auto-UTM if needed)
- exports: CSV, GeoJSON, KML (labels only top-N; rest unlabeled)
- plots + Markdown + optional PDF report
- HTML report w/ Leaflet map + candidate cutout panels (LRM + hillshade)

Project-goal improvements:
- Region shape metrics: bbox fill "extent" (area / bbox_area), aspect ratio, width/height (m)
- Post-filters: min_peak, min_area_m2, min_extent, max_aspect, min_prominence, min_compactness, min_solidity
- Score supports extent/prominence/compactness/solidity terms
- Extra plots: extent/aspect/compactness/solidity/prominence histograms
- Safer out_dir: only delete existing run with --overwrite

Dependencies:
- Required: PDAL (system install), numpy, scipy, rasterio, pyproj, matplotlib
- Optional: scikit-learn (DBSCAN clustering), reportlab (PDF report)

Responsible use:
- Output coordinates can correspond to sensitive locations. Handle responsibly.

Example:
python maya_scan.py \
  -i data/lidar/bz_hr_las31_crs.laz \
  --name bz_hr_tile_31_v2_shape_filters \
  --overwrite \
  --try-smrf \
  --pos-thresh auto:p96 \
  --min-density auto:p60 \
  --density-sigma 40 \
  --min-peak 0.50 \
  --min-area-m2 25 \
  --min-extent 0.38 \
  --max-aspect 3.5 \
  --min-prominence 0.10 \
  --max-slope-deg 20 \
  --min-compactness 0.12 \
  --min-solidity 0.50 \
  --cluster-eps auto \
  --min-samples 4 \
  --report-top-n 30 \
  --label-top-n 60
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import xy as pix2map_xy
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening, gaussian_filter, label as cc_label
from scipy.spatial import ConvexHull, QhullError
from pyproj import CRS, Transformer

# Optional deps
try:
    from sklearn.cluster import DBSCAN  # type: ignore
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:
    DBSCAN = None  # noqa
    NearestNeighbors = None  # noqa

try:
    from reportlab.lib.pagesizes import letter  # type: ignore
    from reportlab.pdfgen import canvas  # type: ignore
except Exception:
    canvas = None  # noqa

LOG = logging.getLogger("maya_scan")


# -----------------------------
# Logging
# -----------------------------
def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "process.log"

    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    LOG.addHandler(ch)
    LOG.addHandler(fh)
    LOG.info("Logging to %s", log_path)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = (proc.stdout or "").strip()
    if out:
        LOG.info(out)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")


# -----------------------------
# Parameters / defaults
# -----------------------------
@dataclass
class Params:
    dtm_resolution_m: float = 1.0

    # Multi-scale LRM sigmas (pixels)
    lrm_sigmas_small: Tuple[float, ...] = (1.0, 2.0)
    lrm_sigmas_large: Tuple[float, ...] = (8.0, 12.0, 16.0)

    # Candidate detection
    pos_relief_threshold_spec: str = "auto:p96"
    min_region_pixels: int = 20
    morph_open_iters: int = 1
    morph_close_iters: int = 1
    # Region-level terrain filter: drop regions whose slope q75 exceeds this.
    max_slope_deg: float = 20.0

    # Density
    density_sigma_pix: float = 40.0
    min_density_spec: str = "auto:p60"

    # Post-filters (shape/physics)
    min_peak_relief_m: float = 0.50        # e.g. 0.4 to cut noise
    min_area_m2: float = 25.0              # e.g. 30
    min_extent: float = 0.38               # bbox fill, 0..1 (e.g. 0.35)
    max_aspect: float = 3.5                # width/height or height/width (e.g. 4.0)
    prominence_ring_pixels: int = 6        # ring width (pixels) for local prominence estimate
    min_prominence_m: float = 0.10         # region mean relief - local ring mean relief
    min_compactness: float = 0.12          # 4*pi*A/P^2 in [0,1], lower = line-like
    min_solidity: float = 0.50             # A / convex_hull_A in [0,1], lower = fragmented/linear

    # Clustering
    cluster_eps_m: float = 150.0
    cluster_eps_mode: str = "auto"  # "auto" or "fixed"
    cluster_min_samples: int = 4

    # Outputs
    kml_label_top_n: int = 50
    report_top_n: int = 25

    # Cutouts / HTML
    html_report: bool = True
    cutout_size_m: float = 140.0  # square window width in meters
    cutout_dpi: int = 160

    # Score
    # score = density^a * peak_relief^b * extent^c * prominence^d * compactness^e * solidity^f * area_m2^g
    score_density_exp: float = 1.0
    score_peak_exp: float = 1.0
    score_extent_exp: float = 0.35
    score_prominence_exp: float = 0.75
    score_compactness_exp: float = 0.20
    score_solidity_exp: float = 0.20
    score_area_exp: float = 0.50


_RUN_NAME_BAD_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
_AUTO_PERCENTILE_RE = re.compile(r"^auto:p([0-9]+(?:\.[0-9]+)?)$", re.IGNORECASE)


def _parse_float_arg(raw: str, field: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{field} must be a number, got '{raw}'.") from exc


def _arg_positive_float(raw: str) -> float:
    v = _parse_float_arg(raw, "Value")
    if v <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return v


def _arg_nonnegative_float(raw: str) -> float:
    v = _parse_float_arg(raw, "Value")
    if v < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return v


def _arg_unit_interval(raw: str) -> float:
    v = _parse_float_arg(raw, "Value")
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError("Value must be between 0 and 1.")
    return v


def _arg_ge_one_float(raw: str) -> float:
    v = _parse_float_arg(raw, "Value")
    if v < 1.0:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return v


def _arg_positive_int(raw: str) -> int:
    try:
        v = int(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Value must be an integer, got '{raw}'.") from exc
    if v < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return v


def _arg_nonnegative_int(raw: str) -> int:
    try:
        v = int(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Value must be an integer, got '{raw}'.") from exc
    if v < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return v


def _normalized_auto_percentile_spec(raw: str) -> Optional[str]:
    s = str(raw).strip().lower()
    m = _AUTO_PERCENTILE_RE.match(s)
    if not m:
        return None
    p = float(m.group(1))
    if not (0.0 <= p <= 100.0):
        raise argparse.ArgumentTypeError("Auto percentile must be between 0 and 100 (auto:pXX).")
    return f"auto:p{p:g}"


def _arg_pos_thresh_spec(raw: str) -> str:
    auto = _normalized_auto_percentile_spec(raw)
    if auto is not None:
        return auto
    _ = _parse_float_arg(str(raw).strip(), "pos-thresh")
    return str(raw).strip().lower()


def _arg_min_density_spec(raw: str) -> str:
    auto = _normalized_auto_percentile_spec(raw)
    if auto is not None:
        return auto
    v = _parse_float_arg(str(raw).strip(), "min-density")
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError("min-density numeric value must be between 0 and 1.")
    return str(raw).strip().lower()


def _arg_cluster_eps_spec(raw: str) -> str:
    s = str(raw).strip().lower()
    if s == "auto":
        return "auto"
    v = _parse_float_arg(s, "cluster-eps")
    if v <= 0:
        raise argparse.ArgumentTypeError("cluster-eps must be > 0 or 'auto'.")
    return f"{v:g}"


def sanitize_run_name(raw: str) -> str:
    name = str(raw).strip()
    name = _RUN_NAME_BAD_CHARS_RE.sub("_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("._-")
    if not name:
        raise ValueError("Run name is empty after sanitization.")
    if name in {".", ".."}:
        raise ValueError("Run name cannot be '.' or '..'.")
    return name[:120]


def ensure_path_within(parent: Path, child: Path) -> None:
    try:
        child.relative_to(parent)
    except ValueError as exc:
        raise RuntimeError(
            f"Resolved output path is outside runs-dir.\n"
            f"runs-dir={parent}\n"
            f"out-dir={child}"
        ) from exc


# -----------------------------
# PDAL DTM builder
# -----------------------------
def pdal_version() -> str:
    try:
        proc = subprocess.run(["pdal", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return (proc.stdout or "").strip()
    except Exception:
        return "pdal (unknown)"


def build_dtm_from_laz(
    laz_path: Path,
    out_dtm_tif: Path,
    tmp_dir: Path,
    resolution_m: float,
    try_smrf: bool,
) -> None:
    """
    Robust PDAL approach:
    - optionally classify ground with SMRF into temp LAS
    - write DTM via writers.gdal using gdalopts
    """
    if not laz_path.exists():
        raise FileNotFoundError(
            f"Input LAZ/LAS not found: {laz_path}\n"
            f"Tip: pass the real tile path."
        )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    in_path = laz_path

    if try_smrf:
        smrf_out = tmp_dir / "ground_smrf.las"
        smrf_json = {
            "pipeline": [
                {"type": "readers.las", "filename": str(in_path)},
                {
                    "type": "filters.smrf",
                    "ignore": "Classification[7:7]",
                    "scalar": 1.25,
                    "slope": 0.15,
                    "threshold": 0.5,
                    "window": 16.0,
                },
                {"type": "writers.las", "filename": str(smrf_out)},
            ]
        }
        smrf_path = tmp_dir / "pdal_smrf.json"
        smrf_path.write_text(json.dumps(smrf_json, indent=2), encoding="utf-8")
        LOG.info("Running: pdal pipeline %s", smrf_path)
        run_cmd(["pdal", "pipeline", str(smrf_path)])
        in_path = smrf_out

    pipeline: List[Dict[str, Any]] = [{"type": "readers.las", "filename": str(in_path)}]
    if try_smrf:
        pipeline.append({"type": "filters.range", "limits": "Classification[2:2]"})

    pipeline.append(
        {
            "type": "writers.gdal",
            "filename": str(out_dtm_tif),
            "resolution": float(resolution_m),
            "output_type": "min",
            "dimension": "Z",
            "data_type": "float32",
            "nodata": -9999.0,
            "gdaldriver": "GTiff",
            "gdalopts": [
                "TILED=YES",
                "BLOCKXSIZE=256",
                "BLOCKYSIZE=256",
                "COMPRESS=DEFLATE",
                "PREDICTOR=2",
                "BIGTIFF=IF_SAFER",
            ],
        }
    )

    dtm_json = {"pipeline": pipeline}
    dtm_path = tmp_dir / "pdal_dtm.json"
    dtm_path.write_text(json.dumps(dtm_json, indent=2), encoding="utf-8")
    LOG.info("Running: pdal pipeline %s", dtm_path)
    run_cmd(["pdal", "pipeline", str(dtm_path)])

    if not out_dtm_tif.exists():
        raise RuntimeError(f"DTM did not get written (expected {out_dtm_tif})")


# -----------------------------
# Raster helpers
# -----------------------------
def load_raster(path: Path) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def _res_m_from_profile(profile: Dict[str, Any]) -> float:
    t = profile["transform"]
    resx = float(abs(t.a))
    resy = float(abs(t.e))
    return float((resx + resy) / 2.0)


def write_float_geotiff(path: Path, arr: np.ndarray, base_profile: Dict[str, Any]) -> None:
    prof = dict(base_profile)
    prof.update(
        dtype="float32",
        count=1,
        nodata=None,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        predictor=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)


def compute_slope_degrees(dtm: np.ndarray, res_m: float) -> np.ndarray:
    fill = np.nanmedian(dtm)
    dtm_f = np.where(np.isnan(dtm), fill, dtm)
    dz_dy, dz_dx = np.gradient(dtm_f, res_m, res_m)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    return np.degrees(slope_rad).astype("float32")


def hillshade(dtm: np.ndarray, res_m: float, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    fill = np.nanmedian(dtm)
    z = np.where(np.isnan(dtm), fill, dtm).astype("float32")
    dy, dx = np.gradient(z, res_m, res_m)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1).astype("float32")


# -----------------------------
# LRM
# -----------------------------
def build_multiscale_lrm(dtm: np.ndarray, params: Params) -> np.ndarray:
    fill = np.nanmedian(dtm)
    dtm_f = np.where(np.isnan(dtm), fill, dtm).astype("float32")

    lrms: List[np.ndarray] = []
    for s_small in params.lrm_sigmas_small:
        small = gaussian_filter(dtm_f, sigma=s_small)
        for s_large in params.lrm_sigmas_large:
            if s_large <= s_small:
                continue
            large = gaussian_filter(dtm_f, sigma=s_large)
            lrms.append(small - large)

    if not lrms:
        raise RuntimeError("No valid sigma pairs for LRM")
    return np.maximum.reduce(lrms).astype("float32")


def parse_auto_percentile(spec: str, values: np.ndarray, positive_only: bool = True) -> float:
    spec_norm = str(spec).strip().lower()
    if not spec_norm.startswith("auto:p"):
        return float(spec_norm)

    m = _AUTO_PERCENTILE_RE.match(spec_norm)
    if not m:
        raise ValueError(f"Invalid auto percentile spec: '{spec}'. Expected auto:pXX.")
    p = float(m.group(1))
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"Auto percentile must be between 0 and 100, got {p}.")

    vals = values[np.isfinite(values)]
    if positive_only:
        vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, p))


# -----------------------------
# Candidates
# -----------------------------
@dataclass
class Candidate:
    cand_id: int
    px_x: float
    px_y: float
    peak_relief_m: float
    mean_relief_m: float
    area_m2: float
    density: float
    extent: float              # bbox fill (0..1)
    aspect: float              # >=1
    prominence_m: float        # mean relief over region minus local ring mean
    compactness: float         # 4*pi*A/P^2 in [0,1]
    solidity: float            # A / convex_hull_A in [0,1]
    width_m: float
    height_m: float
    score: float
    lon: float
    lat: float
    cluster_id: int = -1
    dist_to_core_km: Optional[float] = None
    img_relpath: Optional[str] = None


def _region_perimeter_pixels(region_mask: np.ndarray) -> float:
    """
    Approximate perimeter as boundary-pixel count after 1-pixel erosion.
    Good enough for compactness filtering on raster regions.
    """
    if region_mask.size == 0:
        return 0.0
    eroded = binary_erosion(region_mask)
    boundary = region_mask & ~eroded
    return float(boundary.sum())


def _region_solidity(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Solidity = area / convex_hull_area in raster pixel units.
    Values near 1 are compact/filled; lower values are fragmented/linear.
    """
    n = int(xs.size)
    if n < 3:
        return 1.0
    pts = np.column_stack([xs.astype("float64"), ys.astype("float64")])
    try:
        hull = ConvexHull(pts)
        hull_area_pix2 = float(hull.volume)  # 2D hull area
    except (QhullError, ValueError):
        return 1.0
    if hull_area_pix2 <= 1e-9:
        return 1.0
    return float(np.clip(float(n) / hull_area_pix2, 0.0, 1.0))


def detect_regions(
    lrm: np.ndarray,
    dtm_slope_deg: np.ndarray,
    profile: Dict[str, Any],
    params: Params,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Extract connected components above pos_relief threshold.
    Compute shape metrics per region.
    Returns regions list + pos_thresh used.
    """
    _, regions, _, pos_thresh = _extract_candidate_regions(lrm, dtm_slope_deg, profile, params)
    return regions, pos_thresh


def _extract_candidate_regions(
    lrm: np.ndarray,
    dtm_slope_deg: np.ndarray,
    profile: Dict[str, Any],
    params: Params,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[int], float]:
    res_m = _res_m_from_profile(profile)

    pos_thresh = parse_auto_percentile(params.pos_relief_threshold_spec, lrm, positive_only=True)
    LOG.info("Positive relief threshold (m): %.4f (spec=%s)", pos_thresh, params.pos_relief_threshold_spec)

    lrm_filled = np.where(np.isfinite(lrm), lrm, 0.0).astype("float32")
    mask = lrm_filled > float(pos_thresh)

    mask = binary_opening(mask, iterations=params.morph_open_iters)
    mask = binary_closing(mask, iterations=params.morph_close_iters)

    labeled, n = cc_label(mask)
    LOG.info("Initial regions: %d", n)

    regions: List[Dict[str, Any]] = []
    kept_rids: List[int] = []

    for rid in range(1, n + 1):
        region = labeled == rid
        pix = int(region.sum())
        if pix < params.min_region_pixels:
            continue

        ys, xs = np.where(region)
        cy = float(ys.mean())
        cx = float(xs.mean())

        slope_vals = dtm_slope_deg[region]
        slope_vals = slope_vals[np.isfinite(slope_vals)]
        if slope_vals.size == 0:
            continue
        slope_median_deg = float(np.percentile(slope_vals, 50))
        slope_q75_deg = float(np.percentile(slope_vals, 75))
        slope_max_deg = float(np.max(slope_vals))
        if slope_q75_deg > params.max_slope_deg:
            continue

        peak = float(np.nanmax(lrm[region]))
        mean = float(np.nanmean(lrm[region]))
        ring_iters = max(1, int(params.prominence_ring_pixels))
        ring_mask = binary_dilation(region, iterations=ring_iters) & ~region
        ring_vals = lrm[ring_mask]
        ring_vals = ring_vals[np.isfinite(ring_vals)]
        ring_mean_relief_m = float(np.mean(ring_vals)) if ring_vals.size else mean
        prominence_m = float(mean - ring_mean_relief_m)

        # bbox shape metrics
        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())
        w_pix = float((x1 - x0 + 1))
        h_pix = float((y1 - y0 + 1))
        w_m = w_pix * res_m
        h_m = h_pix * res_m

        bbox_area_m2 = max(1e-9, w_m * h_m)
        area_m2 = pix * res_m * res_m

        extent = float(np.clip(area_m2 / bbox_area_m2, 0.0, 1.0))
        aspect = float(max(w_m / max(1e-9, h_m), h_m / max(1e-9, w_m)))  # >=1
        perimeter_m = _region_perimeter_pixels(region) * res_m
        compactness = float(
            np.clip((4.0 * math.pi * area_m2) / max(1e-9, perimeter_m * perimeter_m), 0.0, 1.0)
        )
        solidity = _region_solidity(xs, ys)

        regions.append(
            {
                "rid": rid,
                "cx": cx,
                "cy": cy,
                "pixels": pix,
                "area_m2": area_m2,
                "peak": peak,
                "mean": mean,
                "ring_mean_relief_m": ring_mean_relief_m,
                "prominence_m": prominence_m,
                "extent": extent,
                "aspect": aspect,
                "width_m": w_m,
                "height_m": h_m,
                "perimeter_m": perimeter_m,
                "compactness": compactness,
                "solidity": solidity,
                "slope_median_deg": slope_median_deg,
                "slope_q75_deg": slope_q75_deg,
                "slope_max_deg": slope_max_deg,
            }
        )
        kept_rids.append(rid)

    LOG.info("Filtered regions (after size + region-slope q75): %d", len(regions))
    return labeled, regions, kept_rids, pos_thresh


def build_density_from_regions(
    labeled_regions: np.ndarray,
    kept_rids: List[int],
    profile: Dict[str, Any],
    params: Params,
    out_density_tif: Path,
) -> np.ndarray:
    mound_binary = np.isin(labeled_regions, np.array(kept_rids, dtype=int)).astype("float32")
    density = gaussian_filter(mound_binary, sigma=float(params.density_sigma_pix)).astype("float32")
    dmin = float(np.min(density))
    dmax = float(np.max(density))
    density_norm = ((density - dmin) / (dmax - dmin + 1e-9)).astype("float32")

    write_float_geotiff(out_density_tif, density_norm, profile)
    LOG.info("Wrote density raster: %s", out_density_tif)
    return density_norm


def detect_candidates(
    lrm: np.ndarray,
    dtm_slope_deg: np.ndarray,
    profile: Dict[str, Any],
    params: Params,
    out_density_tif: Path,
) -> Tuple[List[Dict[str, Any]], np.ndarray, float, float]:
    """
    Convenience wrapper:
    - creates labeled components & regions
    - adds region-level density stats (mean/q75 over each region footprint)
    - writes density raster from kept regions (size+slope kept)
    - returns regions, density_norm, pos_thresh, min_density
    """
    labeled, regions, kept_rids, pos_thresh = _extract_candidate_regions(lrm, dtm_slope_deg, profile, params)

    density_norm = build_density_from_regions(labeled, kept_rids, profile, params, out_density_tif)

    for r in regions:
        rid = int(r["rid"])
        dens_vals = density_norm[labeled == rid]
        dens_vals = dens_vals[np.isfinite(dens_vals)]
        if dens_vals.size == 0:
            r["density_mean"] = 0.0
            r["density_q75"] = 0.0
            continue
        r["density_mean"] = float(np.mean(dens_vals))
        r["density_q75"] = float(np.percentile(dens_vals, 75))

    min_density = parse_auto_percentile(params.min_density_spec, density_norm, positive_only=False)
    LOG.info("Min density threshold: %.4f (spec=%s)", min_density, params.min_density_spec)
    return regions, density_norm, pos_thresh, min_density


# -----------------------------
# Clustering (meters)
# -----------------------------
def _auto_dbscan_eps(coords_m: np.ndarray, min_samples: int) -> float:
    if NearestNeighbors is None or coords_m.shape[0] < max(10, min_samples + 1):
        return 150.0
    k = max(2, int(min_samples))
    nn = NearestNeighbors(n_neighbors=min(k + 1, coords_m.shape[0]))
    nn.fit(coords_m)
    dists, _ = nn.kneighbors(coords_m)
    kth = dists[:, -1]
    eps = float(np.percentile(kth, 85))
    return float(np.clip(eps, 60.0, 300.0))


def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def _projected_unit_factor_to_meters(src_crs: CRS) -> Optional[float]:
    factors: List[float] = []
    for axis in src_crs.axis_info:
        fac = getattr(axis, "unit_conversion_factor", None)
        if fac is None:
            continue
        try:
            val = float(fac)
        except (TypeError, ValueError):
            continue
        if val > 0:
            factors.append(val)

    if not factors:
        return None

    first = factors[0]
    for val in factors[1:]:
        if not math.isclose(first, val, rel_tol=1e-9, abs_tol=1e-12):
            LOG.warning(
                "Projected CRS has mixed axis conversion factors; using first factor %.12g.",
                first,
            )
            break
    return first


def project_points_to_meters(
    src_crs: CRS,
    xs: np.ndarray,
    ys: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, CRS]:
    """
    Ensure we have meter-based coords for clustering/distances.
    If src_crs is projected (meters-ish), pass through.
    If geographic (degrees), convert to EPSG:4326 then to UTM based on centroid.
    """
    xs64 = xs.astype("float64")
    ys64 = ys.astype("float64")

    if src_crs.is_projected:
        factor = _projected_unit_factor_to_meters(src_crs)
        if factor is not None:
            if math.isclose(factor, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                return xs64, ys64, src_crs
            LOG.info("Projected CRS units converted to meters (factor=%.12g).", factor)
            return xs64 * factor, ys64 * factor, src_crs
        LOG.warning("Projected CRS unit conversion factor unavailable; reprojecting to UTM for meter distances.")

    lon = xs64
    lat = ys64
    if src_crs.to_epsg() != 4326:
        to_ll = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform(lon, lat)

    clon = float(np.mean(lon))
    clat = float(np.mean(lat))
    utm_epsg = _utm_epsg_from_lonlat(clon, clat)
    utm_crs = CRS.from_epsg(utm_epsg)
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x_m, y_m = to_utm.transform(lon, lat)
    return np.array(x_m, dtype="float64"), np.array(y_m, dtype="float64"), utm_crs


def cluster_candidates_meters(xs_m: np.ndarray, ys_m: np.ndarray, params: Params) -> np.ndarray:
    if DBSCAN is None:
        LOG.warning("sklearn not installed; clustering disabled (cluster_id=-1).")
        return np.full(xs_m.shape[0], -1, dtype=int)

    coords = np.column_stack([xs_m, ys_m]).astype("float32")

    eps = float(params.cluster_eps_m)
    if params.cluster_eps_mode == "auto":
        eps = _auto_dbscan_eps(coords, int(params.cluster_min_samples))
        LOG.info("DBSCAN eps auto-chosen: %.1f m (min_samples=%d)", eps, params.cluster_min_samples)
    else:
        LOG.info("DBSCAN eps fixed: %.1f m (min_samples=%d)", eps, params.cluster_min_samples)

    model = DBSCAN(eps=eps, min_samples=int(params.cluster_min_samples))
    labels = model.fit_predict(coords)

    out = labels.copy()
    next_id = 1
    mapping: Dict[int, int] = {}
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        mapping[lab] = next_id
        next_id += 1
    for i, lab in enumerate(labels):
        out[i] = mapping.get(lab, -1)
    return out.astype(int)


# -----------------------------
# Exporters
# -----------------------------
def write_geojson(candidates: List[Candidate], out_path: Path) -> None:
    feats = []
    for c in candidates:
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [c.lon, c.lat]},
                "properties": {
                    "cand_id": c.cand_id,
                    "score": c.score,
                    "density": c.density,
                    "peak_relief_m": c.peak_relief_m,
                    "mean_relief_m": c.mean_relief_m,
                    "area_m2": c.area_m2,
                    "extent": c.extent,
                    "aspect": c.aspect,
                    "prominence_m": c.prominence_m,
                    "compactness": c.compactness,
                    "solidity": c.solidity,
                    "width_m": c.width_m,
                    "height_m": c.height_m,
                    "cluster_id": c.cluster_id,
                    "dist_to_core_km": c.dist_to_core_km,
                },
            }
        )
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=2), encoding="utf-8")


def write_csv(candidates: List[Candidate], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cand_id",
                "score",
                "density",
                "peak_relief_m",
                "mean_relief_m",
                "area_m2",
                "extent",
                "aspect",
                "prominence_m",
                "compactness",
                "solidity",
                "width_m",
                "height_m",
                "lon",
                "lat",
                "cluster_id",
                "dist_to_core_km",
            ]
        )
        for c in candidates:
            w.writerow(
                [
                    c.cand_id,
                    f"{c.score:.6f}",
                    f"{c.density:.6f}",
                    f"{c.peak_relief_m:.4f}",
                    f"{c.mean_relief_m:.4f}",
                    f"{c.area_m2:.2f}",
                    f"{c.extent:.4f}",
                    f"{c.aspect:.3f}",
                    f"{c.prominence_m:.4f}",
                    f"{c.compactness:.4f}",
                    f"{c.solidity:.4f}",
                    f"{c.width_m:.2f}",
                    f"{c.height_m:.2f}",
                    f"{c.lon:.8f}",
                    f"{c.lat:.8f}",
                    c.cluster_id,
                    "" if c.dist_to_core_km is None else f"{c.dist_to_core_km:.4f}",
                ]
            )


def write_clusters_csv(candidates: List[Candidate], out_path: Path) -> None:
    clusters: Dict[int, List[Candidate]] = {}
    for c in candidates:
        if c.cluster_id == -1:
            continue
        clusters.setdefault(c.cluster_id, []).append(c)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "n", "mean_score", "mean_density", "centroid_lon", "centroid_lat"])
        for cid in sorted(clusters.keys()):
            pts = clusters[cid]
            n = len(pts)
            mean_score = float(np.mean([p.score for p in pts])) if n else 0.0
            mean_dens = float(np.mean([p.density for p in pts])) if n else 0.0
            clon = float(np.mean([p.lon for p in pts])) if n else 0.0
            clat = float(np.mean([p.lat for p in pts])) if n else 0.0
            w.writerow([cid, n, f"{mean_score:.6f}", f"{mean_dens:.6f}", f"{clon:.8f}", f"{clat:.8f}"])


def _kml_escape(s: str) -> str:
    return s.replace("&", "and").replace("<", "(").replace(">", ")")


def write_kml(candidates: List[Candidate], out_path: Path, label_top_n: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_c = sorted(candidates, key=lambda c: c.score, reverse=True)
    top = sorted_c[: max(0, int(label_top_n))]
    rest = sorted_c[max(0, int(label_top_n)) :]

    kml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        "<name>MayaScan Candidates</name>",
        '<Style id="topPin">',
        "<IconStyle><scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href></Icon></IconStyle>",
        "<LabelStyle><scale>1.0</scale></LabelStyle>",
        "</Style>",
        '<Style id="dot">',
        "<IconStyle><scale>0.35</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle>",
        "<LabelStyle><scale>0</scale></LabelStyle>",
        "</Style>",
    ]

    def placemark(c: Candidate, labeled: bool, style: str) -> List[str]:
        desc = (
            f"cand_id={c.cand_id}<br/>"
            f"score={c.score:.4f}<br/>"
            f"density={c.density:.4f}<br/>"
            f"peak_relief_m={c.peak_relief_m:.3f}<br/>"
            f"area_m2={c.area_m2:.0f}<br/>"
            f"extent={c.extent:.3f}<br/>"
            f"aspect={c.aspect:.2f}<br/>"
            f"prominence_m={c.prominence_m:.3f}<br/>"
            f"compactness={c.compactness:.3f}<br/>"
            f"solidity={c.solidity:.3f}<br/>"
            f"cluster_id={c.cluster_id}"
        )
        if c.dist_to_core_km is not None:
            desc += f"<br/>dist_to_core_km={c.dist_to_core_km:.3f}"

        name = f"Candidate {c.cand_id} (score={c.score:.2f})" if labeled else ""

        return [
            "<Placemark>",
            f"<name>{_kml_escape(name)}</name>",
            f"<styleUrl>#{style}</styleUrl>",
            f"<description>{desc}</description>",
            f"<Point><coordinates>{c.lon:.8f},{c.lat:.8f},0</coordinates></Point>",
            "</Placemark>",
        ]

    kml.append("<Folder><name>Top Ranked (labeled)</name>")
    for c in top:
        kml.extend(placemark(c, labeled=True, style="topPin"))
    kml.append("</Folder>")

    kml.append("<Folder><name>All Candidates (unlabeled)</name>")
    for c in rest:
        kml.extend(placemark(c, labeled=False, style="dot"))
    kml.append("</Folder>")

    kml.append("</Document></kml>")
    out_path.write_text("\n".join(kml), encoding="utf-8")


# -----------------------------
# Plots + report
# -----------------------------
def make_plots(out_dir: Path, lrm: np.ndarray, density: np.ndarray, candidates: List[Candidate]) -> None:
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(density, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title("Settlement Density (normalized)")
    plt.colorbar()
    p = plots_dir / "density.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote density PNG: %s", p)

    plt.figure(figsize=(10, 8))
    # robust clip for display only
    lrm_vals = lrm[np.isfinite(lrm)]
    if lrm_vals.size:
        lo, hi = np.percentile(lrm_vals, [2, 98])
    else:
        lo, hi = -1.0, 1.0
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        lo, hi = -1.0, 1.0
    lrm_show = np.clip(np.where(np.isfinite(lrm), lrm, 0.0), lo, hi)
    plt.imshow(lrm_show, cmap="gray", vmin=lo, vmax=hi)
    xs = [c.px_x for c in candidates]
    ys = [c.px_y for c in candidates]
    plt.scatter(xs, ys, s=12)
    plt.title("Candidates overlay on combined LRM")
    p = plots_dir / "candidates_overlay.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote overlay PNG: %s", p)

    scores = np.array([c.score for c in candidates], dtype=float)
    peaks = np.array([c.peak_relief_m for c in candidates], dtype=float)
    prominence = np.array([c.prominence_m for c in candidates], dtype=float)
    areas = np.array([c.area_m2 for c in candidates], dtype=float)
    extents = np.array([c.extent for c in candidates], dtype=float)
    aspects = np.array([c.aspect for c in candidates], dtype=float)
    compactness = np.array([c.compactness for c in candidates], dtype=float)
    solidity = np.array([c.solidity for c in candidates], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.hist(scores[scores > 0], bins=30)
    plt.title("Score distribution (score>0)")
    plt.xlabel("score")
    plt.ylabel("count")
    p = plots_dir / "score_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(peaks, bins=30)
    plt.title("Peak relief distribution")
    plt.xlabel("peak relief (m)")
    plt.ylabel("count")
    p = plots_dir / "peak_relief_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(prominence, bins=30)
    plt.title("Local prominence distribution")
    plt.xlabel("prominence (m)")
    plt.ylabel("count")
    p = plots_dir / "prominence_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(areas, bins=30)
    plt.title("Area distribution")
    plt.xlabel("area (m^2)")
    plt.ylabel("count")
    p = plots_dir / "area_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(extents, bins=30, range=(0, 1))
    plt.title("Extent distribution (bbox fill)")
    plt.xlabel("extent (0..1)")
    plt.ylabel("count")
    p = plots_dir / "extent_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(aspects, bins=30)
    plt.title("Aspect ratio distribution (>=1)")
    plt.xlabel("aspect")
    plt.ylabel("count")
    p = plots_dir / "aspect_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(compactness, bins=30, range=(0, 1))
    plt.title("Compactness distribution (4*pi*A/P^2)")
    plt.xlabel("compactness (0..1)")
    plt.ylabel("count")
    p = plots_dir / "compactness_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(solidity, bins=30, range=(0, 1))
    plt.title("Solidity distribution (area / convex_hull_area)")
    plt.xlabel("solidity (0..1)")
    plt.ylabel("count")
    p = plots_dir / "solidity_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)


def write_report_md(
    out_dir: Path,
    run_name: str,
    input_path: Path,
    dtm_path: Path,
    lrm_path: Path,
    density_path: Path,
    candidates: List[Candidate],
    clusters_csv: Path,
    params: Params,
    pos_thresh: float,
    min_density: float,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sorted_c = sorted(candidates, key=lambda c: c.score, reverse=True)
    top = sorted_c[: params.report_top_n]

    n_clusters = len({c.cluster_id for c in candidates if c.cluster_id != -1})
    n_noise = sum(1 for c in candidates if c.cluster_id == -1)

    md: List[str] = []
    md.append(f"# MayaScan report: {run_name}")
    md.append("")
    md.append(f"- Timestamp: **{ts}**")
    md.append(f"- Input: `{input_path}`")
    md.append("")
    md.append("## Outputs")
    md.append(f"- DTM: `{dtm_path}`")
    md.append(f"- LRM: `{lrm_path}`")
    md.append(f"- Density raster: `{density_path}`")
    md.append(f"- Candidates: `candidates.csv`, `candidates.geojson`, `candidates.kml`")
    md.append(f"- Clusters: `{clusters_csv.name}`")
    md.append("- Run metadata: `run_params.json`")
    md.append(f"- Plots: `plots/`")
    md.append(f"- HTML: `report.html` + `html/img/`")
    md.append("")
    md.append("## Parameters used (key)")
    md.append(f"- pos_relief_threshold: **{pos_thresh:.4f} m** (spec: `{params.pos_relief_threshold_spec}`)")
    md.append(f"- min_region_pixels: **{params.min_region_pixels}**")
    md.append(f"- max_slope_deg: **{params.max_slope_deg:.1f}**")
    md.append(f"- density_sigma_pix: **{params.density_sigma_pix}**")
    md.append(f"- min_density: **{min_density:.4f}** (spec: `{params.min_density_spec}`)")
    md.append(
        f"- post-filters: min_peak={params.min_peak_relief_m:.2f}m, min_area={params.min_area_m2:.1f}m², "
        f"min_extent={params.min_extent:.2f}, max_aspect={params.max_aspect:.2f}, min_prominence={params.min_prominence_m:.2f}m, "
        f"min_compactness={params.min_compactness:.2f}, min_solidity={params.min_solidity:.2f}"
    )
    md.append(
        f"- score exponents: dens^{params.score_density_exp:.2f}, peak^{params.score_peak_exp:.2f}, "
        f"extent^{params.score_extent_exp:.2f}, prominence^{params.score_prominence_exp:.2f}, compactness^{params.score_compactness_exp:.2f}, "
        f"solidity^{params.score_solidity_exp:.2f}, area^{params.score_area_exp:.2f}"
    )
    md.append(f"- cluster_eps_mode: **{params.cluster_eps_mode}** (base={params.cluster_eps_m:.1f} m), min_samples: **{params.cluster_min_samples}**")
    md.append(f"- KML labeled top-N: **{params.kml_label_top_n}**")
    md.append("")
    md.append("## Summary")
    md.append(f"- Candidates detected: **{len(candidates)}**")
    md.append(f"- Clusters (DBSCAN): **{n_clusters}** (noise: {n_noise})")
    md.append("")
    md.append("## Top candidates")
    md.append("")
    md.append("| rank | cand_id | score | dens | peak(m) | prominence(m) | area(m²) | extent | aspect | compactness | solidity | cluster | lon | lat |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, c in enumerate(top, start=1):
        md.append(
            f"| {i} | {c.cand_id} | {c.score:.4f} | {c.density:.3f} | {c.peak_relief_m:.2f} | {c.prominence_m:.2f} | {c.area_m2:.0f} | "
            f"{c.extent:.2f} | {c.aspect:.2f} | {c.compactness:.2f} | {c.solidity:.2f} | {c.cluster_id} | {c.lon:.6f} | {c.lat:.6f} |"
        )
    md.append("")
    md.append("## Notes")
    md.append("- Extent = **area / bbox_area** (0..1). Higher generally means more coherent/filled region.")
    md.append("- Aspect = max(width/height, height/width). Very large aspect often means linear/noisy ridges.")
    md.append("- Local prominence = **region mean relief - surrounding ring mean relief**. Low values often indicate background trends.")
    md.append("- Compactness = **4πA/P²** (0..1). Lower values are more line-like and likely false positives.")
    md.append("- Solidity = **area / convex_hull_area** (0..1). Lower values are fragmented/irregular shapes.")
    md.append("- Slope filter uses **region slope q75** (not centroid slope).")
    md.append("- Candidate density uses **region mean density** over each connected region.")
    md.append("- Clustering/distances are done in **meters** (auto-UTM if source CRS is geographic).")
    md.append("- KML ‘All Candidates’ dots have label scale=0 to prevent Google Earth text overload.")
    md.append("")

    out_path = out_dir / "report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


def write_report_pdf(md_path: Path, pdf_path: Path) -> None:
    if canvas is None:
        LOG.warning("reportlab not installed; skipping PDF report.")
        return

    text = md_path.read_text(encoding="utf-8").splitlines()
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    x = 50
    y = height - 50
    c.setFont("Helvetica", 10)

    for line in text:
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 50
        c.drawString(x, y, line[:110])
        y -= 12
    c.save()


def update_manifest(runs_dir: Path, run_name: str, out_dir: Path, input_path: Path) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = runs_dir / "manifest.csv"
    new = not manifest.exists()

    row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "input": str(input_path),
        "out_dir": str(out_dir),
    }

    with manifest.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)

    LOG.info("Updated manifest: %s", manifest)


def write_run_params_json(
    out_dir: Path,
    run_name: str,
    input_path: Path,
    params: Params,
    pos_thresh: float,
    min_density: float,
    src_crs: CRS,
    clustering_crs: Optional[CRS],
    pdal_ver: str,
    dropped_density: int,
    dropped_post: int,
    candidate_count: int,
) -> Path:
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "input_path": str(input_path),
        "pdal_version": pdal_ver,
        "source_crs": src_crs.to_string(),
        "clustering_crs": None if clustering_crs is None else clustering_crs.to_string(),
        "resolved_thresholds": {
            "pos_relief_m": float(pos_thresh),
            "min_density": float(min_density),
        },
        "candidate_accounting": {
            "dropped_density": int(dropped_density),
            "dropped_post_filters": int(dropped_post),
            "kept_candidates": int(candidate_count),
        },
        "params": asdict(params),
    }
    out_path = out_dir / "run_params.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


# -----------------------------
# HTML cutouts + report
# -----------------------------
def _clamp_window(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0c = max(0, min(w, x0))
    y0c = max(0, min(h, y0))
    x1c = max(0, min(w, x1))
    y1c = max(0, min(h, y1))
    if x1c <= x0c:
        x1c = min(w, x0c + 1)
    if y1c <= y0c:
        y1c = min(h, y0c + 1)
    return x0c, y0c, x1c, y1c


def generate_candidate_panels(
    out_dir: Path,
    run_name: str,
    dtm: np.ndarray,
    lrm: np.ndarray,
    params: Params,
    candidates: List[Candidate],
    top_n: int,
    res_m: float,
) -> None:
    import matplotlib.pyplot as plt

    img_dir = out_dir / "html" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    hs = hillshade(dtm, res_m=res_m)

    sorted_c = sorted(candidates, key=lambda c: c.score, reverse=True)
    take = sorted_c[: max(0, int(top_n))]

    half_pix = int(round((params.cutout_size_m / res_m) / 2.0))
    LOG.info("Generating cutouts: size=%.1fm (~%d px), top_n=%d", params.cutout_size_m, half_pix * 2, len(take))

    H, W = lrm.shape[:2]

    for c in take:
        cx = int(round(c.px_x))
        cy = int(round(c.px_y))

        x0, y0, x1, y1 = _clamp_window(cx - half_pix, cy - half_pix, cx + half_pix, cy + half_pix, W, H)

        lrm_crop = lrm[y0:y1, x0:x1]
        hs_crop = hs[y0:y1, x0:x1]

        vals = lrm_crop[np.isfinite(lrm_crop)]
        if vals.size:
            lo, hi = np.percentile(vals, [2, 98])
        else:
            lo, hi = -1.0, 1.0
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            lo, hi = -1.0, 1.0
        lrm_show = np.clip(np.where(np.isfinite(lrm_crop), lrm_crop, 0.0), lo, hi)

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(lrm_show, cmap="gray", vmin=lo, vmax=hi)
        ax1.scatter([cx - x0], [cy - y0], s=18)
        ax1.set_title("LRM")
        ax1.set_axis_off()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(hs_crop, cmap="gray", vmin=0.0, vmax=1.0)
        ax2.scatter([cx - x0], [cy - y0], s=18)
        ax2.set_title("Hillshade")
        ax2.set_axis_off()

        fig.suptitle(
            f"{run_name} — cand {c.cand_id} | score {c.score:.3f} | dens {c.density:.3f} | "
            f"peak {c.peak_relief_m:.2f}m | prom {c.prominence_m:.2f}m | extent {c.extent:.2f}",
            fontsize=10,
        )

        fname = f"cand_{c.cand_id:04d}_panel.png"
        out_path = img_dir / fname
        fig.tight_layout()
        fig.savefig(out_path, dpi=int(params.cutout_dpi), bbox_inches="tight")
        plt.close(fig)

        c.img_relpath = f"html/img/{fname}"


def write_html_report(
    out_dir: Path,
    run_name: str,
    input_path: Path,
    candidates: List[Candidate],
    params: Params,
    pos_thresh: float,
    min_density: float,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not candidates:
        center_lat, center_lon = 0.0, 0.0
    else:
        center_lat = float(np.mean([c.lat for c in candidates]))
        center_lon = float(np.mean([c.lon for c in candidates]))

    sorted_c = sorted(candidates, key=lambda c: c.score, reverse=True)
    top = sorted_c[: params.report_top_n]

    points = []
    for c in sorted_c:
        points.append(
            {
                "cand_id": c.cand_id,
                "score": c.score,
                "density": c.density,
                "peak": c.peak_relief_m,
                "prominence": c.prominence_m,
                "area": c.area_m2,
                "extent": c.extent,
                "aspect": c.aspect,
                "compactness": c.compactness,
                "solidity": c.solidity,
                "cluster": c.cluster_id,
                "lat": c.lat,
                "lon": c.lon,
                "img": c.img_relpath or "",
            }
        )

    html_path = out_dir / "report.html"
    s_points = json.dumps(points)

    doc = f"""<!doctype html>
<html><head><meta charset='utf-8'/>
<title>MayaScan Report — {html.escape(run_name)}</title>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>
<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>

<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 18px; color: #111; }}
h1 {{ margin: 0 0 8px 0; }}
.small {{ color: #444; }}
#map {{ height: 520px; border: 1px solid #ddd; border-radius: 10px; margin: 14px 0; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.card {{ border: 1px solid #e5e5e5; border-radius: 12px; padding: 12px; background: #fff; }}
.hr {{ border-top: 1px solid #eee; margin: 18px 0; }}
a {{ color: #0b63ce; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 10px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px; border-bottom: 1px solid #eee; font-size: 14px; }}
th {{ text-align: left; background: #fafafa; position: sticky; top: 0; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f2f2f2; font-size: 12px; }}
.topnote {{ margin-top: 8px; }}
</style>

</head><body>
<h1>MayaScan Report — {html.escape(run_name)}</h1>
<div class='small'>Timestamp: <b>{ts}</b> &nbsp;|&nbsp; Input: <code>{html.escape(str(input_path))}</code></div>
<div class='topnote small'>
pos_relief_threshold: <b>{pos_thresh:.4f} m</b> ({html.escape(params.pos_relief_threshold_spec)}) &nbsp;|&nbsp;
min_density: <b>{min_density:.4f}</b> ({html.escape(params.min_density_spec)}) &nbsp;|&nbsp;
candidates: <b>{len(candidates)}</b> &nbsp;|&nbsp; KML labels: top <b>{params.kml_label_top_n}</b>
</div>

<div id='map'></div>

<div class='grid'>
  <div class='card'>
    <h3 style='margin-top:0'>How to triage</h3>
    <ol class='small'>
      <li>Scan the map for clusters / linear terrace patterns.</li>
      <li>Click a marker: popups show score + quick cutout if available.</li>
      <li>Open high scorers in Google Maps / Google Earth (KML) and compare LRM texture.</li>
      <li>Favor coherent shapes (rectilinear platforms, aligned mounds, terrace lines).</li>
    </ol>
  </div>
  <div class='card'>
    <h3 style='margin-top:0'>Files</h3>
    <ul class='small'>
      <li><code>candidates.csv</code>, <code>candidates.geojson</code>, <code>candidates.kml</code></li>
      <li><code>dtm.tif</code>, <code>lrm.tif</code>, <code>mound_density.tif</code></li>
      <li><code>run_params.json</code> (resolved settings + thresholds)</li>
      <li><code>plots/</code> (density, overlay, histograms)</li>
      <li><code>html/img/</code> (candidate cutouts)</li>
    </ul>
  </div>
</div>

<div class='hr'></div>
<h2>Top candidates</h2>
<div class='small'>Click coordinates to open in Google Maps. Images show LRM + hillshade panel (when generated).</div>
<div class='hr'></div>
"""

    for rank, c in enumerate(top, start=1):
        gmaps = f"https://www.google.com/maps?q={c.lat:.8f},{c.lon:.8f}"
        img_tag = ""
        if c.img_relpath:
            img_tag = f"<img src='{html.escape(c.img_relpath)}' alt='candidate {c.cand_id} cutout'/>"
        doc += f"""
<h3>Candidate {c.cand_id} <span class='badge'>rank {rank}</span> — score {c.score:.3f}</h3>
<p><b>dens</b> {c.density:.3f} | <b>peak</b> {c.peak_relief_m:.2f} m | <b>prom</b> {c.prominence_m:.2f} m | <b>area</b> {c.area_m2:.0f} m² |
<b>extent</b> {c.extent:.2f} | <b>aspect</b> {c.aspect:.2f} | <b>compactness</b> {c.compactness:.2f} | <b>solidity</b> {c.solidity:.2f} | <b>cluster</b> {c.cluster_id}</p>
<p><a href='{gmaps}' target='_blank'>{c.lat:.6f}, {c.lon:.6f}</a></p>
{img_tag}
<div class='hr'></div>
"""

    doc += """
<h2>All candidates</h2>
<div class='small'>Sorted by score (descending). Map includes all points.</div>
<div class='card' style='max-height:520px; overflow:auto;'>
<table>
<thead>
<tr>
<th>rank</th><th>cand_id</th><th>score</th><th>dens</th><th>peak(m)</th><th>prom(m)</th><th>area(m²)</th><th>extent</th><th>aspect</th><th>compact</th><th>solidity</th><th>cluster</th><th>lat</th><th>lon</th>
</tr>
</thead>
<tbody>
"""
    for rank, c in enumerate(sorted_c, start=1):
        gmaps = f"https://www.google.com/maps?q={c.lat:.8f},{c.lon:.8f}"
        doc += (
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{c.cand_id}</td>"
            f"<td>{c.score:.3f}</td>"
            f"<td>{c.density:.3f}</td>"
            f"<td>{c.peak_relief_m:.2f}</td>"
            f"<td>{c.prominence_m:.2f}</td>"
            f"<td>{c.area_m2:.0f}</td>"
            f"<td>{c.extent:.2f}</td>"
            f"<td>{c.aspect:.2f}</td>"
            f"<td>{c.compactness:.2f}</td>"
            f"<td>{c.solidity:.2f}</td>"
            f"<td>{c.cluster_id}</td>"
            f"<td><a href='{gmaps}' target='_blank'>{c.lat:.6f}</a></td>"
            f"<td><a href='{gmaps}' target='_blank'>{c.lon:.6f}</a></td>"
            "</tr>\n"
        )

    doc += f"""
</tbody></table></div>

<script>
const points = {s_points};

const map = L.map('map').setView([{center_lat:.8f}, {center_lon:.8f}], 14);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

function radiusFromScore(score) {{
  const r = 4 + Math.min(18, Math.sqrt(Math.max(0, score)) * 3.0);
  return r;
}}

function colorFromCluster(cid) {{
  if (cid === -1) return '#555';
  const palette = ['#b91c1c','#1d4ed8','#047857','#7c3aed','#c2410c','#0f766e','#a21caf','#4338ca'];
  return palette[(cid-1) % palette.length];
}}

const bounds = [];
points.forEach(p => {{
  const r = radiusFromScore(p.score);
  const col = colorFromCluster(p.cluster);
  const gmaps = `https://www.google.com/maps?q=${{p.lat}},${{p.lon}}`;

  let imgHtml = '';
  if (p.img && p.img.length > 0) {{
    imgHtml = `<div style="margin-top:8px"><img src="${{p.img}}" style="max-width:260px; border:1px solid #ddd; border-radius:8px"/></div>`;
  }}

  const popup = `
    <div style="font-size:14px">
      <b>Candidate ${{p.cand_id}}</b><br/>
      score <b>${{p.score.toFixed(3)}}</b> | dens ${{p.density.toFixed(3)}}<br/>
      peak ${{p.peak.toFixed(2)}}m | prom ${{p.prominence.toFixed(2)}}m | area ${{Math.round(p.area)}} m²<br/>
      extent ${{p.extent.toFixed(2)}} | aspect ${{p.aspect.toFixed(2)}}<br/>
      compactness ${{p.compactness.toFixed(2)}} | solidity ${{p.solidity.toFixed(2)}}<br/>
      cluster ${{p.cluster}}<br/>
      <a href="${{gmaps}}" target="_blank">Open in Google Maps</a>
      ${{imgHtml}}
    </div>
  `;

  const marker = L.circleMarker([p.lat, p.lon], {{
    radius: r,
    color: col,
    weight: 2,
    fillColor: col,
    fillOpacity: 0.55
  }}).addTo(map);
  marker.bindPopup(popup);
  bounds.push([p.lat, p.lon]);
}});

if (bounds.length > 0) {{
  map.fitBounds(bounds, {{padding:[20,20]}});
}}
</script>

</body></html>
"""
    html_path.write_text(doc, encoding="utf-8")
    return html_path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="MayaScan: LiDAR archaeology discovery pipeline")

    ap.add_argument("-i", "--input", required=True, help="Input LAZ/LAS file")
    ap.add_argument("--name", required=True, help="Run name (folder under runs/)")
    ap.add_argument("--runs-dir", default="runs", help="Runs base directory")
    ap.add_argument("--overwrite", action="store_true", help="Allow deleting an existing run folder")

    ap.add_argument("--try-smrf", action="store_true", help="Try PDAL SMRF ground classification before DTM")

    # knobs
    ap.add_argument("--pos-thresh", type=_arg_pos_thresh_spec, default=None, help="Override pos relief threshold (e.g. 0.20 or auto:p96)")
    ap.add_argument("--min-density", type=_arg_min_density_spec, default=None, help="Override min density threshold (e.g. 0.10 or auto:p60)")
    ap.add_argument("--density-sigma", type=_arg_positive_float, default=None, help="Override density sigma (pixels)")
    ap.add_argument("--max-slope-deg", type=_arg_nonnegative_float, default=None, help="Max allowed region slope q75 in degrees (default 20)")

    # post-filters
    ap.add_argument("--min-peak", type=_arg_nonnegative_float, default=None, help="Min peak relief (m) post-filter (e.g. 0.5)")
    ap.add_argument("--min-area-m2", type=_arg_nonnegative_float, default=None, help="Min area (m^2) post-filter (e.g. 30)")
    ap.add_argument("--min-extent", type=_arg_unit_interval, default=None, help="Min extent (0..1) post-filter (e.g. 0.35)")
    ap.add_argument("--max-aspect", type=_arg_ge_one_float, default=None, help="Max aspect ratio post-filter (e.g. 4.0)")
    ap.add_argument("--prominence-ring-pix", type=_arg_positive_int, default=None, help="Ring width (pixels) for local prominence estimate (default 6)")
    ap.add_argument("--min-prominence", type=_arg_nonnegative_float, default=None, help="Min local prominence (m) post-filter (default 0.10)")
    ap.add_argument("--min-compactness", type=_arg_unit_interval, default=None, help="Min compactness 4*pi*A/P^2 (0..1), lower removes line-like shapes")
    ap.add_argument("--min-solidity", type=_arg_unit_interval, default=None, help="Min solidity area/convex_hull_area (0..1), lower removes fragmented/linear shapes")

    # scoring knobs
    ap.add_argument("--score-extent-exp", type=_arg_nonnegative_float, default=None, help="Exponent for extent in score (default 0.35)")
    ap.add_argument("--score-prominence-exp", type=_arg_nonnegative_float, default=None, help="Exponent for prominence in score (default 0.75)")
    ap.add_argument("--score-compactness-exp", type=_arg_nonnegative_float, default=None, help="Exponent for compactness in score (default 0.20)")
    ap.add_argument("--score-solidity-exp", type=_arg_nonnegative_float, default=None, help="Exponent for solidity in score (default 0.20)")
    ap.add_argument("--score-area-exp", type=_arg_nonnegative_float, default=None, help="Exponent for area_m2 in score (default 0.50)")

    # clustering knobs
    ap.add_argument("--cluster-eps", type=_arg_cluster_eps_spec, default=None, help="DBSCAN eps in meters or 'auto' (default auto)")
    ap.add_argument("--min-samples", type=_arg_positive_int, default=None, help="DBSCAN min_samples (default 3)")

    ap.add_argument("--label-top-n", type=_arg_nonnegative_int, default=None, help="Override KML labeled top-N")
    ap.add_argument("--report-top-n", type=_arg_nonnegative_int, default=None, help="Override report top-N table size")

    # HTML / cutouts
    ap.add_argument("--no-html", action="store_true", help="Disable HTML report + cutout images")
    ap.add_argument("--cutout-size-m", type=_arg_positive_float, default=None, help="Cutout panel window size in meters (default 140)")
    ap.add_argument("--cutout-top-n", type=_arg_nonnegative_int, default=None, help="How many top candidates get cutouts (default report_top_n)")

    args = ap.parse_args()

    try:
        run_name = sanitize_run_name(args.name)
    except ValueError as exc:
        ap.error(str(exc))
    if run_name != args.name:
        print(f"Run name sanitized: '{args.name}' -> '{run_name}'", file=sys.stderr)

    params = Params()

    if args.pos_thresh is not None:
        params.pos_relief_threshold_spec = args.pos_thresh
    if args.min_density is not None:
        params.min_density_spec = args.min_density
    if args.density_sigma is not None:
        params.density_sigma_pix = args.density_sigma
    if args.max_slope_deg is not None:
        params.max_slope_deg = args.max_slope_deg

    if args.min_peak is not None:
        params.min_peak_relief_m = args.min_peak
    if args.min_area_m2 is not None:
        params.min_area_m2 = args.min_area_m2
    if args.min_extent is not None:
        params.min_extent = args.min_extent
    if args.max_aspect is not None:
        params.max_aspect = args.max_aspect
    if args.prominence_ring_pix is not None:
        params.prominence_ring_pixels = args.prominence_ring_pix
    if args.min_prominence is not None:
        params.min_prominence_m = args.min_prominence
    if args.min_compactness is not None:
        params.min_compactness = args.min_compactness
    if args.min_solidity is not None:
        params.min_solidity = args.min_solidity

    if args.score_extent_exp is not None:
        params.score_extent_exp = args.score_extent_exp
    if args.score_prominence_exp is not None:
        params.score_prominence_exp = args.score_prominence_exp
    if args.score_compactness_exp is not None:
        params.score_compactness_exp = args.score_compactness_exp
    if args.score_solidity_exp is not None:
        params.score_solidity_exp = args.score_solidity_exp
    if args.score_area_exp is not None:
        params.score_area_exp = args.score_area_exp

    if args.cluster_eps is not None:
        if args.cluster_eps == "auto":
            params.cluster_eps_mode = "auto"
        else:
            params.cluster_eps_mode = "fixed"
            params.cluster_eps_m = float(args.cluster_eps)

    if args.min_samples is not None:
        params.cluster_min_samples = args.min_samples

    if args.label_top_n is not None:
        params.kml_label_top_n = args.label_top_n
    if args.report_top_n is not None:
        params.report_top_n = args.report_top_n

    if args.no_html:
        params.html_report = False
    if args.cutout_size_m is not None:
        params.cutout_size_m = args.cutout_size_m

    input_path = Path(args.input).expanduser().resolve()
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    out_dir = (runs_dir / run_name).resolve()
    ensure_path_within(runs_dir, out_dir)

    if out_dir.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Run dir already exists: {out_dir}\n"
                f"Use --overwrite if you want to delete/recreate it."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir)
    pdal_ver = pdal_version()
    LOG.info("PDAL detected: %s", pdal_ver)

    dtm_path = out_dir / "dtm.tif"
    lrm_path = out_dir / "lrm.tif"
    density_path = out_dir / "mound_density.tif"
    geojson_path = out_dir / "candidates.geojson"
    kml_path = out_dir / "candidates.kml"
    csv_path = out_dir / "candidates.csv"
    clusters_csv = out_dir / "clusters.csv"
    report_pdf = out_dir / "report.pdf"
    html_report_path = out_dir / "report.html"

    tmp_dir = out_dir / "_tmp"

    LOG.info("Step 0: Building DTM from LAZ/LAS")
    build_dtm_from_laz(
        laz_path=input_path,
        out_dtm_tif=dtm_path,
        tmp_dir=tmp_dir,
        resolution_m=params.dtm_resolution_m,
        try_smrf=bool(args.try_smrf),
    )
    LOG.info("DTM written: %s", dtm_path)

    LOG.info("Step 1: Building multi-scale LRM")
    dtm, dtm_prof = load_raster(dtm_path)
    dtm_transform = dtm_prof["transform"]
    res_m = _res_m_from_profile(dtm_prof)

    slope_deg = compute_slope_degrees(dtm, res_m=float(res_m))
    lrm = build_multiscale_lrm(dtm, params)
    write_float_geotiff(lrm_path, lrm, dtm_prof)
    LOG.info("LRM written: %s", lrm_path)

    LOG.info("Step 2: Detecting candidate structures")
    regions, density_norm, pos_thresh, min_density = detect_candidates(
        lrm=lrm,
        dtm_slope_deg=slope_deg,
        profile=dtm_prof,
        params=params,
        out_density_tif=density_path,
    )

    crs_any = dtm_prof.get("crs")
    if crs_any is None:
        raise RuntimeError("DTM has no CRS; cannot export lon/lat")
    src_crs = CRS.from_user_input(crs_any)
    transformer_ll = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    # Candidate build + filters
    candidates: List[Candidate] = []
    dropped_density = 0
    dropped_post = 0

    cand_id = 1
    for r in regions:
        dens = float(r.get("density_mean", np.nan))
        if not np.isfinite(dens):
            dropped_density += 1
            continue
        if dens < float(min_density):
            dropped_density += 1
            continue

        peak = float(r["peak"])
        area_m2 = float(r["area_m2"])
        extent = float(r["extent"])
        aspect = float(r["aspect"])
        prominence = float(r.get("prominence_m", 0.0))
        compactness = float(r.get("compactness", 0.0))
        solidity = float(r.get("solidity", 0.0))

        # post-filters for “project goal”
        if (
            peak < params.min_peak_relief_m
            or area_m2 < params.min_area_m2
            or extent < params.min_extent
            or aspect > params.max_aspect
            or prominence < params.min_prominence_m
            or compactness < params.min_compactness
            or solidity < params.min_solidity
        ):
            dropped_post += 1
            continue

        score = (
            (dens ** params.score_density_exp)
            * (max(1e-9, peak) ** params.score_peak_exp)
            * ((max(1e-6, extent)) ** params.score_extent_exp)
            * ((max(1e-6, prominence)) ** params.score_prominence_exp)
            * ((max(1e-6, compactness)) ** params.score_compactness_exp)
            * ((max(1e-6, solidity)) ** params.score_solidity_exp)
            * (max(1e-9, area_m2) ** params.score_area_exp)
        )

        x_map, y_map = pix2map_xy(dtm_transform, r["cy"], r["cx"])
        lon, lat = transformer_ll.transform(x_map, y_map)

        candidates.append(
            Candidate(
                cand_id=cand_id,
                px_x=float(r["cx"]),
                px_y=float(r["cy"]),
                peak_relief_m=peak,
                mean_relief_m=float(r["mean"]),
                area_m2=area_m2,
                density=dens,
                extent=extent,
                aspect=aspect,
                prominence_m=prominence,
                compactness=compactness,
                solidity=solidity,
                width_m=float(r["width_m"]),
                height_m=float(r["height_m"]),
                score=float(score),
                lon=float(lon),
                lat=float(lat),
            )
        )
        cand_id += 1

    LOG.info("Dropped by density (region mean < min_density): %d", dropped_density)
    LOG.info("Dropped by post-filters: %d", dropped_post)
    LOG.info("Kept candidates after density + post-filters: %d", len(candidates))

    LOG.info("Step 3: Clustering + settlement pattern analysis (meters)")
    used_m_crs: Optional[CRS] = None
    if candidates:
        xs = []
        ys = []
        for c in candidates:
            x_map, y_map = pix2map_xy(dtm_transform, c.px_y, c.px_x)
            xs.append(float(x_map))
            ys.append(float(y_map))
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)

        xs_m, ys_m, used_m_crs = project_points_to_meters(src_crs, xs, ys)
        LOG.info("Clustering CRS: %s", used_m_crs.to_string())

        labels = cluster_candidates_meters(xs_m, ys_m, params)
        for c, lab in zip(candidates, labels):
            c.cluster_id = int(lab)

        core_idx = int(np.argmax([c.density for c in candidates]))
        core_x, core_y = xs_m[core_idx], ys_m[core_idx]
        for i, c in enumerate(candidates):
            dx = xs_m[i] - core_x
            dy = ys_m[i] - core_y
            c.dist_to_core_km = float(math.sqrt(dx * dx + dy * dy) / 1000.0)

        n_clusters = len({c.cluster_id for c in candidates if c.cluster_id != -1})
        LOG.info("Clusters found: %d (noise=%d)", n_clusters, sum(1 for c in candidates if c.cluster_id == -1))

    LOG.info("Step 4: Exporting GIS products")
    write_geojson(candidates, geojson_path)
    LOG.info("Wrote GeoJSON: %s", geojson_path)

    write_kml(candidates, kml_path, label_top_n=params.kml_label_top_n)
    LOG.info("Wrote KML: %s", kml_path)

    write_csv(candidates, csv_path)
    LOG.info("Wrote CSV: %s", csv_path)

    write_clusters_csv(candidates, clusters_csv)
    LOG.info("Wrote clusters CSV: %s", clusters_csv)

    LOG.info("Step 5: Writing plots")
    make_plots(out_dir, lrm, density_norm, candidates)

    LOG.info("Step 6: Writing reports")
    md_path = write_report_md(
        out_dir=out_dir,
        run_name=run_name,
        input_path=input_path,
        dtm_path=dtm_path,
        lrm_path=lrm_path,
        density_path=density_path,
        candidates=candidates,
        clusters_csv=clusters_csv,
        params=params,
        pos_thresh=pos_thresh,
        min_density=min_density,
    )
    LOG.info("Wrote report.md: %s", md_path)

    write_report_pdf(md_path, report_pdf)
    if report_pdf.exists():
        LOG.info("Wrote report.pdf: %s", report_pdf)

    params_json = write_run_params_json(
        out_dir=out_dir,
        run_name=run_name,
        input_path=input_path,
        params=params,
        pos_thresh=pos_thresh,
        min_density=min_density,
        src_crs=src_crs,
        clustering_crs=used_m_crs,
        pdal_ver=pdal_ver,
        dropped_density=dropped_density,
        dropped_post=dropped_post,
        candidate_count=len(candidates),
    )
    LOG.info("Wrote run_params.json: %s", params_json)

    if params.html_report and candidates:
        cutout_top_n = params.report_top_n if args.cutout_top_n is None else int(args.cutout_top_n)
        LOG.info("Step 7: Generating HTML report + cutouts")
        generate_candidate_panels(
            out_dir=out_dir,
            run_name=run_name,
            dtm=dtm,
            lrm=lrm,
            params=params,
            candidates=candidates,
            top_n=cutout_top_n,
            res_m=float(res_m),
        )
        html_out = write_html_report(
            out_dir=out_dir,
            run_name=run_name,
            input_path=input_path,
            candidates=candidates,
            params=params,
            pos_thresh=pos_thresh,
            min_density=min_density,
        )
        LOG.info("Wrote report.html: %s", html_out)

    update_manifest(runs_dir, run_name, out_dir, input_path)

    LOG.info("DONE. Output folder: %s", out_dir)
    LOG.info("Quick open: %s", kml_path)
    if params.html_report:
        LOG.info("HTML report: %s", html_report_path)


if __name__ == "__main__":
    main()
