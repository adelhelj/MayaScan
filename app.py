#!/usr/bin/env python3
import base64
import json
import re
import sys
import time
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = REPO_ROOT / "maya_scan.py"
LOGO_PATH = REPO_ROOT / "assets" / "mayascan_logo.svg"

PRESET_BALANCED = "Balanced (Recommended)"
PRESET_STRICT = "Strict (Low false positives)"
PRESET_EXPLORATORY = "Exploratory (High recall)"

PRESET_VALUES: dict[str, dict[str, object]] = {
    PRESET_BALANCED: {
        "cfg_pos_thresh": "auto:p96",
        "cfg_min_density": "auto:p60",
        "cfg_density_sigma": 40.0,
        "cfg_max_slope_deg": 20.0,
        "cfg_min_peak": 0.50,
        "cfg_min_area_m2": 25.0,
        "cfg_min_extent": 0.38,
        "cfg_max_aspect": 3.5,
        "cfg_min_compactness": 0.12,
        "cfg_min_solidity": 0.50,
        "cfg_cluster_eps": "auto",
        "cfg_min_samples": 4,
        "cfg_report_top_n": 30,
        "cfg_label_top_n": 60,
    },
    PRESET_STRICT: {
        "cfg_pos_thresh": "auto:p97",
        "cfg_min_density": "auto:p65",
        "cfg_density_sigma": 45.0,
        "cfg_max_slope_deg": 18.0,
        "cfg_min_peak": 0.60,
        "cfg_min_area_m2": 35.0,
        "cfg_min_extent": 0.42,
        "cfg_max_aspect": 3.0,
        "cfg_min_compactness": 0.16,
        "cfg_min_solidity": 0.58,
        "cfg_cluster_eps": "auto",
        "cfg_min_samples": 5,
        "cfg_report_top_n": 30,
        "cfg_label_top_n": 60,
    },
    PRESET_EXPLORATORY: {
        "cfg_pos_thresh": "auto:p95",
        "cfg_min_density": "auto:p50",
        "cfg_density_sigma": 35.0,
        "cfg_max_slope_deg": 22.0,
        "cfg_min_peak": 0.35,
        "cfg_min_area_m2": 15.0,
        "cfg_min_extent": 0.30,
        "cfg_max_aspect": 4.5,
        "cfg_min_compactness": 0.08,
        "cfg_min_solidity": 0.40,
        "cfg_cluster_eps": "auto",
        "cfg_min_samples": 3,
        "cfg_report_top_n": 30,
        "cfg_label_top_n": 60,
    },
}


def apply_preset_to_session(preset_name: str, *, update_selector: bool = True) -> None:
    values = PRESET_VALUES.get(preset_name, PRESET_VALUES[PRESET_BALANCED])
    for key, val in values.items():
        st.session_state[key] = val
    if update_selector:
        st.session_state["cfg_preset"] = preset_name


def preset_matches_session(preset_name: str) -> bool:
    values = PRESET_VALUES.get(preset_name, PRESET_VALUES[PRESET_BALANCED])
    for key, val in values.items():
        cur = st.session_state.get(key)
        if isinstance(val, float):
            try:
                if abs(float(cur) - val) > 1e-9:
                    return False
            except Exception:
                return False
        else:
            if str(cur) != str(val):
                return False
    return True


def current_ui_config_snapshot() -> dict[str, object]:
    keys = [
        "cfg_pos_thresh",
        "cfg_min_density",
        "cfg_density_sigma",
        "cfg_max_slope_deg",
        "cfg_min_peak",
        "cfg_min_area_m2",
        "cfg_min_extent",
        "cfg_max_aspect",
        "cfg_min_compactness",
        "cfg_min_solidity",
        "cfg_cluster_eps",
        "cfg_min_samples",
        "cfg_report_top_n",
        "cfg_label_top_n",
    ]
    out: dict[str, object] = {}
    for key in keys:
        out[key] = st.session_state.get(key)
    return out


def preset_slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "preset"


def read_run_summary(run_dir: Path) -> dict[str, object]:
    out: dict[str, object] = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "candidates": None,
        "clusters": None,
        "noise": None,
        "top_score": None,
        "mean_score": None,
        "median_score": None,
    }

    vals = parse_values_used(read_text_safely(run_dir / "process.log"))
    if "candidates_kept" in vals:
        out["candidates"] = vals.get("candidates_kept")
    if "clusters_found" in vals:
        out["clusters"] = vals.get("clusters_found")
    if "clusters_noise" in vals:
        out["noise"] = vals.get("clusters_noise")

    csv_path = run_dir / "candidates.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if "score" in df.columns:
                s = pd.to_numeric(df["score"], errors="coerce").dropna()
                if not s.empty:
                    out["top_score"] = float(s.max())
                    out["mean_score"] = float(s.mean())
                    out["median_score"] = float(s.median())
            if out["candidates"] is None:
                out["candidates"] = int(len(df))
        except Exception:
            pass
    return out


def write_preset_compare_artifacts(
    runs_dir: Path,
    base_name: str,
    compare_payload: dict[str, object],
) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{base_name}_preset_compare_{stamp}"
    json_path = runs_dir / f"{stem}.json"
    md_path = runs_dir / f"{stem}.md"

    json_path.write_text(json.dumps(compare_payload, indent=2), encoding="utf-8")

    runs = compare_payload.get("runs", [])
    baseline = compare_payload.get("baseline_preset", "")
    lines: list[str] = []
    lines.append(f"# MayaScan Preset Comparison: {base_name}")
    lines.append("")
    lines.append(f"- Timestamp: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")
    lines.append(f"- Baseline preset: **{baseline}**")
    lines.append("")
    lines.append("| preset | run_name | candidates | clusters | noise | top_score | mean_score | median_score | d_candidates | d_top_score |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    if isinstance(runs, list):
        for r in runs:
            if not isinstance(r, dict):
                continue
            lines.append(
                f"| {r.get('preset','')} | {r.get('run_name','')} | {r.get('candidates','')} | {r.get('clusters','')} | "
                f"{r.get('noise','')} | {r.get('top_score','')} | {r.get('mean_score','')} | {r.get('median_score','')} | "
                f"{r.get('d_candidates_vs_baseline','')} | {r.get('d_top_score_vs_baseline','')} |"
            )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def resolve_input_path(mode: str, uploaded, input_local: str) -> Path | None:
    if mode == "Upload .laz/.las":
        if uploaded is None:
            st.error("Please upload a .laz or .las file.")
            return None
        data_dir = REPO_ROOT / "data" / "lidar"
        data_dir.mkdir(parents=True, exist_ok=True)
        uploaded_safe = sanitize_filename(uploaded.name)
        upload_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = data_dir / f"{upload_stamp}_{uploaded_safe}"
        input_path.write_bytes(uploaded.getvalue())
        return input_path

    input_path = Path(input_local).expanduser()
    if not input_path.is_absolute():
        input_path = (REPO_ROOT / input_path).resolve()
    if not input_path.exists():
        st.error(f"Input file not found: {input_path}")
        return None
    return input_path


# -----------------------------
# CLI command builder
# -----------------------------
def build_cmd(
    input_path: Path,
    run_name: str,
    runs_dir: Path,
    overwrite: bool,
    try_smrf: bool,
    pos_thresh: str,
    min_density: str,
    density_sigma: float,
    max_slope_deg: float,
    min_peak: float,
    min_area_m2: float,
    min_extent: float,
    max_aspect: float,
    min_compactness: float,
    min_solidity: float,
    cluster_eps: str,
    min_samples: int,
    report_top_n: int,
    label_top_n: int,
    no_html: bool,
):
    cmd = [sys.executable, str(SCRIPT_PATH)]
    cmd += ["-i", str(input_path)]
    cmd += ["--name", run_name]
    cmd += ["--runs-dir", str(runs_dir)]

    if overwrite:
        cmd += ["--overwrite"]
    if try_smrf:
        cmd += ["--try-smrf"]
    if no_html:
        cmd += ["--no-html"]

    if pos_thresh.strip():
        cmd += ["--pos-thresh", pos_thresh.strip()]
    if min_density.strip():
        cmd += ["--min-density", min_density.strip()]
    if density_sigma is not None:
        cmd += ["--density-sigma", str(density_sigma)]
    if max_slope_deg is not None:
        cmd += ["--max-slope-deg", str(max_slope_deg)]

    cmd += ["--min-peak", str(min_peak)]
    cmd += ["--min-area-m2", str(min_area_m2)]
    cmd += ["--min-extent", str(min_extent)]
    cmd += ["--max-aspect", str(max_aspect)]
    cmd += ["--min-compactness", str(min_compactness)]
    cmd += ["--min-solidity", str(min_solidity)]

    if cluster_eps.strip():
        cmd += ["--cluster-eps", cluster_eps.strip()]
    cmd += ["--min-samples", str(min_samples)]

    cmd += ["--report-top-n", str(report_top_n)]
    cmd += ["--label-top-n", str(label_top_n)]

    return cmd


# -----------------------------
# Utilities
# -----------------------------
def resolve_runs_dir(runs_dir_value: str) -> Path:
    p = Path(runs_dir_value).expanduser()
    return (REPO_ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def sanitize_run_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return cleaned[:120]


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)
    cleaned = cleaned.strip("._-")
    return cleaned or "input.laz"


def list_existing_runs(runs_dir_path: Path) -> list[str]:
    if not runs_dir_path.exists():
        return []
    try:
        runs = [p for p in runs_dir_path.iterdir() if p.is_dir()]
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.name for p in runs]
    except Exception:
        return []


def summarize_run_option(run_dir: Path) -> str:
    name = run_dir.name
    try:
        ts = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        ts = "unknown time"

    candidates = "?"
    process_log = run_dir / "process.log"
    if process_log.exists():
        vals = parse_values_used(read_text_safely(process_log))
        if "candidates_kept" in vals:
            candidates = str(vals["candidates_kept"])

    input_name = "input unknown"
    report_md = run_dir / "report.md"
    if report_md.exists():
        m = re.search(r"- Input:\s*`([^`]+)`", read_text_safely(report_md))
        if m:
            input_name = Path(m.group(1)).name

    return f"{name} | {ts} | cands: {candidates} | {input_name}"


def parse_cmd_settings(cmd: list[str]) -> dict:
    out: dict[str, object] = {}
    if not cmd:
        return out
    boolean_flags = {"--overwrite", "--try-smrf", "--no-html"}

    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("--"):
            key = tok.lstrip("-")
            if tok in boolean_flags:
                out[key] = True
            elif i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                out[key] = cmd[i + 1]
                i += 1
        i += 1
    return out


def parse_step_from_log_line(line: str) -> tuple[int, str] | None:
    m = re.search(r"Step\s+([0-9]+):\s*(.+)$", line)
    if not m:
        return None
    try:
        step_idx = int(m.group(1))
    except Exception:
        return None
    return step_idx, m.group(2).strip()


def validate_auto_or_numeric(
    value: str,
    *,
    field_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[bool, str | None]:
    s = value.strip()
    if not s:
        return False, f"{field_name}: value is required."

    m = re.fullmatch(r"auto:p([0-9]+(?:\.[0-9]+)?)", s, flags=re.IGNORECASE)
    if m:
        pct = float(m.group(1))
        if pct < 0 or pct > 100:
            return False, f"{field_name}: auto percentile must be between 0 and 100 (e.g., auto:p96)."
        return True, None

    try:
        val = float(s)
    except Exception:
        return False, f"{field_name}: expected numeric value or auto:pNN (e.g., auto:p96)."

    if min_value is not None and val < min_value:
        return False, f"{field_name}: must be >= {min_value}."
    if max_value is not None and val > max_value:
        return False, f"{field_name}: must be <= {max_value}."
    return True, None


def validate_cluster_eps(value: str) -> tuple[bool, str | None]:
    s = value.strip().lower()
    if not s:
        return False, "Cluster radius (m): value is required."
    if s == "auto":
        return True, None
    try:
        v = float(s)
    except Exception:
        return False, "Cluster radius (m): expected 'auto' or a positive number (e.g., 150)."
    if v <= 0:
        return False, "Cluster radius (m): must be > 0 when numeric."
    return True, None


def zip_run_dir(run_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in run_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(run_dir.parent))
    return zip_path


def read_text_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def wait_for_file(path: Path, timeout_s: float = 6.0, poll_s: float = 0.25) -> bool:
    """
    Small UX helper: pipelines sometimes finish, but the last files land a moment later.
    This avoids “it appeared a minute later” confusion.
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(poll_s)
    return path.exists()


def parse_values_used(process_log: str) -> dict:
    out = {}
    m = re.search(r"Positive relief threshold \(m\):\s*([0-9.]+)\s*\(spec=([^)]+)\)", process_log)
    if m:
        out["pos_thresh_m"] = float(m.group(1))
        out["pos_thresh_spec"] = m.group(2).strip()

    m = re.search(r"Min density threshold:\s*([0-9.]+)\s*\(spec=([^)]+)\)", process_log)
    if m:
        out["min_density"] = float(m.group(1))
        out["min_density_spec"] = m.group(2).strip()

    m = re.search(r"DBSCAN eps auto-chosen:\s*([0-9.]+)\s*m\s*\(min_samples=([0-9]+)\)", process_log)
    if m:
        out["dbscan_eps_m"] = float(m.group(1))
        out["dbscan_min_samples"] = int(m.group(2))

    m = re.search(r"Kept candidates after density \+ post-filters:\s*([0-9]+)", process_log)
    if m:
        out["candidates_kept"] = int(m.group(1))

    m = re.search(r"Clusters found:\s*([0-9]+)\s*\(noise=([0-9]+)\)", process_log)
    if m:
        out["clusters_found"] = int(m.group(1))
        out["clusters_noise"] = int(m.group(2))

    return out


@st.cache_data(show_spinner=False)
def inline_report_images_and_basemap(report_html_path_str: str, run_dir_str: str, report_mtime_ns: int) -> str:
    """
    1) Inline html/img/*.png as data URIs so embedded report shows images.
    2) Add Street/Satellite basemap toggle to the Leaflet map in report.html (if applicable).
    """
    # report_mtime_ns is included for cache invalidation when report.html changes
    _ = report_mtime_ns
    report_html_path = Path(report_html_path_str)
    run_dir = Path(run_dir_str)
    html = read_text_safely(report_html_path)
    if not html.strip():
        return ""

    # Inline local PNGs
    pattern = re.compile(r"""src=(['"])(html/img/[^'"]+)\1""", re.IGNORECASE)

    def repl(match):
        rel = match.group(2)
        img_path = run_dir / rel
        if not img_path.exists():
            return match.group(0)
        try:
            b = img_path.read_bytes()
            b64 = base64.b64encode(b).decode("ascii")
            return f'''src="data:image/png;base64,{b64}"'''
        except Exception:
            return match.group(0)

    html2 = pattern.sub(repl, html)

    # Make embedded map taller
    html2 = html2.replace("#map { height: 520px;", "#map { height: 680px;")

    # Replace the single OSM layer with a Street/Satellite basemap control (best-effort)
    tilelayer_re = re.compile(
        r"""L\.tileLayer\(\s*['"]https://\{s\}\.tile\.openstreetmap\.org/\{z\}/\{x\}/\{y\}\.png['"]\s*,\s*\{.*?\}\s*\)\.addTo\(map\);""",
        re.DOTALL,
    )

    replacement = """
// Basemaps
const street = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
});

const satellite = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  { maxZoom: 19, attribution: 'Tiles &copy; Esri' }
);

// Default basemap
street.addTo(map);

// Layer control
L.control.layers(
  { "Street": street, "Satellite": satellite },
  {},
  { collapsed: true, position: 'topright' }
).addTo(map);
""".strip()

    if tilelayer_re.search(html2):
        html2 = tilelayer_re.sub(replacement, html2)

    return html2


@st.cache_data(show_spinner=False)
def _load_candidates_cached(csv_path_str: str, csv_mtime_ns: int) -> pd.DataFrame | None:
    # csv_mtime_ns is included for cache invalidation when candidates.csv changes
    _ = csv_mtime_ns
    p = Path(csv_path_str)
    try:
        df = pd.read_csv(p)
        if "lat" not in df.columns or "lon" not in df.columns:
            return None
        df = df.dropna(subset=["lat", "lon"]).copy()
        if "cluster_id" in df.columns:
            df["cluster_id"] = df["cluster_id"].fillna(-1).astype(int)
        else:
            df["cluster_id"] = -1
        return df
    except Exception:
        return None


def load_candidates(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "candidates.csv"
    if not p.exists():
        return None
    try:
        return _load_candidates_cached(str(p), p.stat().st_mtime_ns)
    except Exception:
        return None


# -----------------------------
# Leaflet map (no Mapbox token needed)
# -----------------------------
def leaflet_map_html(df: pd.DataFrame) -> str:
    """
    A reliable, no-API-key map:
    - Street + Satellite basemap toggle
    - Fits to all candidate points
    - Click a point to see key metrics
    """
    # Keep only what we need (prevents gigantic HTML)
    cols = [c for c in ["cand_id", "score", "peak_relief_m", "area_m2", "extent", "aspect", "compactness", "solidity", "cluster_id", "lat", "lon"] if c in df.columns]
    pts = df[cols].copy()

    # Ensure numeric for JS formatting
    for c in ["score", "peak_relief_m", "area_m2", "extent", "aspect", "compactness", "solidity", "lat", "lon"]:
        if c in pts.columns:
            pts[c] = pd.to_numeric(pts[c], errors="coerce")

    pts = pts.dropna(subset=["lat", "lon"]).to_dict(orient="records")
    data_json = json.dumps(pts)

    return f"""
<div id="map" style="height:720px; border-radius:12px; overflow:hidden;"></div>

<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script>
const data = {data_json};

const map = L.map('map', {{
  zoomControl: true
}});

const street = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}});

const satellite = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{
    maxZoom: 19,
    attribution: 'Tiles &copy; Esri'
  }}
);

street.addTo(map);
L.control.layers({{ "Street": street, "Satellite": satellite }}, {{}}, {{ collapsed: true, position: 'topright' }}).addTo(map);

const bounds = [];
const clusterCounts = {{}};
const palette = [
  "#c0392b", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
  "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f"
];

function fmt(v, digits=3) {{
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}}

function clusterColor(cid) {{
  if (cid === -1) return "#666";
  const idx = Math.abs(cid - 1) % palette.length;
  return palette[idx];
}}

data.forEach(p => {{
  const lat = Number(p.lat);
  const lon = Number(p.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

  const cid = (p.cluster_id === null || p.cluster_id === undefined) ? -1 : Number(p.cluster_id);
  const color = clusterColor(cid);
  clusterCounts[cid] = (clusterCounts[cid] || 0) + 1;

  const marker = L.circleMarker([lat, lon], {{
    radius: 6,
    color: color,
    weight: 2,
    fillOpacity: 0.7
  }});

    const popup = `
      <b>Candidate ${"{p.cand_id}"}</b><br/>
      Score: ${"{fmt(p.score, 3)}"}<br/>
      Peak relief (m): ${"{fmt(p.peak_relief_m, 2)}"}<br/>
      Area (m²): ${"{fmt(p.area_m2, 1)}"}<br/>
      Extent: ${"{fmt(p.extent, 3)}"}<br/>
      Compactness: ${"{fmt(p.compactness, 3)}"}<br/>
      Solidity: ${"{fmt(p.solidity, 3)}"}<br/>
      Elongation: ${"{fmt(p.aspect, 2)}"}<br/>
      Cluster: ${"{cid}"}
    `;

  marker.bindPopup(popup);
  marker.addTo(map);
  bounds.push([lat, lon]);
}});

if (Object.keys(clusterCounts).length > 0) {{
  const legend = L.control({{ position: "bottomright" }});
  legend.onAdd = function() {{
    const div = L.DomUtil.create("div");
    div.style.background = "rgba(255,255,255,0.92)";
    div.style.padding = "8px 10px";
    div.style.border = "1px solid #ddd";
    div.style.borderRadius = "8px";
    div.style.boxShadow = "0 1px 4px rgba(0,0,0,0.15)";
    div.style.fontSize = "12px";
    div.style.lineHeight = "1.3";
    div.style.maxHeight = "180px";
    div.style.overflowY = "auto";

    const ids = Object.keys(clusterCounts)
      .map(v => Number(v))
      .sort((a, b) => (a === -1 ? -999 : a) - (b === -1 ? -999 : b));

    let rows = '<div style="font-weight:600; margin-bottom:6px;">Clusters</div>';
    ids.forEach(cid => {{
      const color = clusterColor(cid);
      const label = (cid === -1) ? "Noise (-1)" : "Cluster " + cid;
      const count = clusterCounts[cid] || 0;
      rows += '<div style="display:flex; align-items:center; margin:2px 0;">'
        + '<span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:' + color + '; margin-right:8px;"></span>'
        + '<span>' + label + " (" + count + ")" + "</span>"
        + "</div>";
    }});

    div.innerHTML = rows;
    return div;
  }};
  legend.addTo(map);
}}

if (bounds.length > 0) {{
  map.fitBounds(bounds, {{ padding: [24, 24] }});
}} else {{
  map.setView([17.0, -89.0], 10);
}}
</script>
"""


# -----------------------------
# Page chrome
# -----------------------------
st.set_page_config(page_title="MayaScan", layout="wide")

st.markdown(
    """
<style>
/* Make sure header never looks clipped */
.block-container { padding-top: 4.5rem !important; padding-bottom: 2rem; }

/* Sidebar spacing */
section[data-testid="stSidebar"] .block-container { padding-top: 0.9rem; }

/* Reduce widget gaps slightly */
div[data-testid="stVerticalBlock"] > div { gap: 0.6rem; }

/* Round code blocks */
pre { border-radius: 12px !important; }

/* Tabs padding */
button[data-baseweb="tab"] { padding-top: 8px; padding-bottom: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# Header
if LOGO_PATH.exists():
    try:
        logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        header_icon_html = (
            f"<img src='data:image/svg+xml;base64,{logo_b64}' "
            "alt='MayaScan logo' style='width:78px; height:78px; display:block;'/>"
        )
    except Exception:
        header_icon_html = "<div style='font-size:30px; line-height:1;'>🏛️</div>"
else:
    header_icon_html = "<div style='font-size:30px; line-height:1;'>🏛️</div>"

st.markdown(
    f"""
<div style="display:flex; align-items:flex-start; gap:12px; margin:0.1rem 0 0.35rem 0;">
  <div style="margin-top:-8px;">{header_icon_html}</div>
  <div>
    <div style="font-size:30px; font-weight:800; line-height:1.05;">MayaScan</div>
    <div style="color:#555; font-size:14px; margin-top:3px;">
      Upload a LiDAR tile, run the pipeline, and review results (map + ranked candidates).
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if not SCRIPT_PATH.exists():
    st.error(f"Could not find maya_scan.py at: {SCRIPT_PATH}")
    st.stop()

if "last_run_dir" not in st.session_state:
    st.session_state.last_run_dir = None
if "last_cmd" not in st.session_state:
    st.session_state.last_cmd = None
if "last_logs" not in st.session_state:
    st.session_state.last_logs = ""
if "zip_ready_for_run" not in st.session_state:
    st.session_state.zip_ready_for_run = None
if "zip_path" not in st.session_state:
    st.session_state.zip_path = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "last_preset_selected" not in st.session_state:
    st.session_state.last_preset_selected = None
if "last_preset_match" not in st.session_state:
    st.session_state.last_preset_match = None
if "last_ui_config" not in st.session_state:
    st.session_state.last_ui_config = None
if "last_compare_summary" not in st.session_state:
    st.session_state.last_compare_summary = None

if "cfg_preset" not in st.session_state:
    st.session_state.cfg_preset = PRESET_BALANCED
if "cfg_pos_thresh" not in st.session_state:
    apply_preset_to_session(st.session_state.cfg_preset)
if "cfg_preset_prev" not in st.session_state:
    st.session_state.cfg_preset_prev = st.session_state.cfg_preset


# -----------------------------
# Sidebar: Inputs + Parameters
# -----------------------------
with st.sidebar:
    st.markdown("### 1) Input")
    mode = st.radio("Source", ["Upload .laz/.las", "Use local path"], horizontal=False)

    uploaded = None
    input_local = ""
    if mode == "Upload .laz/.las":
        uploaded = st.file_uploader("LAZ/LAS file", type=["laz", "las"])
        st.caption("Saved into `data/lidar/` for this run.")
    else:
        input_local = st.text_input("Local path", value="data/lidar/bz_hr_las31_crs.laz")

    st.divider()
    st.markdown("### 2) Run")
    default_run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = st.text_input("Run name", value=default_run_name)
    runs_dir = st.text_input("Runs directory", value="runs")
    runs_dir_preview = resolve_runs_dir(runs_dir)

    existing_runs = list_existing_runs(runs_dir_preview)
    run_option_labels = {name: summarize_run_option(runs_dir_preview / name) for name in existing_runs}
    selected_existing_run = st.selectbox(
        "Load existing run",
        options=["(none)"] + existing_runs,
        index=0,
        format_func=lambda name: "(none)" if name == "(none)" else run_option_labels.get(name, name),
        help="Load outputs from a previous run without rerunning the pipeline.",
    )
    if st.button(
        "Load selected run",
        disabled=(selected_existing_run == "(none)" or st.session_state.is_running),
    ):
        loaded_dir = runs_dir_preview / selected_existing_run
        st.session_state.last_run_dir = str(loaded_dir)
        st.session_state.last_cmd = None
        st.session_state.last_logs = ""
        st.session_state.last_preset_selected = None
        st.session_state.last_preset_match = None
        st.session_state.last_ui_config = None
        st.session_state.last_compare_summary = None
        st.session_state.zip_ready_for_run = None
        st.session_state.zip_path = None
        st.success(f"Loaded run: {selected_existing_run}")

    overwrite = st.checkbox("Overwrite run folder", value=False, help="If the run folder already exists, delete and recreate it.")
    try_smrf = st.checkbox(
        "Ground classification (SMRF)",
        value=True,
        help="Attempts to classify ground points before building the terrain model (DTM). Can help in vegetation-heavy areas.",
    )
    no_html = st.checkbox(
        "Skip HTML report",
        value=False,
        help="If enabled, MayaScan will not generate report.html or cutout image panels.",
    )

    st.divider()
    st.markdown("### 3) Scientific preset")
    preset_name = st.selectbox(
        "Parameter profile",
        options=list(PRESET_VALUES.keys()),
        key="cfg_preset",
        help="Use presets for reproducible runs, then fine-tune any field below if needed.",
    )
    preset_changed = preset_name != st.session_state.get("cfg_preset_prev")
    if preset_changed:
        apply_preset_to_session(preset_name, update_selector=False)
        st.session_state.cfg_preset_prev = preset_name
        st.success(f"Applied preset automatically: {preset_name}")

    apply_preset_clicked = st.button("Re-apply preset values", use_container_width=True)
    if apply_preset_clicked:
        apply_preset_to_session(preset_name, update_selector=False)
        st.session_state.cfg_preset_prev = preset_name
        st.success(f"Re-applied preset: {preset_name}")
    preset_match = preset_matches_session(preset_name)
    if preset_match:
        st.caption("Current values match the selected preset.")
    else:
        st.caption("Current values are custom (modified from preset).")

    st.divider()
    st.markdown("### 4) Detection (what becomes a candidate?)")
    pos_thresh = st.text_input(
        "Relief threshold (auto percentile or meters)",
        key="cfg_pos_thresh",
        help=(
            "Controls how strong a terrain bump must be (in the Local Relief Model) to become a candidate.\n\n"
            "Examples:\n"
            "- `auto:p96` = use the 96th percentile of positive relief values in this tile\n"
            "- `0.25` = fixed 0.25 m relief threshold"
        ),
    )
    min_density = st.text_input(
        "Neighborhood density threshold (auto percentile or 0–1)",
        key="cfg_min_density",
        help=(
            "After candidates are detected, MayaScan builds a smoothed “feature density” raster.\n"
            "This removes isolated noise and keeps candidates in locally feature-rich zones.\n\n"
            "Examples:\n"
            "- `auto:p60` = keep candidates above the 60th percentile density\n"
            "- `0.12` = keep candidates where density ≥ 0.12"
        ),
    )
    density_sigma = st.number_input(
        "Density smoothing scale (pixels)",
        min_value=1.0,
        max_value=500.0,
        key="cfg_density_sigma",
        step=1.0,
        help="Smoothing radius for density. Bigger = settlement-scale patterns; smaller = more local sensitivity.",
    )
    max_slope_deg = st.number_input(
        "Maximum region slope q75 (degrees)",
        min_value=0.0,
        max_value=89.0,
        key="cfg_max_slope_deg",
        step=0.5,
        help=(
            "Candidate regions are rejected if their 75th-percentile slope exceeds this value. "
            "Lower values suppress steep-slope artifacts."
        ),
    )

    st.divider()
    st.markdown("### 5) Shape cleanup (remove noise-like shapes)")
    min_peak = st.number_input(
        "Minimum peak relief (m)",
        min_value=0.0,
        max_value=5.0,
        key="cfg_min_peak",
        step=0.05,
        help="Drops candidates whose strongest relief peak is below this (often tiny terrain wiggles).",
    )
    min_area_m2 = st.number_input(
        "Minimum footprint area (m²)",
        min_value=0.0,
        max_value=5000.0,
        key="cfg_min_area_m2",
        step=1.0,
        help="Drops very small candidate patches (area in square meters).",
    )
    min_extent = st.number_input(
        "Minimum extent (0–1)",
        min_value=0.0,
        max_value=1.0,
        key="cfg_min_extent",
        step=0.01,
        help="Extent = area / bounding-box-area. Higher = more coherent; lower often = thin/noisy ridges.",
    )
    max_aspect = st.number_input(
        "Maximum elongation (≥1)",
        min_value=1.0,
        max_value=50.0,
        key="cfg_max_aspect",
        step=0.1,
        help="Elongation = max(width/height, height/width). High values are long/skinny shapes (often ridges/edges/artifacts).",
    )
    min_compactness = st.number_input(
        "Minimum compactness (0–1)",
        min_value=0.0,
        max_value=1.0,
        key="cfg_min_compactness",
        step=0.01,
        help=(
            "Compactness = 4*pi*Area / Perimeter^2. "
            "Lower values are line-like features (often drainage/ridges)."
        ),
    )
    min_solidity = st.number_input(
        "Minimum solidity (0–1)",
        min_value=0.0,
        max_value=1.0,
        key="cfg_min_solidity",
        step=0.01,
        help=(
            "Solidity = Area / Convex Hull Area. "
            "Lower values are fragmented or highly irregular features."
        ),
    )

    st.divider()
    st.markdown("### 6) Clustering (settlement patterns)")
    cluster_eps = st.text_input("Cluster radius (m)", key="cfg_cluster_eps", help="DBSCAN radius. Use `auto` or a number like `150`.")
    min_samples = st.number_input("Min candidates per cluster", min_value=1, max_value=50, key="cfg_min_samples", step=1)

    st.divider()
    st.markdown("### 7) Outputs")
    report_top_n = st.number_input("Featured candidates (Top N)", min_value=1, max_value=500, key="cfg_report_top_n", step=1)
    label_top_n = st.number_input("KML labeled points (Top N)", min_value=0, max_value=5000, key="cfg_label_top_n", step=5)

    st.divider()
    run_btn = st.button("▶ Run MayaScan", type="primary", disabled=st.session_state.is_running)
    st.markdown("### 8) Compare presets")
    compare_presets = st.multiselect(
        "Presets to compare",
        options=list(PRESET_VALUES.keys()),
        default=[PRESET_BALANCED, PRESET_STRICT],
        help="Runs each selected preset on the same input tile and summarizes differences.",
    )
    compare_btn = st.button(
        "▶ Run preset comparison",
        disabled=st.session_state.is_running or len(compare_presets) < 2,
        help="Select at least two presets to enable comparison.",
    )
    if st.session_state.is_running:
        st.caption("Run in progress...")


# -----------------------------
# Main area: Tabs
# -----------------------------
tab_results, tab_runlogs, tab_glossary = st.tabs(["📍 Results", "⚙️ Run details", "📖 Glossary"])

def resolve_input_and_run():
    st.session_state.last_compare_summary = None
    runs_dir_path = resolve_runs_dir(runs_dir)
    runs_dir_path.mkdir(parents=True, exist_ok=True)

    safe_run_name = sanitize_run_name(run_name)
    if safe_run_name != run_name.strip():
        st.warning(f"Run name sanitized to `{safe_run_name}` for filesystem safety.")

    pos_thresh_spec = pos_thresh.strip()
    min_density_spec = min_density.strip()
    cluster_eps_spec = cluster_eps.strip()

    errors: list[str] = []
    ok, err = validate_auto_or_numeric(
        pos_thresh_spec,
        field_name="Relief threshold",
        min_value=0.0,
    )
    if not ok and err:
        errors.append(err)

    ok, err = validate_auto_or_numeric(
        min_density_spec,
        field_name="Neighborhood density threshold",
        min_value=0.0,
        max_value=1.0,
    )
    if not ok and err:
        errors.append(err)

    ok, err = validate_cluster_eps(cluster_eps_spec)
    if not ok and err:
        errors.append(err)

    if errors:
        for e in errors:
            st.error(e)
        return None, None, None

    if cluster_eps_spec.lower() == "auto":
        cluster_eps_spec = "auto"

    input_path = resolve_input_path(mode, uploaded, input_local)
    if input_path is None:
        return None, None, None

    run_dir = runs_dir_path / safe_run_name

    cmd = build_cmd(
        input_path=input_path,
        run_name=safe_run_name,
        runs_dir=runs_dir_path,
        overwrite=overwrite,
        try_smrf=try_smrf,
        pos_thresh=pos_thresh_spec,
        min_density=min_density_spec,
        density_sigma=float(density_sigma),
        max_slope_deg=float(max_slope_deg),
        min_peak=float(min_peak),
        min_area_m2=float(min_area_m2),
        min_extent=float(min_extent),
        max_aspect=float(max_aspect),
        min_compactness=float(min_compactness),
        min_solidity=float(min_solidity),
        cluster_eps=cluster_eps_spec,
        min_samples=int(min_samples),
        report_top_n=int(report_top_n),
        label_top_n=int(label_top_n),
        no_html=no_html,
    )

    st.session_state.last_run_dir = str(run_dir)
    st.session_state.last_cmd = cmd
    st.session_state.last_preset_selected = preset_name
    st.session_state.last_preset_match = bool(preset_match)
    st.session_state.last_ui_config = current_ui_config_snapshot()
    st.session_state.last_compare_summary = None
    st.session_state.zip_ready_for_run = None
    st.session_state.zip_path = None

    log_lines = []
    with tab_runlogs:
        st.markdown("### Run")
        st.caption("When finished, switch to **Results** to review the map, candidates, and report.")

        with st.expander("Command", expanded=False):
            st.code(" ".join(cmd), language="bash")

        status = st.empty()
        status.info("🏛️ Scanning terrain…")
        progress_bar = st.progress(0.02)

        with st.expander("Live logs", expanded=True):
            log_box = st.empty()

            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if line:
                    log_lines.append(line.rstrip("\n"))
                    tail = "\n".join(log_lines[-450:])
                    log_box.code(tail, language="text")
                    step_info = parse_step_from_log_line(line.rstrip("\n"))
                    if step_info:
                        step_idx, step_title = step_info
                        # maya_scan has step 0..7
                        frac = max(0.05, min(0.95, float(step_idx + 1) / 8.0))
                        progress_bar.progress(frac)
                        status.info(f"🏛️ Step {step_idx}: {step_title}")
                if line == "" and proc.poll() is not None:
                    break
                time.sleep(0.01)

            rc = proc.returncode

    st.session_state.last_logs = "\n".join(log_lines)

    with tab_runlogs:
        if rc == 0:
            progress_bar.progress(1.0)
            status.success(f"Done ✅  Output folder: {run_dir}")
        else:
            status.error(f"Failed ❌  Exit code: {rc}")

    return run_dir, runs_dir_path, cmd


def resolve_and_run_preset_comparison():
    st.session_state.last_compare_summary = None
    runs_dir_path = resolve_runs_dir(runs_dir)
    runs_dir_path.mkdir(parents=True, exist_ok=True)

    safe_base_name = sanitize_run_name(run_name)
    if len(compare_presets) < 2:
        st.error("Select at least two presets for comparison.")
        return None

    input_path = resolve_input_path(mode, uploaded, input_local)
    if input_path is None:
        return None

    compare_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_specs: list[dict[str, object]] = []
    for p_name in compare_presets:
        vals = PRESET_VALUES[p_name]
        suffix = preset_slug(p_name)
        run_name_i = sanitize_run_name(f"{safe_base_name}_{suffix}_{compare_stamp}")
        run_dir_i = runs_dir_path / run_name_i

        cmd_i = build_cmd(
            input_path=input_path,
            run_name=run_name_i,
            runs_dir=runs_dir_path,
            overwrite=overwrite,
            try_smrf=try_smrf,
            pos_thresh=str(vals["cfg_pos_thresh"]),
            min_density=str(vals["cfg_min_density"]),
            density_sigma=float(vals["cfg_density_sigma"]),
            max_slope_deg=float(vals["cfg_max_slope_deg"]),
            min_peak=float(vals["cfg_min_peak"]),
            min_area_m2=float(vals["cfg_min_area_m2"]),
            min_extent=float(vals["cfg_min_extent"]),
            max_aspect=float(vals["cfg_max_aspect"]),
            min_compactness=float(vals["cfg_min_compactness"]),
            min_solidity=float(vals["cfg_min_solidity"]),
            cluster_eps=str(vals["cfg_cluster_eps"]),
            min_samples=int(vals["cfg_min_samples"]),
            report_top_n=int(vals["cfg_report_top_n"]),
            label_top_n=int(vals["cfg_label_top_n"]),
            no_html=no_html,
        )
        run_specs.append(
            {
                "preset": p_name,
                "run_name": run_name_i,
                "run_dir": run_dir_i,
                "cmd": cmd_i,
            }
        )

    log_lines: list[str] = []
    run_summaries: list[dict[str, object]] = []
    n = len(run_specs)

    with tab_runlogs:
        st.markdown("### Preset comparison")
        st.caption("Running selected presets on the same input tile.")

        with st.expander("Commands", expanded=False):
            for spec in run_specs:
                st.code(" ".join(spec["cmd"]), language="bash")

        status = st.empty()
        progress_bar = st.progress(0.02)
        with st.expander("Live logs", expanded=True):
            log_box = st.empty()

            for i, spec in enumerate(run_specs):
                preset_i = str(spec["preset"])
                cmd_i = spec["cmd"]
                run_dir_i = Path(spec["run_dir"])
                status.info(f"🏛️ Running {preset_i} ({i + 1}/{n})...")

                proc = subprocess.Popen(
                    cmd_i,
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                while True:
                    line = proc.stdout.readline() if proc.stdout else ""
                    if line:
                        msg = line.rstrip("\n")
                        log_lines.append(f"[{preset_i}] {msg}")
                        tail = "\n".join(log_lines[-450:])
                        log_box.code(tail, language="text")
                        step_info = parse_step_from_log_line(msg)
                        if step_info:
                            step_idx, step_title = step_info
                            local_frac = max(0.02, min(0.98, float(step_idx + 1) / 8.0))
                            global_frac = ((i + local_frac) / n)
                            progress_bar.progress(max(0.02, min(0.99, global_frac)))
                            status.info(f"🏛️ {preset_i}: Step {step_idx} - {step_title}")
                    if line == "" and proc.poll() is not None:
                        break
                    time.sleep(0.01)

                rc = proc.returncode
                if rc != 0:
                    st.session_state.last_logs = "\n".join(log_lines)
                    st.session_state.last_cmd = cmd_i
                    st.session_state.last_compare_summary = None
                    status.error(f"Failed ❌  {preset_i} exited with code {rc}.")
                    return None

                s = read_run_summary(run_dir_i)
                s["preset"] = preset_i
                s["run_name"] = str(spec["run_name"])
                run_summaries.append(s)
                progress_bar.progress(max(0.02, min(0.99, float(i + 1) / n)))

        progress_bar.progress(1.0)
        status.success("Preset comparison complete ✅")

    baseline = run_summaries[0] if run_summaries else {}
    base_candidates = baseline.get("candidates")
    base_top_score = baseline.get("top_score")
    for s in run_summaries:
        cand = s.get("candidates")
        top = s.get("top_score")
        try:
            s["d_candidates_vs_baseline"] = int(cand) - int(base_candidates) if cand is not None and base_candidates is not None else None
        except Exception:
            s["d_candidates_vs_baseline"] = None
        try:
            s["d_top_score_vs_baseline"] = round(float(top) - float(base_top_score), 4) if top is not None and base_top_score is not None else None
        except Exception:
            s["d_top_score_vs_baseline"] = None

    compare_payload: dict[str, object] = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": str(input_path),
        "base_name": safe_base_name,
        "baseline_preset": baseline.get("preset"),
        "presets": [str(x) for x in compare_presets],
        "runs": run_summaries,
    }
    json_path, md_path = write_preset_compare_artifacts(runs_dir_path, safe_base_name, compare_payload)
    compare_payload["comparison_json"] = str(json_path)
    compare_payload["comparison_md"] = str(md_path)

    focus_preset = PRESET_BALANCED if PRESET_BALANCED in compare_presets else str(compare_presets[0])
    focus_spec = next((s for s in run_specs if str(s["preset"]) == focus_preset), run_specs[0])
    st.session_state.last_run_dir = str(focus_spec["run_dir"])
    st.session_state.last_cmd = focus_spec["cmd"]
    st.session_state.last_logs = "\n".join(log_lines)
    st.session_state.last_preset_selected = focus_preset
    st.session_state.last_preset_match = True
    st.session_state.last_ui_config = dict(PRESET_VALUES.get(focus_preset, {}))
    st.session_state.last_compare_summary = compare_payload
    st.session_state.zip_ready_for_run = None
    st.session_state.zip_path = None
    return compare_payload


if run_btn and not st.session_state.is_running:
    st.session_state.is_running = True
    try:
        resolve_input_and_run()
    finally:
        st.session_state.is_running = False

if compare_btn and not st.session_state.is_running:
    st.session_state.is_running = True
    try:
        resolve_and_run_preset_comparison()
    finally:
        st.session_state.is_running = False


# -----------------------------
# Results tab
# -----------------------------
with tab_results:
    st.markdown("### Review")

    compare_summary = st.session_state.get("last_compare_summary")
    if compare_summary:
        st.markdown("#### Preset comparison summary")
        baseline = compare_summary.get("baseline_preset")
        if baseline:
            st.caption(f"Baseline preset: {baseline}")

        rows = compare_summary.get("runs", [])
        if isinstance(rows, list) and rows:
            cmp_df = pd.DataFrame(rows)
            show_cols = [
                c for c in [
                    "preset",
                    "run_name",
                    "candidates",
                    "clusters",
                    "noise",
                    "top_score",
                    "mean_score",
                    "median_score",
                    "d_candidates_vs_baseline",
                    "d_top_score_vs_baseline",
                ]
                if c in cmp_df.columns
            ]
            if show_cols:
                st.dataframe(cmp_df[show_cols], use_container_width=True)

        cjson = compare_summary.get("comparison_json")
        cmd = compare_summary.get("comparison_md")
        d1, d2 = st.columns(2)
        with d1:
            if cjson:
                p = Path(str(cjson))
                if p.exists():
                    st.download_button(
                        "Download comparison JSON",
                        data=p.read_bytes(),
                        file_name=p.name,
                        mime="application/json",
                    )
        with d2:
            if cmd:
                p = Path(str(cmd))
                if p.exists():
                    st.download_button(
                        "Download comparison markdown",
                        data=p.read_bytes(),
                        file_name=p.name,
                        mime="text/markdown",
                    )
        st.divider()

    if not st.session_state.last_run_dir:
        st.info("Run MayaScan from the sidebar to see results here.")
    else:
        run_dir = Path(st.session_state.last_run_dir)
        if not run_dir.exists():
            st.warning(f"Last run folder not found: {run_dir}")
        else:
            # Wait briefly for pipeline outputs to fully land
            wait_for_file(run_dir / "candidates.csv", timeout_s=6.0)
            wait_for_file(run_dir / "process.log", timeout_s=6.0)

            process_log = read_text_safely(run_dir / "process.log")
            used = parse_values_used(process_log)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Candidates", used.get("candidates_kept", "—"))
            c2.metric("Relief threshold used (m)", used.get("pos_thresh_m", "—"))
            c3.metric("Density threshold used", used.get("min_density", "—"))
            c4.metric("Cluster radius used (m)", used.get("dbscan_eps_m", "—"))
            c5.metric("Clusters", used.get("clusters_found", "—"))

            if used:
                st.caption(
                    f"Actual values (from process.log): "
                    f"relief={used.get('pos_thresh_m','—')} ({used.get('pos_thresh_spec','—')}), "
                    f"density={used.get('min_density','—')} ({used.get('min_density_spec','—')}), "
                    f"cluster_eps={used.get('dbscan_eps_m','—')} m."
                )

            with st.expander("Run settings used", expanded=False):
                settings_data: dict[str, object] = {"run_dir": str(run_dir)}
                settings_data["ui_preset_selected"] = st.session_state.get("last_preset_selected")
                settings_data["ui_preset_match"] = st.session_state.get("last_preset_match")
                if st.session_state.last_cmd:
                    settings_data["command"] = parse_cmd_settings(st.session_state.last_cmd)
                if st.session_state.get("last_ui_config"):
                    settings_data["ui_config_snapshot"] = st.session_state["last_ui_config"]

                run_params_path = run_dir / "run_params.json"
                if run_params_path.exists():
                    try:
                        settings_data["run_params_json"] = json.loads(run_params_path.read_text(encoding="utf-8"))
                    except Exception:
                        settings_data["run_params_json"] = "Could not parse run_params.json"

                if used:
                    settings_data["resolved_from_log"] = {
                        "relief_threshold_m": used.get("pos_thresh_m"),
                        "relief_threshold_spec": used.get("pos_thresh_spec"),
                        "min_density": used.get("min_density"),
                        "min_density_spec": used.get("min_density_spec"),
                        "dbscan_eps_m": used.get("dbscan_eps_m"),
                        "dbscan_min_samples": used.get("dbscan_min_samples"),
                        "candidates_kept": used.get("candidates_kept"),
                        "clusters_found": used.get("clusters_found"),
                        "clusters_noise": used.get("clusters_noise"),
                    }

                report_md_path = run_dir / "report.md"
                if report_md_path.exists():
                    m = re.search(r"- Input:\s*`([^`]+)`", read_text_safely(report_md_path))
                    if m:
                        settings_data["input"] = m.group(1)

                st.json(settings_data)

            st.divider()

            df = load_candidates(run_dir)
            filtered_df = df

            if df is not None and not df.empty:
                filter_col1, filter_col2 = st.columns(2)

                min_score_filter: float | None = None
                with filter_col1:
                    if "score" in df.columns:
                        score_series = pd.to_numeric(df["score"], errors="coerce").dropna()
                        if not score_series.empty:
                            score_min = float(score_series.min())
                            score_max = float(score_series.max())
                            if score_max > score_min:
                                step = max((score_max - score_min) / 200.0, 0.001)
                                min_score_filter = st.slider(
                                    "Minimum score",
                                    min_value=score_min,
                                    max_value=score_max,
                                    value=score_min,
                                    step=step,
                                    format="%.3f",
                                )
                            else:
                                min_score_filter = score_min
                                st.caption(f"All scores are {score_min:.3f}.")
                    else:
                        st.caption("No score column available; score filter disabled.")

                cluster_label_to_id: dict[str, int] = {}
                selected_cluster_labels: list[str] = ["All clusters"]
                with filter_col2:
                    if "cluster_id" in df.columns:
                        cluster_ids = sorted(pd.to_numeric(df["cluster_id"], errors="coerce").dropna().astype(int).unique().tolist())
                        cluster_label_to_id = {
                            ("Noise (-1)" if cid == -1 else f"Cluster {cid}"): cid
                            for cid in cluster_ids
                        }
                        cluster_options = ["All clusters"] + list(cluster_label_to_id.keys())
                        selected_cluster_labels = st.multiselect(
                            "Clusters",
                            options=cluster_options,
                            default=["All clusters"],
                        )
                    else:
                        st.caption("No cluster_id column available; cluster filter disabled.")

                filtered_df = df.copy()
                if min_score_filter is not None and "score" in filtered_df.columns:
                    filtered_df = filtered_df[pd.to_numeric(filtered_df["score"], errors="coerce") >= float(min_score_filter)]

                if "cluster_id" in filtered_df.columns and "All clusters" not in selected_cluster_labels:
                    allowed_ids = [cluster_label_to_id[label] for label in selected_cluster_labels if label in cluster_label_to_id]
                    filtered_df = filtered_df[filtered_df["cluster_id"].isin(allowed_ids)]

                st.caption(f"Showing {len(filtered_df)} of {len(df)} candidates after filters.")

            report_html = run_dir / "report.html"
            report_md = run_dir / "report.md"
            run_params_json = run_dir / "run_params.json"
            candidates_csv = run_dir / "candidates.csv"
            geojson = run_dir / "candidates.geojson"
            kml = run_dir / "candidates.kml"
            plots_dir = run_dir / "plots"
            img_dir = run_dir / "html" / "img"

            res_tabs = st.tabs(["🗺️ Map", "🏷️ Top candidates", "📋 Table", "🧾 Report", "📄 Files", "📈 Plots"])

            # --- Map (Leaflet: always works, Street + Satellite)
            with res_tabs[0]:
                if filtered_df is None or filtered_df.empty:
                    if df is not None and not df.empty:
                        st.info("No candidates match the current filters.")
                    else:
                        st.info("No candidates.csv found yet. Run MayaScan to generate candidates.")
                else:
                    st.markdown("#### Candidates map")
                    components.html(leaflet_map_html(filtered_df), height=740, scrolling=False)
                    st.caption("Street and Satellite layers are available in the top-right map control. Click points for details.")

            # --- Top candidates + cutouts
            with res_tabs[1]:
                if filtered_df is None or filtered_df.empty:
                    if df is not None and not df.empty:
                        st.info("No candidates match the current filters.")
                    else:
                        st.info("No candidates.csv found yet.")
                else:
                    topn = min(30, len(filtered_df))
                    top = filtered_df.sort_values("score", ascending=False).head(topn).copy()
                    st.dataframe(
                        top[
                            [c for c in [
                                "cand_id","score","density","peak_relief_m","area_m2","extent","aspect","compactness","solidity","cluster_id","lat","lon"
                            ] if c in top.columns]
                        ],
                        use_container_width=True,
                    )

                    if "cand_id" not in top.columns:
                        st.info("No cand_id column available for candidate inspection.")
                    else:
                        top_view = top.copy()
                        top_view["_cand_id_int"] = pd.to_numeric(top_view["cand_id"], errors="coerce").astype("Int64")
                        top_view = top_view.dropna(subset=["_cand_id_int"]).copy()
                        top_view["_cand_id_int"] = top_view["_cand_id_int"].astype(int)

                        if top_view.empty:
                            st.info("No valid cand_id values available for candidate inspection.")
                        else:
                            option_ids = top_view["_cand_id_int"].tolist()

                            def _label_for_cid(cid: int) -> str:
                                row = top_view[top_view["_cand_id_int"] == cid].iloc[0]
                                try:
                                    score_val = float(row.get("score"))
                                    return f"Candidate {cid} (score {score_val:.3f})"
                                except Exception:
                                    return f"Candidate {cid}"

                            selected_cid = st.selectbox(
                                "Inspect candidate",
                                options=option_ids,
                                format_func=_label_for_cid,
                                key=f"inspect_cand_{run_dir.name}",
                            )

                            selected_row = top_view[top_view["_cand_id_int"] == int(selected_cid)].iloc[0]
                            st.markdown("#### Candidate detail")
                            d1, d2, d3, d4, d5, d6 = st.columns(6)
                            d1.metric("Score", f"{float(selected_row['score']):.3f}" if "score" in selected_row else "—")
                            d2.metric("Peak relief (m)", f"{float(selected_row['peak_relief_m']):.2f}" if "peak_relief_m" in selected_row else "—")
                            d3.metric("Area (m²)", f"{float(selected_row['area_m2']):.1f}" if "area_m2" in selected_row else "—")
                            d4.metric("Extent", f"{float(selected_row['extent']):.3f}" if "extent" in selected_row else "—")
                            d5.metric("Compactness", f"{float(selected_row['compactness']):.3f}" if "compactness" in selected_row else "—")
                            d6.metric("Solidity", f"{float(selected_row['solidity']):.3f}" if "solidity" in selected_row else "—")
                            if "cluster_id" in selected_row and pd.notna(selected_row["cluster_id"]):
                                st.caption(f"Cluster: {int(selected_row['cluster_id'])}")

                            if "lat" in selected_row and "lon" in selected_row and pd.notna(selected_row["lat"]) and pd.notna(selected_row["lon"]):
                                lat = float(selected_row["lat"])
                                lon = float(selected_row["lon"])
                                st.markdown(f"[Open in Google Maps](https://www.google.com/maps?q={lat:.8f},{lon:.8f})")

                            if img_dir.exists():
                                panel_path = img_dir / f"cand_{int(selected_cid):04d}_panel.png"
                                if panel_path.exists():
                                    st.image(
                                        str(panel_path),
                                        caption=f"Candidate {int(selected_cid)} panel (LRM + hillshade)",
                                        use_container_width=True,
                                    )
                                else:
                                    st.info("No cutout panel found for this candidate (HTML/cutouts may be disabled).")
                            else:
                                st.info("No cutout images folder found for this run.")

            # --- Table
            with res_tabs[2]:
                if filtered_df is None or filtered_df.empty:
                    if df is not None and not df.empty:
                        st.info("No candidates match the current filters.")
                    else:
                        st.info("No candidates.csv found yet.")
                else:
                    st.dataframe(filtered_df.sort_values("score", ascending=False), use_container_width=True)

            # --- Report
            with res_tabs[3]:
                if report_html.exists():
                    st.markdown("#### Report (interactive)")
                    html_inlined = inline_report_images_and_basemap(
                        str(report_html),
                        str(run_dir),
                        report_html.stat().st_mtime_ns,
                    )
                    if html_inlined.strip():
                        components.html(html_inlined, height=980, scrolling=True)
                    else:
                        st.warning("report.html exists but could not be loaded as text.")
                else:
                    st.info("No report.html found (you may have disabled HTML output).")

            # --- Files
            with res_tabs[4]:
                st.markdown("#### Key outputs")
                st.code(str(run_dir), language="text")

                cols = st.columns(3)
                with cols[0]:
                    st.write("**Candidates table**")
                    if candidates_csv.exists():
                        st.success("candidates.csv")
                        st.download_button(
                            "Download candidates.csv",
                            data=candidates_csv.read_bytes(),
                            file_name=candidates_csv.name,
                            mime="text/csv",
                        )
                    else:
                        st.warning("candidates.csv not found")

                with cols[1]:
                    st.write("**GIS exports**")
                    if geojson.exists():
                        st.success("candidates.geojson")
                        st.download_button(
                            "Download candidates.geojson",
                            data=geojson.read_bytes(),
                            file_name=geojson.name,
                            mime="application/geo+json",
                        )
                    if kml.exists():
                        st.success("candidates.kml")
                        st.download_button(
                            "Download candidates.kml",
                            data=kml.read_bytes(),
                            file_name=kml.name,
                            mime="application/vnd.google-earth.kml+xml",
                        )

                with cols[2]:
                    st.write("**Reports**")
                    if report_html.exists():
                        st.success("report.html")
                        st.download_button(
                            "Download report.html",
                            data=report_html.read_bytes(),
                            file_name=report_html.name,
                            mime="text/html",
                        )
                    if report_md.exists():
                        st.success("report.md")
                        st.download_button(
                            "Download report.md",
                            data=report_md.read_bytes(),
                            file_name=report_md.name,
                            mime="text/markdown",
                        )
                    if run_params_json.exists():
                        st.success("run_params.json")
                        st.download_button(
                            "Download run_params.json",
                            data=run_params_json.read_bytes(),
                            file_name=run_params_json.name,
                            mime="application/json",
                        )

                st.divider()
                st.markdown("#### Download everything")
                zip_path = run_dir.parent / f"{run_dir.name}_outputs.zip"
                if st.button("Prepare run outputs (.zip)", key=f"prepare_zip_{run_dir.name}"):
                    with st.spinner("Preparing ZIP archive..."):
                        zip_run_dir(run_dir, zip_path)
                    st.session_state.zip_ready_for_run = str(run_dir)
                    st.session_state.zip_path = str(zip_path)

                zip_ready = (
                    st.session_state.get("zip_ready_for_run") == str(run_dir)
                    and st.session_state.get("zip_path")
                    and Path(st.session_state["zip_path"]).exists()
                )

                if zip_ready:
                    zip_ready_path = Path(st.session_state["zip_path"])
                    st.download_button(
                        label="Download run outputs (.zip)",
                        data=zip_ready_path.read_bytes(),
                        file_name=zip_ready_path.name,
                        mime="application/zip",
                    )
                else:
                    st.caption('Click "Prepare run outputs (.zip)" to generate the archive.')

            # --- Plots
            with res_tabs[5]:
                st.markdown("#### Plots")
                if plots_dir.exists():
                    # In case they appear a beat late
                    time.sleep(0.15)
                    pngs = sorted(plots_dir.glob("*.png"))
                    if not pngs:
                        st.info("No plot PNGs found.")
                    else:
                        for p in pngs:
                            st.image(str(p), caption=p.name, use_container_width=True)
                else:
                    st.info("plots/ folder not found.")


# -----------------------------
# Run details tab (show last logs)
# -----------------------------
with tab_runlogs:
    if st.session_state.get("last_compare_summary"):
        st.markdown("### Last preset comparison")
        st.json(st.session_state["last_compare_summary"])

    if st.session_state.last_cmd:
        st.markdown("### Last run")
        with st.expander("Command", expanded=False):
            st.code(" ".join(st.session_state.last_cmd), language="bash")
        with st.expander("Logs", expanded=False):
            st.code(
                st.session_state.last_logs[-20000:] if st.session_state.last_logs else "(no logs yet)",
                language="text",
            )
    else:
        st.info("Run MayaScan to see details here.")


# -----------------------------
# Glossary tab
# -----------------------------
with tab_glossary:
    st.markdown("### Glossary (plain-English)")
    st.markdown(
        """
**DTM (Digital Terrain Model)**  
A “bare earth” elevation raster derived from the point cloud.

**LRM (Local Relief Model)**  
A terrain-enhancement layer that highlights subtle bumps/edges by subtracting a smoothed terrain from a less-smoothed terrain.

**Relief threshold**  
How strong a bump must be in the LRM to become a candidate region.  
`auto:p96` means “use the 96th percentile of positive relief values” for that tile.

**Candidate region**  
A connected patch of pixels above threshold (after cleanup and region-slope q75 filtering).

**Neighborhood density**  
A smoothed “how many candidates are nearby” signal. Helps emphasize settlement-like zones and suppress isolated noise.

**Extent (bbox fill, 0–1)**  
How filled-in a region is: area / bounding-box-area.  
Higher = more coherent; lower often = thin/noisy ridges.

**Compactness (0–1)**  
Defined as `4*pi*Area / Perimeter^2`.  
Lower values are more line-like and often correspond to drainage/ridge artifacts.

**Solidity (0–1)**  
Defined as `Area / ConvexHullArea`.  
Lower values indicate fragmented/irregular shapes; higher values are more solid footprints.

**Elongation (aspect ratio, ≥1)**  
How stretched a region is.  
High values are long/skinny shapes (often ridges/edges/artifacts).

**DBSCAN clustering**  
Groups candidates into clusters based on distance in meters (useful for settlement patterns).

**Scientific presets (Strict / Balanced / Exploratory)**  
Reproducible parameter profiles to trade precision vs recall before manual tuning.

**Preset comparison run**  
Runs multiple presets on the same input tile and reports side-by-side metric deltas.
"""
    )
