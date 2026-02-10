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
    min_peak: float,
    min_area_m2: float,
    min_extent: float,
    max_aspect: float,
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

    cmd += ["--min-peak", str(min_peak)]
    cmd += ["--min-area-m2", str(min_area_m2)]
    cmd += ["--min-extent", str(min_extent)]
    cmd += ["--max-aspect", str(max_aspect)]

    if cluster_eps.strip():
        cmd += ["--cluster-eps", cluster_eps.strip()]
    cmd += ["--min-samples", str(min_samples)]

    cmd += ["--report-top-n", str(report_top_n)]
    cmd += ["--label-top-n", str(label_top_n)]

    return cmd


# -----------------------------
# Utilities
# -----------------------------
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


def inline_report_images_and_basemap(report_html_path: Path, run_dir: Path) -> str:
    """
    1) Inline html/img/*.png as data URIs so embedded report shows images.
    2) Add Street/Satellite basemap toggle to the Leaflet map in report.html (if applicable).
    """
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


def load_candidates(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "candidates.csv"
    if not p.exists():
        return None
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
    cols = [c for c in ["cand_id", "score", "peak_relief_m", "area_m2", "extent", "aspect", "cluster_id", "lat", "lon"] if c in df.columns]
    pts = df[cols].copy()

    # Ensure numeric for JS formatting
    for c in ["score", "peak_relief_m", "area_m2", "extent", "aspect", "lat", "lon"]:
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

function fmt(v, digits=3) {{
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}}

data.forEach(p => {{
  const lat = Number(p.lat);
  const lon = Number(p.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

  const cid = (p.cluster_id === null || p.cluster_id === undefined) ? -1 : Number(p.cluster_id);
  const color = (cid === -1) ? "#666" : "#c0392b";

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
    Compactness: ${"{fmt(p.extent, 3)}"}<br/>
    Elongation: ${"{fmt(p.aspect, 2)}"}<br/>
    Cluster: ${"{cid}"}
  `;

  marker.bindPopup(popup);
  marker.addTo(map);
  bounds.push([lat, lon]);
}});

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
st.markdown(
    """
<div style="display:flex; align-items:flex-start; gap:12px; margin:0.1rem 0 0.35rem 0;">
  <div style="font-size:34px; line-height:1;">🗿</div>
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
    st.markdown("### 3) Detection (what becomes a candidate?)")
    pos_thresh = st.text_input(
        "Relief threshold (auto percentile or meters)",
        value="auto:p96",
        help=(
            "Controls how strong a terrain bump must be (in the Local Relief Model) to become a candidate.\n\n"
            "Examples:\n"
            "- `auto:p96` = use the 96th percentile of positive relief values in this tile\n"
            "- `0.25` = fixed 0.25 m relief threshold"
        ),
    )
    min_density = st.text_input(
        "Neighborhood density threshold (auto percentile or 0–1)",
        value="auto:p55",
        help=(
            "After candidates are detected, MayaScan builds a smoothed “feature density” raster.\n"
            "This removes isolated noise and keeps candidates in locally feature-rich zones.\n\n"
            "Examples:\n"
            "- `auto:p55` = keep candidates above the 55th percentile density\n"
            "- `0.12` = keep candidates where density ≥ 0.12"
        ),
    )
    density_sigma = st.number_input(
        "Density smoothing scale (pixels)",
        min_value=1.0,
        max_value=500.0,
        value=40.0,
        step=1.0,
        help="Smoothing radius for density. Bigger = settlement-scale patterns; smaller = more local sensitivity.",
    )

    st.divider()
    st.markdown("### 4) Shape cleanup (remove noise-like shapes)")
    min_peak = st.number_input(
        "Minimum peak relief (m)",
        min_value=0.0,
        max_value=5.0,
        value=0.50,
        step=0.05,
        help="Drops candidates whose strongest relief peak is below this (often tiny terrain wiggles).",
    )
    min_area_m2 = st.number_input(
        "Minimum footprint area (m²)",
        min_value=0.0,
        max_value=5000.0,
        value=25.0,
        step=1.0,
        help="Drops very small candidate patches (area in square meters).",
    )
    min_extent = st.number_input(
        "Minimum compactness (0–1)",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.01,
        help="Compactness = area / bounding-box-area. Higher = more coherent; lower often = thin/noisy ridges.",
    )
    max_aspect = st.number_input(
        "Maximum elongation (≥1)",
        min_value=1.0,
        max_value=50.0,
        value=4.0,
        step=0.1,
        help="Elongation = max(width/height, height/width). High values are long/skinny shapes (often ridges/edges/artifacts).",
    )

    st.divider()
    st.markdown("### 5) Clustering (settlement patterns)")
    cluster_eps = st.text_input("Cluster radius (m)", value="auto", help="DBSCAN radius. Use `auto` or a number like `150`.")
    min_samples = st.number_input("Min candidates per cluster", min_value=1, max_value=50, value=4, step=1)

    st.divider()
    st.markdown("### 6) Outputs")
    report_top_n = st.number_input("Featured candidates (Top N)", min_value=1, max_value=500, value=30, step=1)
    label_top_n = st.number_input("KML labeled points (Top N)", min_value=0, max_value=5000, value=60, step=5)

    st.divider()
    run_btn = st.button("▶ Run MayaScan", type="primary")


# -----------------------------
# Main area: Tabs
# -----------------------------
tab_results, tab_runlogs, tab_glossary = st.tabs(["📍 Results", "⚙️ Run details", "📖 Glossary"])

if "last_run_dir" not in st.session_state:
    st.session_state.last_run_dir = None
if "last_cmd" not in st.session_state:
    st.session_state.last_cmd = None
if "last_logs" not in st.session_state:
    st.session_state.last_logs = ""


def resolve_input_and_run():
    runs_dir_path = (REPO_ROOT / runs_dir).resolve() if not Path(runs_dir).is_absolute() else Path(runs_dir).resolve()
    runs_dir_path.mkdir(parents=True, exist_ok=True)

    if mode == "Upload .laz/.las":
        if uploaded is None:
            st.error("Please upload a .laz or .las file.")
            return None, None, None
        data_dir = REPO_ROOT / "data" / "lidar"
        data_dir.mkdir(parents=True, exist_ok=True)
        input_path = data_dir / uploaded.name
        input_path.write_bytes(uploaded.getvalue())
    else:
        input_path = Path(input_local).expanduser()
        if not input_path.is_absolute():
            input_path = (REPO_ROOT / input_path).resolve()
        if not input_path.exists():
            st.error(f"Input file not found: {input_path}")
            return None, None, None

    run_dir = runs_dir_path / run_name

    cmd = build_cmd(
        input_path=input_path,
        run_name=run_name,
        runs_dir=runs_dir_path,
        overwrite=overwrite,
        try_smrf=try_smrf,
        pos_thresh=pos_thresh,
        min_density=min_density,
        density_sigma=float(density_sigma),
        min_peak=float(min_peak),
        min_area_m2=float(min_area_m2),
        min_extent=float(min_extent),
        max_aspect=float(max_aspect),
        cluster_eps=cluster_eps,
        min_samples=int(min_samples),
        report_top_n=int(report_top_n),
        label_top_n=int(label_top_n),
        no_html=no_html,
    )

    st.session_state.last_run_dir = str(run_dir)
    st.session_state.last_cmd = cmd

    log_lines = []
    with tab_runlogs:
        st.markdown("### Run")
        st.caption("When finished, switch to **Results** to review the map, candidates, and report.")

        with st.expander("Command", expanded=False):
            st.code(" ".join(cmd), language="bash")

        status = st.empty()
        status.info("🏛️ Scanning terrain…")

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
                if line == "" and proc.poll() is not None:
                    break
                time.sleep(0.01)

            rc = proc.returncode

    st.session_state.last_logs = "\n".join(log_lines)

    with tab_runlogs:
        if rc == 0:
            status.success(f"Done ✅  Output folder: {run_dir}")
        else:
            status.error(f"Failed ❌  Exit code: {rc}")

    return run_dir, runs_dir_path, cmd


if run_btn:
    resolve_input_and_run()


# -----------------------------
# Results tab
# -----------------------------
with tab_results:
    st.markdown("### Review")

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

            st.divider()

            df = load_candidates(run_dir)

            report_html = run_dir / "report.html"
            report_md = run_dir / "report.md"
            candidates_csv = run_dir / "candidates.csv"
            geojson = run_dir / "candidates.geojson"
            kml = run_dir / "candidates.kml"
            plots_dir = run_dir / "plots"
            img_dir = run_dir / "html" / "img"

            res_tabs = st.tabs(["🗺️ Map", "🏷️ Top candidates", "📋 Table", "🧾 Report", "📄 Files", "📈 Plots"])

            # --- Map (Leaflet: always works, Street + Satellite)
            with res_tabs[0]:
                if df is None or df.empty:
                    st.info("No candidates.csv found yet. Run MayaScan to generate candidates.")
                else:
                    st.markdown("#### Candidates map")
                    components.html(leaflet_map_html(df), height=740, scrolling=False)
                    st.caption("Street and Satellite layers are available in the top-right map control. Click points for details.")

            # --- Top candidates + cutouts
            with res_tabs[1]:
                if df is None or df.empty:
                    st.info("No candidates.csv found yet.")
                else:
                    topn = min(30, len(df))
                    top = df.sort_values("score", ascending=False).head(topn).copy()
                    st.dataframe(
                        top[
                            [c for c in [
                                "cand_id","score","density","peak_relief_m","area_m2","extent","aspect","cluster_id","lat","lon"
                            ] if c in top.columns]
                        ],
                        use_container_width=True,
                    )

                    if img_dir.exists():
                        st.markdown("#### Cutouts (LRM + hillshade panels)")
                        shown = 0
                        for _, r in top.iterrows():
                            try:
                                fname = f"cand_{int(r['cand_id']):04d}_panel.png"
                            except Exception:
                                continue
                            p = img_dir / fname
                            if p.exists():
                                st.image(
                                    str(p),
                                    caption=f"Candidate {int(r['cand_id'])} — score {float(r['score']):.3f}",
                                    use_container_width=True,
                                )
                                shown += 1
                        if shown == 0:
                            st.info("No cutout images found (they generate for top candidates when HTML is enabled).")

            # --- Table
            with res_tabs[2]:
                if df is None or df.empty:
                    st.info("No candidates.csv found yet.")
                else:
                    st.dataframe(df.sort_values("score", ascending=False), use_container_width=True)

            # --- Report
            with res_tabs[3]:
                if report_html.exists():
                    st.markdown("#### Report (interactive)")
                    html_inlined = inline_report_images_and_basemap(report_html, run_dir)
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

                st.divider()
                st.markdown("#### Download everything")
                zip_path = run_dir.parent / f"{run_dir.name}_outputs.zip"
                zip_run_dir(run_dir, zip_path)
                st.download_button(
                    label="Download run outputs (.zip)",
                    data=zip_path.read_bytes(),
                    file_name=zip_path.name,
                    mime="application/zip",
                )

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
A connected patch of pixels above threshold (after cleanup and slope filtering).

**Neighborhood density**  
A smoothed “how many candidates are nearby” signal. Helps emphasize settlement-like zones and suppress isolated noise.

**Compactness (extent, 0–1)**  
How filled-in a region is: area / bounding-box-area.  
Higher = more coherent; lower often = thin/noisy ridges.

**Elongation (aspect ratio, ≥1)**  
How stretched a region is.  
High values are long/skinny shapes (often ridges/edges/artifacts).

**DBSCAN clustering**  
Groups candidates into clusters based on distance in meters (useful for settlement patterns).
"""
    )