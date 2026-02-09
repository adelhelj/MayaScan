#!/usr/bin/env python3
import os
import sys
import time
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = REPO_ROOT / "maya_scan.py"


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

    # knobs (only include if set)
    if pos_thresh.strip():
        cmd += ["--pos-thresh", pos_thresh.strip()]
    if min_density.strip():
        cmd += ["--min-density", min_density.strip()]
    if density_sigma is not None:
        cmd += ["--density-sigma", str(density_sigma)]

    # post-filters
    cmd += ["--min-peak", str(min_peak)]
    cmd += ["--min-area-m2", str(min_area_m2)]
    cmd += ["--min-extent", str(min_extent)]
    cmd += ["--max-aspect", str(max_aspect)]

    # clustering
    if cluster_eps.strip():
        cmd += ["--cluster-eps", cluster_eps.strip()]
    cmd += ["--min-samples", str(min_samples)]

    # report / labels
    cmd += ["--report-top-n", str(report_top_n)]
    cmd += ["--label-top-n", str(label_top_n)]

    return cmd


def zip_run_dir(run_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in run_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(run_dir.parent))
    return zip_path


st.set_page_config(page_title="MayaScan GUI", layout="wide")
st.title("MayaScan — GUI Wrapper (CLI Runner)")

if not SCRIPT_PATH.exists():
    st.error(f"Could not find maya_scan.py at: {SCRIPT_PATH}")
    st.stop()

st.caption("This GUI runs your existing CLI with `subprocess` (no refactor). PDAL must be installed and available as `pdal`.")

left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    mode = st.radio("Input mode", ["Upload .laz/.las", "Use local path"], horizontal=True)

    uploaded = None
    input_local = ""
    if mode == "Upload .laz/.las":
        uploaded = st.file_uploader("Upload LAZ/LAS", type=["laz", "las"])
    else:
        input_local = st.text_input("Local path to .laz/.las", value="data/lidar/bz_hr_las31_crs.laz")

    st.subheader("Run")
    default_run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = st.text_input("Run name", value=default_run_name)
    runs_dir = st.text_input("Runs directory", value="runs")

    overwrite = st.checkbox("Overwrite existing run folder", value=False)
    try_smrf = st.checkbox("Try SMRF ground classification (PDAL)", value=True)
    no_html = st.checkbox("Disable HTML report + cutouts", value=False)

    st.subheader("Thresholds")
    pos_thresh = st.text_input("pos-thresh", value="auto:p96", help="Positive relief threshold; auto percentile works well.")
    min_density = st.text_input("min-density", value="auto:p55", help="Density threshold; auto percentile recommended.")
    density_sigma = st.number_input("density-sigma (pixels)", min_value=1.0, max_value=500.0, value=40.0, step=1.0)

    st.subheader("Post-filters")
    min_peak = st.number_input("min-peak (m)", min_value=0.0, max_value=5.0, value=0.50, step=0.05)
    min_area_m2 = st.number_input("min-area-m2", min_value=0.0, max_value=5000.0, value=25.0, step=1.0)
    min_extent = st.number_input("min-extent (0..1)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
    max_aspect = st.number_input("max-aspect", min_value=1.0, max_value=50.0, value=4.0, step=0.1)

    st.subheader("Clustering")
    cluster_eps = st.text_input("cluster-eps", value="auto", help="Meters; use 'auto' or a number like 150.")
    min_samples = st.number_input("min-samples", min_value=1, max_value=50, value=4, step=1)

    st.subheader("Outputs")
    report_top_n = st.number_input("report-top-n", min_value=1, max_value=500, value=30, step=1)
    label_top_n = st.number_input("label-top-n (KML labels)", min_value=0, max_value=5000, value=60, step=5)

    run_btn = st.button("Run MayaScan", type="primary")


with right:
    st.subheader("Execution")
    cmd_box = st.empty()
    log_box = st.empty()
    status = st.empty()

    if "running" not in st.session_state:
        st.session_state.running = False
    if "last_run_dir" not in st.session_state:
        st.session_state.last_run_dir = None

    if run_btn:
        # Resolve input
        runs_dir_path = (REPO_ROOT / runs_dir).resolve() if not Path(runs_dir).is_absolute() else Path(runs_dir).resolve()
        runs_dir_path.mkdir(parents=True, exist_ok=True)

        if mode == "Upload .laz/.las":
            if uploaded is None:
                st.error("Please upload a .laz or .las file.")
                st.stop()
            # Save uploaded file into data/lidar/
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
                st.stop()

        # Build command
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

        cmd_box.code(" ".join(cmd), language="bash")
        status.info("Starting… (streaming logs below)")

        run_dir = runs_dir_path / run_name
        st.session_state.last_run_dir = str(run_dir)

        # Run subprocess and stream logs
        st.session_state.running = True
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        lines = []
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                lines.append(line.rstrip("\n"))
                # keep last ~400 lines in UI
                tail = "\n".join(lines[-400:])
                log_box.code(tail, language="text")
            if line == "" and proc.poll() is not None:
                break
            time.sleep(0.01)

        rc = proc.returncode
        st.session_state.running = False

        if rc == 0:
            status.success(f"Done ✅  Output folder: {run_dir}")
        else:
            status.error(f"Failed ❌  Exit code: {rc}  (see logs above)")

        # Show quick links + download
        if run_dir.exists():
            report_html = run_dir / "report.html"
            candidates_csv = run_dir / "candidates.csv"

            st.markdown("### Outputs")
            st.code(str(run_dir), language="text")
            if report_html.exists():
                st.write("✅ report.html created")
                st.code(str(report_html), language="text")
            if candidates_csv.exists():
                st.write("✅ candidates.csv created")
                st.code(str(candidates_csv), language="text")

            zip_path = run_dir.parent / f"{run_name}_outputs.zip"
            zip_run_dir(run_dir, zip_path)
            st.download_button(
                label="Download outputs (.zip)",
                data=zip_path.read_bytes(),
                file_name=zip_path.name,
                mime="application/zip",
            )

st.divider()
st.caption("Tip: If PDAL isn't found, install it (conda-forge is usually easiest) and ensure `pdal --version` works in your terminal.")