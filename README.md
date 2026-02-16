# MayaScan

**Automated LiDAR Archaeological Detection Pipeline (CLI + Streamlit)**

*From raw LiDAR to ranked archaeological targets in one run.*

---

## Table of Contents

- [Overview](#overview)
- [Responsible Use](#responsible-use)
- [Project Background](#project-background)
- [Goal-Focused Design](#goal-focused-design)
- [What You Get](#what-you-get)
- [Installation](#installation)
- [Quick Start (Streamlit App)](#quick-start-streamlit-app)
- [Quick Start (CLI)](#quick-start-cli)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [LiDAR Data Sources](#lidar-data-sources)
- [Key Parameters (Practical Meaning)](#key-parameters-practical-meaning)
- [Known Limitations](#known-limitations)
- [Technologies Used](#technologies-used)
- [AI-Assisted Development](#ai-assisted-development)
- [Skills Demonstrated](#skills-demonstrated)
- [Repository Structure](#repository-structure)
- [Future Work](#future-work)
- [License](#license)
- [Author](#author)
- [Image Credits](#image-credits)

---

## Overview

*From Visible Ruins to Hidden Landscapes*

<p align="center">
  <img src="assets/caracol_caana.png" width="700">
</p>
<p align="center">
  <em>Caana pyramid at Caracol, Belize — visible monumental architecture</em>
</p>

<p align="center">
  <img src="assets/aguada_fenix_lidar.png" width="700">
</p>
<p align="center">
  <em>LiDAR terrain model of Aguada Fénix, Mexico — large-scale Maya architecture revealed beneath forest canopy</em>
</p>

Airborne LiDAR can reveal landscape-scale archaeological features that are often invisible at ground level. The challenge is no longer discovery alone; it is prioritizing large terrain datasets for expert review.

**MayaScan** is a Python-based geospatial pipeline that converts raw LAZ/LAS point clouds into terrain models, automatically detects and ranks candidate archaeological features, and provides an integrated review workflow through both a command-line interface and a Streamlit web app.

Using multi-scale relief analysis and spatial density modeling, MayaScan highlights subtle topography, identifies potential anthropogenic structures (mounds, platforms, terraces, and settlement patterns), clusters them spatially, and generates GIS-ready outputs and interactive reports. It has been tested on publicly available OpenTopography datasets (e.g., Caracol, Belize).

### Key capabilities

- LAZ/LAS → DTM, LRM, and density surfaces
- Automated candidate detection with region-level + consensus filtering
- Scientific scoring with interpretable component terms
- Settlement clustering using DBSCAN
- In-app preset comparison (bars + baseline deltas)
- Run-quality badge + reproducibility provenance block
- Filter-waterfall diagnostics (edge/consensus/density/post/spacing)
- Candidate score explainability (component-level contribution view)
- In-app analyst labeling + label-guided precision metrics
- Runtime profiling by pipeline stage
- Interactive HTML reports with cutouts and metrics
- GIS-ready exports (CSV, GeoJSON, KML)
- Streamlit interface for end-to-end review

---

## Responsible Use

This project is intended for research, education, and software engineering demonstration.

- MayaScan identifies **terrain anomalies**, not confirmed archaeological sites.
- Results require **expert review and ground validation**.
- Location information should be handled responsibly to avoid site disturbance or looting.
- This repository **does not include LiDAR datasets or derived site location outputs**.
- The project intentionally avoids publishing curated site interpretations or sensitive geographic information.

This repository is shared as a **portfolio and technical demonstration** of geospatial terrain-processing workflows.

---

## Project Background

Airborne LiDAR has transformed Maya archaeology by revealing landscapes hidden beneath dense tropical canopy. The challenge now is prioritizing millions of terrain cells and subtle features for expert review.

MayaScan was built to:

- Automatically extract candidate structures from a LiDAR tile  
- Identify settlement clusters and spatial patterns  
- Rank features by likelihood of anthropogenic origin  
- Generate outputs that make expert review fast and intuitive  

**Optimized for**
- Low-relief tropical landscapes  
- Subtle platforms and mounds (0.3–2 m relief)  
- Rapid exploratory workflows  

---

## Goal-Focused Design

The core project goal is reducing review time while keeping likely anthropogenic targets near the top. MayaScan focuses on that through:
- Region-level filtering instead of centroid-only decisions (more stable candidate behavior)
- Multi-threshold consensus support to suppress one-threshold artifacts
- Shape + physics constraints (prominence, compactness, solidity, extent, aspect, area bounds)
- Edge exclusion + spacing de-dup to reduce tile artifacts and duplicate nearby detections
- Analyst-in-the-loop labeling and score diagnostics to improve review quality over repeated runs

---

## What You Get

### Outputs (per run)

- GeoTIFFs (DTM, LRM, density)
- CSV candidate table (`candidates.csv`)
- GeoJSON / KML exports (`candidates.geojson`, `candidates.kml`)
- Reproducibility + diagnostics metadata (`run_params.json`, including candidate accounting + stage runtimes)
- Plots and histograms (`plots/`)
- Markdown + optional PDF summary (`report.md`, `report.pdf`)
- Optional interactive HTML report with map + cutout panels (`report.html`, `html/img/`)
- Optional analyst labels file from the app (`candidate_labels.csv`)
- Optional preset-comparison artifacts (`*_preset_compare_*.json`, `*_preset_compare_*.md`)

### Review UX (Streamlit App)

A lightweight Streamlit UI wraps the CLI pipeline so you can upload a `.laz/.las` tile (or use a local path), tune thresholds with tooltips, run the pipeline with live logs, and review results in one place:
- Preset profiles for reproducible runs (`Strict`, `Balanced`, `Exploratory`)
- Side-by-side preset comparison summary (`.json` + `.md`) from inside the app
- Dedicated **Comparison** tab with visual bars + deltas vs baseline preset
- Run-quality badge with explicit heuristic checks
- Copyable/downloadable provenance block for reproducibility
- Filter waterfall panel (edge/consensus/density/post-filter/spacing drop counts)
- Automated parameter tuning hints from waterfall outcomes
- Candidate score breakdown panel (component-level contribution view)
- Candidate-detail analyst labels (`likely` / `unlikely` / `unknown`) + notes
- Optional portfolio mode to hide diagnostics-heavy sections
- Interactive Leaflet map (Street + Satellite basemap toggle, no API keys)
- Ranked candidates table
- Candidate cutout panels (LRM + hillshade)
- Embedded `report.html` (when enabled)
- Download actions for CSV, GeoJSON, KML, and whole-run ZIP

---

## Example Results

*Sample outputs from a typical MayaScan run.*

> WIP: Demo screenshots (`assets/demo_streamlit.png`, `assets/demo_report.png`) will be added soon.

Typical results include:
- Ranked candidate features prioritized by score
- Spatial clustering highlighting settlement patterns
- Multi-scale terrain visualizations (LRM, hillshade, density)

---

## Installation

### Requirements
- Python 3.10+ (requirements currently specify minimums)
- **PDAL installed separately** (system install)
- `scikit-learn` for DBSCAN clustering (installed by default via `requirements.txt`)
- `reportlab` for PDF report output (installed by default via `requirements.txt`)
- `matplotlib` for plots and cutout rendering (installed by default via `requirements.txt`)

Python package minimums in `requirements.txt`:
- `numpy>=1.23`, `scipy>=1.9`, `pandas>=1.5`
- `rasterio>=1.3`, `pyproj>=3.4`, `shapely>=2.0`
- `scikit-learn>=1.2`, `matplotlib>=3.6`, `reportlab>=3.6`, `streamlit>=1.30`

**PDAL must be installed separately.** Example installs:

- macOS: `brew install pdal`
- Ubuntu: `sudo apt install pdal`
- Windows (conda): `conda install -c conda-forge pdal`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Sanity checks:
- `pdal --version` should work
- `python -c "import rasterio, pyproj, scipy"` should work

---

## Quick Start (Streamlit App)

If you have the Streamlit app file in the repo (example: `app.py`), run:

```bash
# Run the Streamlit app
streamlit run app.py
```

Then:
1. Upload a `.laz/.las` tile (or use a local path)
2. Pick a scientific preset (`Balanced` recommended) and set run name
3. Click **Run MayaScan**
4. Review map, ranked candidates, filter waterfall, and score breakdown in **Results**
5. (Optional) Run **Preset comparison** and review deltas in the **Comparison** tab
6. (Optional) Add analyst labels (`likely` / `unlikely` / `unknown`) to track triage outcomes over time.

Outputs are saved to:

```
runs/<run_name>/
```

## Quick Start (CLI)

Place a LiDAR tile locally (example):

```
data/lidar/bz_hr_las31_crs.laz
```

(LiDAR data is not included in this repository.)

Run MayaScan:

```bash
python maya_scan.py \
  -i data/lidar/bz_hr_las31_crs.laz \
  --name example_run \
  --overwrite \
  --try-smrf \
  --pos-thresh auto:p96 \
  --min-density auto:p60 \
  --density-sigma 40 \
  --max-slope-deg 20 \
  --consensus-percentiles 95,96,97 \
  --consensus-min-support 2 \
  --consensus-radius-m 12 \
  --min-peak 0.50 \
  --min-area-m2 25 \
  --max-area-m2 1200 \
  --min-extent 0.38 \
  --max-aspect 3.5 \
  --edge-buffer-m 10 \
  --min-spacing-m 15 \
  --min-prominence 0.10 \
  --min-compactness 0.12 \
  --min-solidity 0.50 \
  --cluster-eps auto \
  --min-samples 4 \
  --report-top-n 30 \
  --label-top-n 60
```

Outputs will be written to:

```
runs/example_run/
```

Open the interactive report:

```
runs/example_run/report.html
```

---

## End-to-End Pipeline

### 1. Ground Model (DTM)
- PDAL converts LAZ/LAS → raster DTM
- Optional SMRF ground classification
- 1 m resolution GeoTIFF

### 2. Multi-scale Local Relief Model (LRM)
- Small-scale smoothing (σ = 1–2)
- Large-scale smoothing (σ = 8–16)
- Difference (small − large)

Highlights subtle anthropogenic topography.

### 3. Region Detection
Connected-component extraction with filters:
- Minimum size (pixel count)
- Region slope limit (q75 of slope values in each region)
- Morphological cleanup

### 4. Consensus Support (Optional but enabled by default)
- Candidate extraction can run across multiple positive-relief percentiles (default: `95,96,97`)
- Regions are matched across runs by center distance
- Candidates can be dropped unless they meet minimum support (default: 2)

### 5. Shape + Terrain Metrics
For each region:
- Area
- Peak / mean relief
- Local prominence (region mean relief minus surrounding ring mean)
- Extent (area / bounding box) *(compactness / “filled-ness”)*
- Aspect ratio *(elongation)*
- Compactness (`4πA/P²`)
- Solidity (`Area / ConvexHullArea`)
- Width / height (meters)

### 6. Settlement Density Modeling
- Binary mound mask
- Gaussian smoothing
- Percentile thresholding
- Region-level density statistics (mean/q75) per candidate region

### 7. Post-filters + De-duplication
- Region mean density gate
- Shape/physics gates (`min_peak`, `min_area`, `max_area`, `min_extent`, `max_aspect`, `min_prominence`, `min_compactness`, `min_solidity`)
- Edge buffer exclusion
- Score-ordered spacing de-duplication (local non-maximum suppression in meters)

### 8. Scoring

```
score = (density^a) × (peak^b) × (extent^c) × (consensus_support^d) × (prominence^e) × (compactness^f) × (solidity^g) × (area^h)
```

Where `density` is the **region mean density** (not a single centroid pixel).
`consensus_support` is how many threshold runs support a region.

### 9. Spatial Clustering
- Meter-based coordinates for clustering/distances (projected CRS is used directly with unit→meter conversion when needed; geographic CRS auto-projects to UTM)
- DBSCAN clustering
- Automatic epsilon selection (optional)
- Distance to cluster core

### 10. Outputs + Diagnostics
Produces the run artifacts listed in [What You Get](#what-you-get).

---


## LiDAR Data Sources

Public datasets can be obtained from:

**OpenTopography**  
<https://opentopography.org/>

Example workflow:
1. Create a free OpenTopography account  
2. Find a dataset and select an area of interest (e.g., Belize / Caracol)  
3. Download LAZ tiles and place them in:

```
data/lidar/
```

### API Access

MayaScan currently processes only locally downloaded data.  
No API key is required.

---

## Key Parameters (Practical Meaning)

- `--pos-thresh auto:p96`  
  Relief threshold for candidate detection in the LRM. Higher percentile = fewer, stronger bumps.

- `--min-density auto:p60` + `--density-sigma 40`  
  Builds a smoothed “feature density” raster, suppressing isolated noise and emphasizing settlement-like zones. Candidate gating/scoring use **region mean density**.

- `--max-slope-deg 20` (default)  
  Uses the **75th percentile slope (q75)** over each region footprint to reject steep/noisy terrain artifacts.

- Shape cleanup filters:
  - `--min-peak` (m): drop tiny terrain wiggles
  - `--min-area-m2` (m²): drop very small patches
  - `--max-area-m2` (m²): drop very large merged terrain blobs (`0` disables)
  - `--min-extent` (0–1): keep coherent/filled regions (area / bbox_area)
  - `--max-aspect` (≥1): drop long skinny ridge-like artifacts
  - `--edge-buffer-m` (m): drop regions near tile boundaries to reduce edge-cut artifacts
  - `--min-spacing-m` (m): score-ordered de-dup radius between nearby candidate centers
  - `--min-prominence` (m): drop features that do not stand out from local background ring
  - `--min-compactness` (0–1): drop line-like regions (`4πA/P²`)
  - `--min-solidity` (0–1): drop fragmented/irregular regions (`A / hull_area`)

- `--cluster-eps auto` + `--min-samples 4`  
  DBSCAN clustering in **meters** (projected CRS units are converted to meters when needed; geographic CRS auto-projects to UTM). Useful for settlement pattern grouping.

- Consensus controls (`--consensus-percentiles`, `--consensus-min-support`, `--consensus-radius-m`, `--no-consensus`)  
  Reduce threshold-specific one-offs by favoring regions repeatedly detected across nearby percentiles.

- Score exponents (`--score-extent-exp`, `--score-consensus-exp`, `--score-prominence-exp`, `--score-compactness-exp`, `--score-solidity-exp`, `--score-area-exp`)  
  Control how strongly each component influences rank ordering.

### Score Interpretation

- Treat score as a **within-run prioritization metric**, not a probability of archaeology.
- In practice, combine score with geometry/context checks (support, prominence, compactness/solidity, cluster context).
- Higher `consensus_support` and strong shape metrics generally indicate better review priority than isolated high area alone.

---

## Known Limitations

- MayaScan flags terrain anomalies, not confirmed archaeology; expert interpretation is required.
- Candidate scores are for **relative ranking within a run**, not calibrated archaeological probabilities.
- Consensus filtering can suppress isolated true positives if configured too strictly.
- False positives can increase in rugged terrain, modern earthworks, or heavily modified agricultural zones.
- Newer region-level filtering can produce fewer candidates and lower absolute scores than centroid-based filtering; this is expected and often improves ranking stability.
- Performance depends on point-cloud quality, ground classification quality, and chosen thresholds.
- Analyst labels in the app are review metadata, not ground-truth training labels.
- Current workflow is primarily tuned for single-tile or tile-at-a-time exploratory analysis.

---

## Technologies Used

Python · NumPy · SciPy · Rasterio · PyProj  
PDAL · GeoTIFF · UTM reprojection  
Scikit-learn (DBSCAN)  
Matplotlib · Leaflet · ReportLab  
Streamlit (UI wrapper)

---

## AI-Assisted Development

Large language models were used to assist with:
- Architecture exploration
- Debugging geospatial workflows
- Rapid prototyping
- Documentation refinement

All design decisions, validation, and interpretation were performed manually.

---

## Skills Demonstrated

- End-to-end data pipeline design  
- Large-scale raster processing  
- Coordinate systems and spatial analysis  
- Density modeling and clustering  
- Multi-scale signal processing  
- Automated reporting and visualization  
- Building a review UI (Streamlit) for fast analyst workflows

---

## Repository Structure

Tracked in this repository (typical):

```
MayaScan/
├── maya_scan.py          # CLI pipeline
├── app.py                # Streamlit UI (runs the CLI, renders results)
├── assets/
│   ├── mayascan_logo.svg
│   ├── caracol_caana.png
│   └── aguada_fenix_lidar.png
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    └── lidar/
        └── .gitkeep
```

Generated locally (gitignored):

```
runs/
runs/*/candidate_labels.csv
runs/*/run_params.json
runs/*_preset_compare_*.json
runs/*_preset_compare_*.md
data/lidar/*.laz
data/lidar/*.las
data/lidar/*.tif
```

---

## Future Work

- Multi-tile regional analysis
- Linear feature detection
- ML-based classification
- Automated parameter tuning from filter diagnostics and analyst labels

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Author

**James Adelhelm**  
Software Developer — Data Ingestion, AccuWeather  

MayaScan is an independent personal research project and is **not affiliated with AccuWeather**.

**Professional focus**
- Scala development for operational weather data systems  
- High-volume ingestion of global meteorological alerts  
- Reliable cloud-based data processing pipelines  

**Personal research interests**
- Mesoamerican archaeology and Maya history  
- LiDAR-based settlement analysis  
- Landscape-scale interpretation  

This project reflects a personal interest in Maya history and explores how modern software engineering can be applied to large-scale archaeological terrain analysis.

---

## Image Credits

**Caana, Caracol (Belize)**  
Photo by Devon Jones — Wikimedia Commons  
License: CC BY-SA 3.0  
<https://commons.wikimedia.org/wiki/File:Caracol-Temple.jpg>

**Aguada Fénix LiDAR**  
Courtesy of Takeshi Inomata — Wikimedia Commons  
License: CC BY-SA 4.0  
<https://commons.wikimedia.org/wiki/File:Aguada_F%C3%A9nix_1.jpg>
