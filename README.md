# MayaScan

**Automated LiDAR Archaeological Detection Pipeline (CLI + Streamlit App)**

MayaScan is an end-to-end geospatial analysis system for detecting and prioritizing potential archaeological structures (e.g., Maya mounds, platforms, terraces, and settlement patterns) from airborne LiDAR data.

The pipeline converts raw LAZ/LAS point clouds into terrain models, extracts subtle micro-topography using multi-scale relief analysis, identifies candidate anthropogenic features, ranks them, clusters them into settlement patterns, and generates GIS-ready outputs and interactive reports to help experts quickly review results.

Tested using publicly available datasets (e.g., Caracol, Belize) obtained from OpenTopography.

---

## ⚠️ Responsible Use

This project is intended for research, education, and software engineering demonstration.

- MayaScan identifies **terrain anomalies**, not confirmed archaeological sites.
- Results require **expert review and ground validation**.
- Location information should be handled responsibly to avoid site disturbance or looting.
- This repository **does not include LiDAR datasets or derived site location outputs**.
- The project intentionally avoids publishing curated site interpretations or sensitive geographic information.

This repository is shared as a **portfolio and technical demonstration**, illustrating geospatial analysis and large-scale terrain processing techniques.

---

## Project Background

Airborne LiDAR has transformed Maya archaeology by revealing landscapes hidden beneath dense tropical canopy. Modern datasets contain millions of terrain cells and thousands of subtle features.

The challenge is no longer finding features, but efficiently reviewing large areas of data.

MayaScan was built as a personal research-engineering project to:

- Automatically extract candidate structures from a LiDAR tile  
- Identify settlement clusters and spatial patterns  
- Rank features by likelihood of anthropogenic origin  
- Generate outputs that make expert review fast and intuitive  

**Optimized for**
- Low-relief tropical landscapes  
- Subtle platforms and mounds (0.3–2 m relief)  
- Rapid exploratory workflows  

---

## What You Get

### Outputs (per run)

- GeoTIFFs (DTM, LRM, density)
- CSV candidate table (`candidates.csv`)
- GeoJSON / KML exports (`candidates.geojson`, `candidates.kml`)
- Plots and histograms (`plots/`)
- Markdown + optional PDF summary (`report.md`, `report.pdf`)
- Interactive HTML report with map + cutout panels (`report.html`, `html/img/`)

### Review UX (Streamlit App)

A lightweight Streamlit UI wraps the CLI pipeline so you can:

- Upload a `.laz/.las` tile (or point to a local path)
- Tune thresholds and filters with helpful tooltips
- Run the pipeline and see live logs
- Review results in one place:
  - Interactive Leaflet map (Street + Satellite basemap toggle, no API keys)
  - Ranked candidates table
  - Candidate cutout panels (LRM + hillshade)
  - Embedded `report.html`
  - One-click downloads (CSV, GeoJSON, KML, whole-run ZIP)

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
- Slope limits
- Morphological cleanup

### 4. Shape Metrics
For each region:
- Area
- Peak / mean relief
- Extent (area / bounding box) *(compactness / “filled-ness”)*
- Aspect ratio *(elongation)*
- Width / height (meters)

### 5. Settlement Density Modeling
- Binary mound mask
- Gaussian smoothing
- Percentile thresholding

### 6. Scoring

```
score = (density^a) × (peak^b) × (extent^c) × √area
```

### 7. Spatial Clustering
- Auto-projected to UTM (meters)
- DBSCAN clustering
- Automatic epsilon selection (optional)
- Distance to cluster core

### 8. Outputs
- GeoTIFFs (DTM, LRM, density)
- CSV candidate table
- GeoJSON / KML
- Plots and histograms
- Markdown / PDF summary
- Interactive HTML report

---

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
  --min-density auto:p55 \
  --density-sigma 40 \
  --min-peak 0.50 \
  --min-area-m2 25 \
  --min-extent 0.35 \
  --max-aspect 4.0 \
  --cluster-eps auto \
  --min-samples 4 \
  --report-top-n 30 \
  --label-top-n 60
```

Single-line version:

```bash
python maya_scan.py -i data/lidar/bz_hr_las31_crs.laz --name example_run --overwrite --try-smrf --pos-thresh auto:p96 --min-density auto:p55 --density-sigma 40 --min-peak 0.50 --min-area-m2 25 --min-extent 0.35 --max-aspect 4.0 --cluster-eps auto --min-samples 4 --report-top-n 30 --label-top-n 60
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

## Quick Start (Streamlit App)

If you have the Streamlit app file in the repo (example: `app.py`), run:

```bash
streamlit run app.py
```

Then:
1. Upload a `.laz/.las` tile (or use a local path)
2. Set a run name + parameters
3. Click **Run MayaScan**
4. Review results in the **Results** tab (map, ranked table, report)

---

## LiDAR Data Sources

LiDAR data is not included in this repository.

Public datasets can be obtained from:

**OpenTopography**  
https://opentopography.org/

Example workflow:
1. Create a free OpenTopography account  
2. Search for available datasets (e.g., Belize / Caracol)  
3. Select an area of interest  
4. Download LAZ tiles  
5. Place them in:

```
data/lidar/
```

### API Access

MayaScan currently processes locally downloaded data only.  
No API key is required.

---

## Installation

### Requirements
- Python 3.10+
- **PDAL installed separately** (system install)
- (Optional) scikit-learn for DBSCAN clustering
- (Optional) reportlab for PDF report output

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Sanity checks:
- `pdal --version` should work
- `python -c "import rasterio, pyproj, scipy"` should work

---

## Key Parameters (Practical Meaning)

- `--pos-thresh auto:p96`  
  Relief threshold for candidate detection in the LRM. Higher percentile = fewer, stronger bumps.

- `--min-density auto:p55` + `--density-sigma 40`  
  Builds a smoothed “feature density” raster, suppressing isolated noise and emphasizing settlement-like zones.

- Shape cleanup filters:
  - `--min-peak` (m): drop tiny terrain wiggles
  - `--min-area-m2` (m²): drop very small patches
  - `--min-extent` (0–1): keep coherent/filled regions (area / bbox_area)
  - `--max-aspect` (≥1): drop long skinny ridge-like artifacts

- `--cluster-eps auto` + `--min-samples 4`  
  DBSCAN clustering in **meters** (auto-UTM projection if source CRS is geographic). Useful for settlement pattern grouping.

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
data/lidar/*.laz
data/lidar/*.las
data/lidar/*.tif
```

---

## Future Work

- Multi-tile regional analysis
- Linear feature detection
- ML-based classification
- Parameter auto-tuning

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