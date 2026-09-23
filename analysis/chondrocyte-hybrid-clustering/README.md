# Chondrocyte Hybrid Clustering

This directory contains only the spatial-clustering method used in the associated
manuscript. It is intentionally separate from the graphical segmentation
application in the repository root.

## Method implemented

For each Cellpose instance-label mask, the program:

1. extracts the centroid of every positive cell label;
2. identifies candidate clusters with DBSCAN (`eps=15` pixels,
   `min_samples=3`);
3. detects direct cell contact from shared edges or corners in the instance mask
   (8-neighbor connectivity);
4. splits each DBSCAN candidate into contact-connected components and retains
   only components containing at least three cells; and
5. divides the number of retained clusters by the cartilage-mask area in pixels.

The contact step is implemented from neighboring label pixels and does not make
a full-size binary copy of the image for every cell.

## Inputs

- **Instance masks:** single-channel integer masks exported by Cellpose. `0` is
  background and each cell has a unique positive label. The study files use
  names such as `cell_00000_cp_masks.png`.
- **Cartilage masks:** images with white background and non-white cartilage,
  named like `mask_00000.jpg`. To reproduce the study's area measurement, the
  program thresholds white pixels (`>254`), applies an 8 x 8 morphological
  closing operation, inverts the result, and fills holes.

Files are paired using the last numeric identifier in each filename, so the two
example names above are matched as slice `00000`.

## Installation

Python 3.10 or newer is required.

```bash
cd analysis/chondrocyte-hybrid-clustering
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
```

`requirements-lock.txt` records the package versions used for final verification
on Windows. The ranges in `pyproject.toml` allow installation on other systems.

## Run

Run the command separately for each experimental cohort:

```bash
chondrocyte-hybrid-clustering \
  --instance-mask-dir path/to/SHR/cell_masks \
  --cartilage-mask-dir path/to/SHR/cartilage_masks \
  --output-dir results/SHR
```

The paper parameters are the defaults. They may also be stated explicitly:

```bash
chondrocyte-hybrid-clustering \
  --instance-mask-dir path/to/cell_masks \
  --cartilage-mask-dir path/to/cartilage_masks \
  --output-dir results/cohort_name \
  --eps 15 \
  --min-samples 3
```

## Outputs

- `cluster_metrics.csv`: per-slice cell counts, candidate and final cluster
  counts, clustered-cell fraction, cartilage area, and clusters per area.
- `cluster_masks/*.tif`: 16-bit label images for review in Fiji/ImageJ, in which
  `0` is background and each retained cluster has a unique positive integer
  label.

`clusters_per_area_px2` is the exact ratio used by the code. The CSV also
contains `clusters_per_1e6_px2`, a rescaled value that is easier to read but does
not change the analysis.

## Tests

```bash
python -m pip install -e .[dev]
python -m pytest -q
```

The tests cover centroid extraction, 8-neighbor contact, rejection of nearby but
non-contacting cells, splitting of a DBSCAN candidate into separate contact
components, cartilage area measurement, and batch filename matching.
