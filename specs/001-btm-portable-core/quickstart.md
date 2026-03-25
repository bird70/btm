# Quickstart: BTM Portable Core

## Prerequisites

- Python 3.11+
- Git

## 1. Install

```bash
git clone https://github.com/EsriOceans/btm.git
cd btm
pip install -e ".[dev]"
```

## 2. Verify installation

```bash
python -m btm.run_model --help
```

## 3. Run the full BTM pipeline (CLI)

```bash
python -m btm.run_model \
  --bathy tests/data/bathy5m_clip.tif \
  --broad-inner 2 --broad-outer 20 \
  --fine-inner 1 --fine-outer 5 \
  --classdict tests/data/fagatelebay.csv \
  --outdir /tmp/btm_output
```

Expected outputs in `/tmp/btm_output/`:

```
broad_bpi.tif       fine_bpi.tif
broad_std.tif       fine_std.tif
slope.tif           classified_zones.tif
```

## 4. Use as a Python library

```python
import numpy as np
import rasterio
from btm.io.raster import RasterDataset
from btm.core.bpi import compute_bpi
from btm.core.slope import compute_slope
from btm.core.vrm import compute_vrm
from btm.core.model import run_full_model

# --- Individual algorithm ---
ds = RasterDataset.from_file("tests/data/bathy5m_clip.tif")
bpi = compute_bpi(ds.array, inner_radius=1, outer_radius=5,
                   cell_size=ds.cell_size(), nodata=ds.nodata)

# --- Full pipeline ---
run_full_model(
    bathy_path="tests/data/bathy5m_clip.tif",
    broad_bpi_inner=2, broad_bpi_outer=20,
    fine_bpi_inner=1,  fine_bpi_outer=5,
    classification_file="tests/data/fagatelebay.csv",
    outdir="/tmp/btm_output",
    keep_intermediates=True,
)
```

## 5. Run tests (no GIS runtime required)

```bash
pytest -m "not arcgis and not qgis" --cov=btm/core --cov-report=term-missing
```

## 6. Run with ArcGIS Pro

1. Open ArcGIS Pro ≥ 3.2.
2. In the Catalog pane → Toolboxes → Add Toolbox → navigate to `Install/toolbox/btm.pyt`.
3. Expand **Benthic Terrain Modeler** → run any tool.

> The `.pyt` toolbox delegates all computation to `btm.core` via `btm.adapters.arcgis`.
> The `btm` package must be on the ArcGIS Pro Python environment's `sys.path`.
> Install via: `conda run -n arcgispro-py3 pip install -e .` from the repo root.

## 7. Troubleshooting

| Symptom                                  | Fix                                                                                                   |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: btm` in ArcGIS Pro | Run `conda run -n arcgispro-py3 pip install -e .`                                                     |
| All cells classified as 0 (unclassified) | Check classification dictionary thresholds; run `btm-bpi` individually to inspect intermediate values |
| Memory error on large raster             | Add `--block-size 512` to CLI (or pass `block_size=512` to `run_full_model`) to reduce tile size      |
