# Testing with Real Data

This guide walks you through running the BTM pipeline on the bundled
**Fagatele Bay** dataset — a real 5 m bathymetric survey of a coral reef
system in American Samoa. No ArcGIS, no extra downloads.

---

## 1. Prerequisites

Make sure the virtual environment is active and the package is installed:

```powershell
# From the repo root
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -e ".[dev]"          # if not already installed
```

Verify the CLI is available:

```powershell
btm-run-model --help
```

---

## 2. The test dataset

| File                          | Description                                    |
| ----------------------------- | ---------------------------------------------- |
| `tests/data/bathy5m_clip.tif` | 5 m GeoTIFF bathymetry clipped to Fagatele Bay |
| `tests/data/fagatelebay.csv`  | 11-class coral-reef classification dictionary  |
| `tests/data/fagatelebay.xml`  | Same dictionary in XML format                  |
| `tests/data/fagatelebay.xlsx` | Same dictionary in XLSX format                 |

The classification dictionary defines 11 benthic terrain classes
(Reef Crest, Mid-Slope Ridges, Back Reef, Upper Slopes, etc.) using ranges
of broad BPI, fine BPI, slope, and depth.

---

## 3. Run the full pipeline

```powershell
btm-run-model `
  --bathy     tests\data\bathy5m_clip.tif `
  --classdict tests\data\fagatelebay.csv `
  --broad-inner 10 --broad-outer 30 `
  --fine-inner   1 --fine-outer   5 `
  --outdir    outputs\fagatelebay
```

On Linux/macOS use `\` → `/` and backtick → backslash for line continuation.

This produces six GeoTIFF files in `outputs/fagatelebay/`:

| Output file            | Contents                                        |
| ---------------------- | ----------------------------------------------- |
| `broad_bpi.tif`        | Broad-scale BPI (inner 10, outer 30 cells)      |
| `fine_bpi.tif`         | Fine-scale BPI (inner 1, outer 5 cells)         |
| `broad_std.tif`        | Standardised broad BPI (z-score × 100)          |
| `fine_std.tif`         | Standardised fine BPI                           |
| `slope.tif`            | Slope in degrees                                |
| `classified_zones.tif` | Final terrain classification (class codes 1–11) |

The command prints a summary on completion:

```
broad_bpi     → outputs\fagatelebay\broad_bpi.tif
fine_bpi      → outputs\fagatelebay\fine_bpi.tif
broad_std     → outputs\fagatelebay\broad_std.tif
fine_std      → outputs\fagatelebay\fine_std.tif
slope         → outputs\fagatelebay\slope.tif
classified_zones → outputs\fagatelebay\classified_zones.tif
```

---

## 4. Inspect the results

Run the summary script for a quick statistical overview in the terminal:

```powershell
python scripts\run_fagatelebay.py --outdir outputs\fagatelebay
```

Expected output (values will vary slightly):

```
=== BTM output summary ===

broad_bpi        min=-78   max= 57   mean=  0.1  nodata=0
fine_bpi         min=-32   max= 28   mean=  0.0  nodata=0
broad_std        min=-97   max= 70   mean=  0.2  nodata=0
fine_std         min=-99   max= 87   mean=  0.0  nodata=0
slope            min=  0.0 max= 74.1 mean=  8.3  nodata=0

=== Classified zones distribution ===
Code  Zone                  Pixels    %
   1  Reef Crest               NNN   X.X
   2  Mid-Slope Ridges         NNN   X.X
   ...
  (unclassified = code 0)
```

Run with `--help` for options (e.g. `--classdict` to print class names from a
different classification file).

---

## 5. Visualise the outputs

The output GeoTIFFs are standard single-band rasters with LZW compression.
Open them in any GIS:

**QGIS**: `Layer → Add Layer → Add Raster Layer` → browse to
`outputs/fagatelebay/classified_zones.tif`. Right-click → Properties →
Symbology → Paletted/Unique Values → Classify.

**ArcGIS Pro**: Add Data → browse to the `.tif` files. Use Classify or Unique
Values renderer on `classified_zones.tif`.

**Python** (quick check):

```python
import rasterio, numpy as np

with rasterio.open("outputs/fagatelebay/classified_zones.tif") as src:
    arr = src.read(1)
    codes, counts = np.unique(arr, return_counts=True)
    for code, count in zip(codes, counts):
        pct = count / arr.size * 100
        print(f"  class {code:2d}: {count:6d} pixels ({pct:4.1f}%)")
```

---

## 6. Run individual algorithms

Each algorithm is also available as a standalone CLI command.

**BPI only:**

```powershell
btm-bpi `
  --input  tests\data\bathy5m_clip.tif `
  --inner  10 --outer 30 `
  --output outputs\broad_bpi_only.tif
```

**Slope only:**

```powershell
btm-slope `
  --input  tests\data\bathy5m_clip.tif `
  --output outputs\slope_only.tif
```

**Classify from pre-computed grids:**

```powershell
btm-classify `
  --broad-std outputs\fagatelebay\broad_std.tif `
  --fine-std  outputs\fagatelebay\fine_std.tif `
  --slope     outputs\fagatelebay\slope.tif `
  --classdict tests\data\fagatelebay.csv `
  --output    outputs\classified_only.tif
```

---

## 7. Run the automated test suite

The integration tests run the same pipeline programmatically and assert
correctness properties:

```powershell
pytest tests/unit/ tests/integration/ -m "not arcgis and not qgis" -v
```

The four `test_numerical_parity` tests are **skipped** unless reference `.npy`
files exist in `tests/data/reference/`. See `tests/data/reference/README.md`
for instructions on generating them from BTM 3.0 (requires ArcGIS Pro).

---

## 8. Using a different classification file

BTM supports three classification file formats:

```powershell
# CSV (default demo format)
btm-run-model ... --classdict tests\data\fagatelebay.csv

# XML
btm-run-model ... --classdict tests\data\fagatelebay.xml

# XLSX
btm-run-model ... --classdict tests\data\fagatelebay.xlsx
```

All three contain the same 11 classes for Fagatele Bay.

To use your own data, prepare a CSV with these 10 columns:

```
Class, Zone, BroadBPI_Lower, BroadBPI_Upper, FineBPI_Lower, FineBPI_Upper,
Slope_Lower, Slope_Upper, Depth_Lower, Depth_Upper
```

Empty cells mean "no constraint" for that field.

---

## 9. Performance notes

The full pipeline on `bathy5m_clip.tif` (~500 × 500 cells) completes in
under 5 seconds. For larger datasets (millions of cells), the `--block-size`
option (reserved for future tiled processing) will be enabled in a future
release.
