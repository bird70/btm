# CLI Contract: btm-run-model

`python -m btm.run_model` | `btm-run-model`

**Purpose**: Run the complete BTM pipeline — BPI (broad + fine), standardize, slope, classify — against a bathymetric raster.

## Synopsis

```
python -m btm.run_model
    --bathy PATH
    --broad-inner INT
    --broad-outer INT
    --fine-inner INT
    --fine-outer INT
    --classdict PATH
    --outdir PATH
    [--no-intermediates]
    [--verbose | --quiet]
    [--log-file PATH]
```

## Parameters

| Parameter            | Type              | Required | Description                                              |
| -------------------- | ----------------- | -------- | -------------------------------------------------------- |
| `--bathy PATH`       | file path         | ✅       | Input single-band bathymetric raster (any GDAL format)   |
| `--broad-inner INT`  | int ≥ 1           | ✅       | Inner radius (cells) for broad-scale BPI annulus         |
| `--broad-outer INT`  | int > broad-inner | ✅       | Outer radius (cells) for broad-scale BPI annulus         |
| `--fine-inner INT`   | int ≥ 1           | ✅       | Inner radius (cells) for fine-scale BPI annulus          |
| `--fine-outer INT`   | int > fine-inner  | ✅       | Outer radius (cells) for fine-scale BPI annulus          |
| `--classdict PATH`   | file path         | ✅       | Classification dictionary (.csv, .xml, or .xlsx)         |
| `--outdir PATH`      | directory path    | ✅       | Output directory (created if absent)                     |
| `--no-intermediates` | flag              | ❌       | Write only `classified_zones.tif`; discard intermediates |
| `--verbose`          | flag              | ❌       | Set log level to DEBUG                                   |
| `--quiet`            | flag              | ❌       | Set log level to WARNING                                 |
| `--log-file PATH`    | file path         | ❌       | Also write log records to this file                      |

## Outputs (default — all intermediates kept)

```
{outdir}/broad_bpi.tif
{outdir}/fine_bpi.tif
{outdir}/broad_std.tif
{outdir}/fine_std.tif
{outdir}/slope.tif
{outdir}/classified_zones.tif
```

## Exit Codes

| Code | Meaning                                         |
| ---- | ----------------------------------------------- |
| 0    | Success                                         |
| 1    | Invalid parameters (e.g., inner ≥ outer radius) |
| 2    | Input file not found or unreadable              |
| 3    | Classification dictionary parse error           |
| 4    | No valid classes produced (empty result)        |
| 5    | Unexpected runtime error                        |
