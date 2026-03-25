# CLI Contract: btm-depth-statistics

`python -m btm.depth_statistics` | `btm-depth-stats`

**Purpose**: Compute focal depth statistics over a bathymetric raster.

## Synopsis

```
python -m btm.depth_statistics
    --bathy PATH
    --neighborhood INT
    --outdir PATH
    --stats STAT [STAT ...]
    [--window {rectangle,circle}]
    [--verbose | --quiet]
    [--log-file PATH]
```

## Parameters

| Parameter            | Type                 | Required | Description                                                    |
| -------------------- | -------------------- | -------- | -------------------------------------------------------------- |
| `--bathy PATH`       | file path            | ✅       | Input bathymetric raster                                       |
| `--neighborhood INT` | int ≥ 3              | ✅       | Neighbourhood size in cells                                    |
| `--outdir PATH`      | directory            | ✅       | Output directory for stat rasters                              |
| `--stats STAT [...]` | one or more          | ✅       | Stats to compute: `mean`, `std`, `variance`, `iqr`, `kurtosis` |
| `--window`           | `rectangle`/`circle` | ❌       | Neighbourhood shape (default `rectangle`)                      |

## Outputs

One GeoTIFF per requested statistic, named `{bathy_stem}_{stat}_{nnn}.tif` where `nnn` is the zero-padded neighbourhood size.

## Exit Codes

| Code | Meaning                                 |
| ---- | --------------------------------------- |
| 0    | Success                                 |
| 1    | Invalid parameters or unknown stat name |
| 2    | Input file not found                    |
| 5    | Unexpected runtime error                |
