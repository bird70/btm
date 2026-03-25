# CLI Contract: btm-bpi

`python -m btm.bpi` | `btm-bpi`

**Purpose**: Compute a Bathymetric Position Index (BPI) raster from a bathymetric raster.

## Synopsis

```
python -m btm.bpi
    --bathy PATH
    --inner INT
    --outer INT
    --output PATH
    [--scale {broad,fine}]
    [--verbose | --quiet]
    [--log-file PATH]
```

## Parameters

| Parameter       | Type           | Required | Description                   |
| --------------- | -------------- | -------- | ----------------------------- |
| `--bathy PATH`  | file path      | ✅       | Input bathymetric raster      |
| `--inner INT`   | int ≥ 1        | ✅       | Annulus inner radius in cells |
| `--outer INT`   | int > inner    | ✅       | Annulus outer radius in cells |
| `--output PATH` | file path      | ✅       | Output GeoTIFF path           |
| `--scale`       | `broad`/`fine` | ❌       | Label only; default `broad`   |

## Output

Single-band integer GeoTIFF. Values: `round(bathy − focal_annulus_mean)`.

## Exit Codes

| Code | Meaning                  |
| ---- | ------------------------ |
| 0    | Success                  |
| 1    | Invalid parameters       |
| 2    | Input file not found     |
| 5    | Unexpected runtime error |
