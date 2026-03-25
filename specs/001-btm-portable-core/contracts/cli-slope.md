# CLI Contract: btm-slope

`python -m btm.slope` | `btm-slope`

**Purpose**: Compute slope in degrees using the Horn (1981) 3×3 finite-difference method.

## Synopsis

```
python -m btm.slope
    --bathy PATH
    --output PATH
    [--verbose | --quiet]
    [--log-file PATH]
```

## Parameters

| Parameter       | Type      | Required | Description               |
| --------------- | --------- | -------- | ------------------------- |
| `--bathy PATH`  | file path | ✅       | Input bathymetric raster  |
| `--output PATH` | file path | ✅       | Output slope GeoTIFF path |

## Output

Single-band float32 GeoTIFF. Values in degrees [0, 90].

## Exit Codes

| Code | Meaning                  |
| ---- | ------------------------ |
| 0    | Success                  |
| 1    | Invalid parameters       |
| 2    | Input file not found     |
| 5    | Unexpected runtime error |
