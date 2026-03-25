# CLI Contract: btm-vrm

`python -m btm.vrm` | `btm-vrm`

**Purpose**: Compute the Vector Ruggedness Measure (Sappington et al. 2007).

## Synopsis

```
python -m btm.vrm
    --bathy PATH
    --neighborhood INT
    --output PATH
    [--verbose | --quiet]
    [--log-file PATH]
```

## Parameters

| Parameter            | Type        | Required | Description                               |
| -------------------- | ----------- | -------- | ----------------------------------------- |
| `--bathy PATH`       | file path   | ✅       | Input bathymetric raster                  |
| `--neighborhood INT` | odd int ≥ 3 | ✅       | Square neighbourhood side length in cells |
| `--output PATH`      | file path   | ✅       | Output VRM GeoTIFF path                   |

## Output

Single-band float32 GeoTIFF. Values in [0, 1]; 0 = flat, ~1 = extremely rugged.

## Exit Codes

| Code | Meaning                  |
| ---- | ------------------------ |
| 0    | Success                  |
| 1    | Invalid parameters       |
| 2    | Input file not found     |
| 5    | Unexpected runtime error |
