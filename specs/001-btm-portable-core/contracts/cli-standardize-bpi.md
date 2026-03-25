# CLI Contract: btm-standardize-bpi

`python -m btm.standardize_bpi` | `btm-standardize-bpi`

**Purpose**: Standardize a BPI raster to a dimensionless z-score × 100.

## Synopsis

```
python -m btm.standardize_bpi
    --bpi PATH
    --output PATH
    [--verbose | --quiet]
    [--log-file PATH]
```

## Parameters

| Parameter       | Type      | Required | Description                          |
| --------------- | --------- | -------- | ------------------------------------ |
| `--bpi PATH`    | file path | ✅       | Input BPI raster                     |
| `--output PATH` | file path | ✅       | Output standardised BPI GeoTIFF path |

## Output

Single-band int32 GeoTIFF. Values: `round((bpi − mean) / std × 100)`.

## Exit Codes

| Code | Meaning                  |
| ---- | ------------------------ |
| 0    | Success                  |
| 1    | Invalid parameters       |
| 2    | Input file not found     |
| 5    | Unexpected runtime error |
