# CLI Contract: btm-classify

`python -m btm.classify` | `btm-classify`

**Purpose**: Classify benthic terrain using pre-computed derivative rasters and a classification dictionary.

## Synopsis

```
python -m btm.classify
    --broad-std PATH
    --fine-std PATH
    --slope PATH
    --bathy PATH
    --classdict PATH
    --output PATH
    [--verbose | --quiet]
    [--log-file PATH]
```

## Parameters

| Parameter          | Type      | Required | Description                                   |
| ------------------ | --------- | -------- | --------------------------------------------- |
| `--broad-std PATH` | file path | ✅       | Standardised broad-scale BPI raster           |
| `--fine-std PATH`  | file path | ✅       | Standardised fine-scale BPI raster            |
| `--slope PATH`     | file path | ✅       | Slope raster (degrees)                        |
| `--bathy PATH`     | file path | ✅       | Original bathymetric raster                   |
| `--classdict PATH` | file path | ✅       | Classification dictionary (.csv, .xml, .xlsx) |
| `--output PATH`    | file path | ✅       | Output classified raster GeoTIFF path         |

## Output

Single-band int32 GeoTIFF. Values are class codes from the dictionary; 0 = unclassified.

## Exit Codes

| Code | Meaning                               |
| ---- | ------------------------------------- |
| 0    | Success                               |
| 1    | Invalid parameters                    |
| 2    | Input file not found                  |
| 3    | Classification dictionary parse error |
| 4    | No valid classes produced             |
| 5    | Unexpected runtime error              |
