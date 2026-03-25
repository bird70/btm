"""Full BTM model orchestrator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from btm.classification.reader import read_classification
from btm.core.bpi import compute_bpi
from btm.core.classify import classify_terrain
from btm.core.slope import compute_slope
from btm.core.standardize import standardize_bpi
from btm.io.raster import RasterDataset


def run_full_model(
    bathy_path: str,
    broad_bpi_inner: int,
    broad_bpi_outer: int,
    fine_bpi_inner: int,
    fine_bpi_outer: int,
    classification_file: str,
    outdir: str,
    keep_intermediates: bool = True,
    block_size: int | None = None,
) -> dict[str, str]:
    """Run the complete BTM analysis pipeline.

    Parameters
    ----------
    bathy_path:
        Path to input single-band bathymetric raster.
    broad_bpi_inner / broad_bpi_outer:
        Inner/outer radius in cells for the broad-scale BPI annulus.
    fine_bpi_inner / fine_bpi_outer:
        Inner/outer radius in cells for the fine-scale BPI annulus.
    classification_file:
        Path to classification dictionary (.csv, .xml, or .xlsx).
    outdir:
        Output directory (created if absent).
    keep_intermediates:
        If False, only ``classified_zones.tif`` is written.
    block_size:
        Reserved for future block-based processing; currently unused.

    Returns
    -------
    dict[str, str]
        Mapping of output name → absolute file path.
    """
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # 1. Read bathymetry
    bathy_ds = RasterDataset.from_file(bathy_path)
    bathy = bathy_ds.array
    nodata = bathy_ds.nodata
    cell_size = bathy_ds.cell_size()

    def _out(name: str) -> str:
        return str(outdir_path / name)

    outputs: dict[str, str] = {}

    # 2. Broad BPI
    broad_bpi = compute_bpi(
        bathy, broad_bpi_inner, broad_bpi_outer, cell_size, nodata=nodata
    )
    broad_bpi_path = _out("broad_bpi.tif")
    if keep_intermediates:
        bathy_ds.array = broad_bpi
        bathy_ds.to_file(broad_bpi_path, dtype="int32")
        outputs["broad_bpi"] = broad_bpi_path

    # 3. Fine BPI
    fine_bpi = compute_bpi(
        bathy, fine_bpi_inner, fine_bpi_outer, cell_size, nodata=nodata
    )
    fine_bpi_path = _out("fine_bpi.tif")
    if keep_intermediates:
        bathy_ds.array = fine_bpi
        bathy_ds.to_file(fine_bpi_path, dtype="int32")
        outputs["fine_bpi"] = fine_bpi_path

    # 4. Standardise broad BPI
    broad_std = standardize_bpi(
        broad_bpi.astype(np.float64), nodata=_nodata_or_none(broad_bpi, nodata)
    )
    broad_std_path = _out("broad_std.tif")
    if keep_intermediates:
        bathy_ds.array = broad_std
        bathy_ds.to_file(broad_std_path, dtype="int32")
        outputs["broad_std"] = broad_std_path

    # 5. Standardise fine BPI
    fine_std = standardize_bpi(
        fine_bpi.astype(np.float64), nodata=_nodata_or_none(fine_bpi, nodata)
    )
    fine_std_path = _out("fine_std.tif")
    if keep_intermediates:
        bathy_ds.array = fine_std
        bathy_ds.to_file(fine_std_path, dtype="int32")
        outputs["fine_std"] = fine_std_path

    # 6. Slope
    slope = compute_slope(bathy.astype(np.float64), cell_size, nodata=nodata)
    slope_path = _out("slope.tif")
    if keep_intermediates:
        bathy_ds.array = slope
        bathy_ds.to_file(slope_path, dtype="float32")
        outputs["slope"] = slope_path

    # 7. Read classification
    classes = read_classification(classification_file)

    # 8. Classify
    classified = classify_terrain(
        broad_std.astype(np.float64),
        fine_std.astype(np.float64),
        slope.astype(np.float64),
        bathy.astype(np.float64),
        classes,
        nodata=nodata,
    )

    # 9. Write classified zones (always)
    classified_path = _out("classified_zones.tif")
    bathy_ds.array = classified
    bathy_ds.nodata = 0
    bathy_ds.to_file(classified_path, dtype="int32")
    outputs["classified_zones"] = classified_path

    # Restore nodata to original
    bathy_ds.nodata = nodata

    return outputs


def _nodata_or_none(array: np.ndarray, nodata: float | int | None) -> float | None:
    """Return nodata if any cell in array equals it, else None."""
    if nodata is None:
        return None
    if np.any(array == nodata):
        return float(nodata)
    return None
