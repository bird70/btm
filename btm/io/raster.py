"""RasterDataset: single-band raster I/O via rasterio."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS


@dataclasses.dataclass
class RasterDataset:
    """In-memory representation of a single-band raster."""

    path: str | None
    array: np.ndarray  # 2-D, single band
    crs: CRS | None
    transform: Affine
    nodata: float | int | None
    cell_width: float
    cell_height: float

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> RasterDataset:
        """Read a single-band raster from *path* via rasterio."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Raster not found: {path}")
        with rasterio.open(str(p)) as src:
            array = src.read(1)
            return cls(
                path=str(p),
                array=array,
                crs=src.crs,
                transform=src.transform,
                nodata=src.nodata,
                cell_width=abs(src.transform.a),
                cell_height=abs(src.transform.e),
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def to_file(
        self,
        path: str,
        dtype: str | None = None,
        compress: str = "lzw",
    ) -> None:
        """Write the array to *path* as a single-band GeoTIFF (always overwrites)."""
        out_dtype = dtype or self.array.dtype.name
        profile = {
            "driver": "GTiff",
            "dtype": out_dtype,
            "width": self.array.shape[1],
            "height": self.array.shape[0],
            "count": 1,
            "crs": self.crs,
            "transform": self.transform,
            "compress": compress,
        }
        if self.nodata is not None:
            profile["nodata"] = self.nodata
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(self.array.astype(out_dtype), 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def cell_size(self) -> float:
        """Mean of horizontal and vertical cell size in CRS units."""
        return (self.cell_width + self.cell_height) / 2
