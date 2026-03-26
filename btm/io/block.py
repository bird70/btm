"""Windowed block processor using rasterio windows (no netCDF4 dependency)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import rasterio
from rasterio.windows import Window


class BlockProcessor:
    """Process a raster tile-by-tile via rasterio windowed reads/writes."""

    def process(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        src_path: str,
        dst_path: str,
        block_size: int = 512,
        overlap: int = 0,
    ) -> None:
        """Apply *func* over *src_path* and save result to *dst_path*.

        Parameters
        ----------
        func:
            Function that takes a 2-D numpy array (the tile with overlap
            padding applied) and returns a 2-D numpy array of the inner
            tile size (without border).
        src_path:
            Source raster path (any GDAL-readable format).
        dst_path:
            Destination GeoTIFF path. Always overwritten.
        block_size:
            Tile size in pixels (square tiles).
        overlap:
            Number of overlap border pixels added on each side.  The
            function receives `block_size + 2*overlap` wide/tall tiles;
            only the *inner* `block_size × block_size` portion is written.
        """
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            profile.update(driver="GTiff", compress="lzw")
            height, width = src.height, src.width
            _nodata = src.nodata  # preserved in profile; not used directly

            with rasterio.open(dst_path, "w", **profile) as dst:
                y = 0
                while y < height:
                    x = 0
                    while x < width:
                        # Inner tile bounds (clamped to raster)
                        inner_h = min(block_size, height - y)
                        inner_w = min(block_size, width - x)

                        # Read window with overlap padding (clamped)
                        read_y = max(0, y - overlap)
                        read_x = max(0, x - overlap)
                        read_h = min(height - read_y, inner_h + 2 * overlap)
                        read_w = min(width - read_x, inner_w + 2 * overlap)

                        win = Window(read_x, read_y, read_w, read_h)
                        tile = src.read(1, window=win)

                        result = func(tile)

                        # The func is expected to return the inner portion
                        # when overlap > 0.  If it returns the full tile,
                        # we need to trim the overlap border ourselves.
                        if overlap > 0:
                            pad_top = y - read_y
                            pad_left = x - read_x
                            result = result[
                                pad_top : pad_top + inner_h,
                                pad_left : pad_left + inner_w,
                            ]

                        out_win = Window(x, y, inner_w, inner_h)
                        dst.write(result, 1, window=out_win)

                        x += block_size
                    y += block_size
