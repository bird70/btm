"""CSV classification dictionary reader."""

from __future__ import annotations

import csv
from pathlib import Path

from btm.classification.reader import ClassEntry

# Expected minimum number of columns (positional, header ignored)
_EXPECTED_COLS = 10
_REQUIRED_HEADERS = {
    "Class",
    "Zone",
}


def _to_float(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def read_csv(path: str) -> list[ClassEntry]:
    """Read a BTM classification CSV file.

    Columns (positional, must match this order):
        Class, Zone, BroadBPI_Lower, BroadBPI_Upper, FineBPI_Lower,
        FineBPI_Upper, Slope_Lower, Slope_Upper, Depth_Lower, Depth_Upper

    OR (legacy XML-style header names are also accepted):
        Class, Zone, SSB_LowerBounds, SSB_UpperBounds, LSB_LowerBounds,
        LSB_UpperBounds, Slope_LowerBounds, Slope_UpperBounds,
        Depth_LowerBounds, Depth_UpperBounds

    Raises
    ------
    ValueError
        If the file is missing required columns or has malformed rows.
    """
    p = Path(path)
    with open(p, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            dialect = "excel"
            has_header = True

        reader = csv.reader(fh, dialect)
        header: list[str] | None = None
        if has_header:
            header = [h.strip() for h in next(reader)]
            # Validate that at minimum "Class" and "Zone" columns are present
            if not _REQUIRED_HEADERS.issubset(set(header)):
                missing = _REQUIRED_HEADERS - set(header)
                raise ValueError(
                    f"CSV missing required column(s): {', '.join(sorted(missing))}"
                )
            if len(header) < _EXPECTED_COLS:
                raise ValueError(
                    f"CSV must have at least {_EXPECTED_COLS} columns, "
                    f"found {len(header)}"
                )

        rows = [row for row in reader if row]

    if not rows:
        raise ValueError("CSV file contains no data rows.")

    entries: list[ClassEntry] = []
    for i, row in enumerate(rows, start=2):
        if len(row) < _EXPECTED_COLS:
            raise ValueError(
                f"Row {i} has {len(row)} columns; expected {_EXPECTED_COLS}."
            )
        (
            class_code,
            zone,
            broad_lower,
            broad_upper,
            fine_lower,
            fine_upper,
            slope_lower,
            slope_upper,
            depth_lower,
            depth_upper,
        ) = row[:10]
        try:
            entries.append(
                ClassEntry(
                    code=int(float(class_code)),
                    name=str(zone).strip(),
                    broad_bpi_lower=_to_float(broad_lower),
                    broad_bpi_upper=_to_float(broad_upper),
                    fine_bpi_lower=_to_float(fine_lower),
                    fine_bpi_upper=_to_float(fine_upper),
                    slope_lower=_to_float(slope_lower),
                    slope_upper=_to_float(slope_upper),
                    depth_lower=_to_float(depth_lower),
                    depth_upper=_to_float(depth_upper),
                )
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Malformed data in row {i}: {exc}") from exc

    return entries
