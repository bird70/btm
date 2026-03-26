"""XLSX classification dictionary reader using openpyxl (replaces xlrd)."""

from __future__ import annotations

from pathlib import Path

import openpyxl

from btm.classification.reader import ClassEntry

_EXPECTED_COLS = 10


def _to_float(val) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def read_xlsx(path: str) -> list[ClassEntry]:
    """Read a BTM classification XLSX file.

    The workbook must have data in the first sheet.  The first data row after
    the last header row must contain a numeric Class code in column A.  Any
    leading rows that cannot be parsed as a class code are skipped as headers.

    Column order (positional):
        Class, Zone, SSB_Lower, SSB_Upper, LSB_Lower, LSB_Upper,
        Slope_Lower, Slope_Upper, Depth_Lower, Depth_Upper

    Raises
    ------
    ValueError
        If the sheet contains no parseable data rows.
    """
    wb = openpyxl.load_workbook(str(Path(path)), read_only=True, data_only=True)
    ws = wb.worksheets[0]

    entries: list[ClassEntry] = []
    for row in ws.iter_rows(values_only=True):
        # Skip header / empty rows — first column must be a number
        if not row or row[0] is None:
            continue
        try:
            code = int(float(row[0]))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue  # header or blank row

        if len(row) < _EXPECTED_COLS:
            raise ValueError(
                f"XLSX row for class {row[0]} has {len(row)} columns; "
                f"expected {_EXPECTED_COLS}."
            )

        (
            _,
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

        entries.append(
            ClassEntry(
                code=code,
                name=str(zone).strip() if zone is not None else "",
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

    wb.close()
    if not entries:
        raise ValueError(f"No parseable data rows found in XLSX: {path!r}")
    return entries
