"""Classification entry point: dispatch to CSV/XML/XLSX readers."""

from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class ClassEntry:
    """One classification class row."""

    code: int
    name: str
    depth_lower: float | None = None
    depth_upper: float | None = None
    slope_lower: float | None = None
    slope_upper: float | None = None
    broad_bpi_lower: float | None = None
    broad_bpi_upper: float | None = None
    fine_bpi_lower: float | None = None
    fine_bpi_upper: float | None = None


def read_classification(path: str) -> list[ClassEntry]:
    """Read a classification dictionary from *path*.

    Dispatches to the appropriate reader based on file extension.
    Supports .csv, .xml, and .xlsx.

    Raises
    ------
    TypeError
        If the file extension is not supported.
    ValueError
        If the file is malformed or missing required columns.
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        from btm.classification.csv_reader import read_csv

        return read_csv(path)
    elif ext == ".xml":
        from btm.classification.xml_reader import read_xml

        return read_xml(path)
    elif ext in (".xlsx", ".xls"):
        from btm.classification.xlsx_reader import read_xlsx

        return read_xlsx(path)
    else:
        raise TypeError(f"Unsupported classification file type: {ext!r}")
