"""XML classification dictionary reader (port of BtmXmlDocument from legacy utils.py)."""

from __future__ import annotations

from pathlib import Path
from xml.dom.minidom import parse

from btm.classification.reader import ClassEntry


def _to_float(val: str | None) -> float | None:
    if val is None or val.strip() == "":
        return None
    return float(val.strip())


def _get_text(node) -> str | None:
    """Return concatenated text content of a DOM node, or None if empty."""
    text = ""
    for child in node.childNodes:
        if child.nodeType == child.TEXT_NODE:
            text += child.nodeValue
    return text.strip() or None


def _get_field(rec, tag: str) -> str | None:
    nodes = rec.getElementsByTagName(tag)
    if not nodes:
        return None
    return _get_text(nodes[0])


def read_xml(path: str) -> list[ClassEntry]:
    """Read a BTM XML classification file.

    The file format is the ClassDict/Classifications/ClassRec structure used
    by BTM v3.0.

    Raises
    ------
    ValueError
        If the XML structure is unrecognisable.
    """
    dom = parse(str(Path(path)))
    class_recs = dom.getElementsByTagName("ClassRec")
    if not class_recs:
        raise ValueError(f"No <ClassRec> elements found in {path!r}")

    entries: list[ClassEntry] = []
    for rec in class_recs:
        code_str = _get_field(rec, "Class")
        name_str = _get_field(rec, "Zone")
        if code_str is None or name_str is None:
            raise ValueError("ClassRec missing <Class> or <Zone> element.")
        entries.append(
            ClassEntry(
                code=int(code_str),
                name=name_str,
                broad_bpi_lower=_to_float(_get_field(rec, "SSB_LowerBounds")),
                broad_bpi_upper=_to_float(_get_field(rec, "SSB_UpperBounds")),
                fine_bpi_lower=_to_float(_get_field(rec, "LSB_LowerBounds")),
                fine_bpi_upper=_to_float(_get_field(rec, "LSB_UpperBounds")),
                slope_lower=_to_float(_get_field(rec, "Slope_LowerBounds")),
                slope_upper=_to_float(_get_field(rec, "Slope_UpperBounds")),
                depth_lower=_to_float(_get_field(rec, "Depth_LowerBounds")),
                depth_upper=_to_float(_get_field(rec, "Depth_UpperBounds")),
            )
        )
    return entries
