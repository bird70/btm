"""Tests for btm.classification readers (CSV, XML, XLSX)."""

import pathlib

import openpyxl
import pytest

from btm.classification.reader import ClassEntry, read_classification

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fagatelebay_xlsx_path(tmp_path):
    """Create a minimal XLSX matching the fagatelebay CSV data."""
    wb = openpyxl.Workbook()
    ws = wb.active
    # Header row 1 (skipped by reader)
    ws.append(["BTM Classification Dictionary"])
    # Header row 2 (column names)
    ws.append(
        [
            "Class",
            "Zone",
            "SSB_LowerBounds",
            "SSB_UpperBounds",
            "LSB_LowerBounds",
            "LSB_UpperBounds",
            "Slope_LowerBounds",
            "Slope_UpperBounds",
            "Depth_LowerBounds",
            "Depth_UpperBounds",
        ]
    )
    # Data rows — mirror fagatelebay.csv classes
    ws.append([1, "Reef Crest", None, -100, None, -100, None, None, None, None])
    ws.append([2, "Mid-Slope Ridges", None, -100, -100, 100, None, None, None, None])
    ws.append([3, "Back Reef", None, -100, 100, None, None, None, None, None])
    p = tmp_path / "test.xlsx"
    wb.save(str(p))
    return str(p)


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------
class TestCsvReader:
    def test_returns_class_entries(self, fagatelebay_csv_path):
        classes = read_classification(fagatelebay_csv_path)
        assert isinstance(classes, list)
        assert len(classes) > 0
        assert all(isinstance(c, ClassEntry) for c in classes)

    def test_first_class_code(self, fagatelebay_csv_path):
        classes = read_classification(fagatelebay_csv_path)
        assert classes[0].code == 1

    def test_first_class_name(self, fagatelebay_csv_path):
        classes = read_classification(fagatelebay_csv_path)
        assert classes[0].name  # non-empty

    def test_numeric_bounds(self, fagatelebay_csv_path):
        classes = read_classification(fagatelebay_csv_path)
        for cls in classes:
            for attr in (
                "depth_lower",
                "depth_upper",
                "slope_lower",
                "slope_upper",
                "broad_bpi_lower",
                "broad_bpi_upper",
                "fine_bpi_lower",
                "fine_bpi_upper",
            ):
                val = getattr(cls, attr)
                assert val is None or isinstance(val, float)

    def test_missing_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("Class,Zone\n1,Foo\n")
        with pytest.raises(ValueError, match="column"):
            read_classification(str(p))

    def test_malformed_csv_raises(self, tmp_path):
        p = tmp_path / "malformed.csv"
        # Only 3 columns, not 10
        p.write_text("Class,Zone,Extra\nfoo,bar,baz\n")
        with pytest.raises(ValueError):
            read_classification(str(p))


# ---------------------------------------------------------------------------
# XML reader
# ---------------------------------------------------------------------------
class TestXmlReader:
    def test_returns_class_entries(self, fagatelebay_xml_path):
        classes = read_classification(fagatelebay_xml_path)
        assert isinstance(classes, list)
        assert len(classes) > 0
        assert all(isinstance(c, ClassEntry) for c in classes)

    def test_class_codes_unique(self, fagatelebay_xml_path):
        classes = read_classification(fagatelebay_xml_path)
        codes = [c.code for c in classes]
        assert len(codes) == len(set(codes))


# ---------------------------------------------------------------------------
# XLSX reader
# ---------------------------------------------------------------------------
class TestXlsxReader:
    def test_returns_class_entries(self, fagatelebay_xlsx_path):
        classes = read_classification(fagatelebay_xlsx_path)
        assert isinstance(classes, list)
        assert len(classes) > 0
        assert all(isinstance(c, ClassEntry) for c in classes)

    def test_first_entry(self, fagatelebay_xlsx_path):
        classes = read_classification(fagatelebay_xlsx_path)
        assert classes[0].code == 1
