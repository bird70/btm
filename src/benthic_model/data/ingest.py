from __future__ import annotations

from pathlib import Path

_ALLOWED_EXTENSIONS = frozenset(
    {".csv", ".tif", ".tiff", ".md", ".json", ".yaml", ".yml"}
)
_ALLOWED_NAMES = frozenset({"METADATA.MD", "METADATA.md"})
_ALLOWED_SUBDIRS = frozenset({"MBES", "mbes", "btm_rasters"})

_REQUIRED_METADATA_FIELDS = frozenset({"title", "description", "source", "license"})


def enforce_competition_data_whitelist(
    target: str | Path,
    data_root: str | Path,
) -> Path:
    """Check that *target* is an allowed competition artifact under *data_root*.

    Allowed artifacts:
        - Any ``.csv`` or ``.tif`` / ``.tiff`` file directly in *data_root*
        - Files in the ``MBES/`` or ``btm_rasters/`` subdirectory of *data_root*
        - The metadata file ``METADATA.MD``

    Returns the resolved absolute path on success.

    Raises:
        ValueError: when the file is outside the allowed path set.
    """
    target = Path(target).resolve()
    data_root = Path(data_root).resolve()

    # Must be within data_root
    try:
        rel = target.relative_to(data_root)
    except ValueError:
        raise ValueError(
            f"{target} is not an allowed competition artifact: "
            f"it is not under the data root {data_root}."
        )

    parts = rel.parts
    # Allow METADATA.MD directly in data_root
    if len(parts) == 1 and parts[0].upper() == "METADATA.MD":
        return target

    # Allow files directly in data_root with known extensions
    if len(parts) == 1 and target.suffix.lower() in _ALLOWED_EXTENSIONS:
        return target

    # Allow files one level deep in an allowed subdirectory
    if (
        len(parts) == 2
        and parts[0] in _ALLOWED_SUBDIRS
        and target.suffix.lower() in _ALLOWED_EXTENSIONS
    ):
        return target

    raise ValueError(
        f"{target} is not an allowed competition artifact. "
        "Only CSV, TIF, and YAML files within the data directory and its "
        "recognised subdirectories (MBES/, btm_rasters/) are permitted."
    )


def parse_and_validate_metadata(metadata_path: str | Path) -> dict[str, str]:
    """Parse a ``METADATA.MD`` file and validate required fields.

    Supports two formats:
    - Simple ``Key: Value`` pairs (one per line)
    - Markdown with ``# Title`` headings, ``## Section`` headers, and
      ``- Key: Value`` bullet items

    Returns a dict mapping lowercase field names to their string values.

    Raises:
        ValueError: when a required metadata field is absent.
    """
    path = Path(metadata_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed: dict[str, str] = {}

    # First pass: parse all "Key: Value" and "- Key: Value" lines
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Strip leading bullet marker if present
        if stripped.startswith("- ") or stripped.startswith("* "):
            stripped = stripped[2:]
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip().lower()
            value = value.strip()
            # Only store entries with a non-empty key and value
            if key and value:
                parsed.setdefault(key, value)

    # Second pass: extract values from markdown-style structure for missing fields

    # Title: extract from first top-level heading (# Title, not ## Section)
    if "title" not in parsed:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                parsed["title"] = stripped[2:].strip()
                break

    # License / funding(→source) / description: scan section context
    current_section: str | None = None
    _SECTION_TO_FIELD = {
        "license": "license",
        "funding": "source",
        "source": "source",
        "description": "description",
    }
    for line in lines:
        stripped = line.strip()
        # Detect level-2 section headers (## but not ###)
        if stripped.startswith("## ") and not stripped.startswith("### "):
            current_section = stripped[3:].strip().lower()
            continue
        if current_section in _SECTION_TO_FIELD:
            target_key = _SECTION_TO_FIELD[current_section]
            if target_key not in parsed:
                # Accept bare bullet items (no colon) as the section value
                bullet = stripped
                if bullet.startswith("- ") or bullet.startswith("* "):
                    bullet = bullet[2:].strip()
                if bullet and not bullet.startswith("#"):
                    parsed[target_key] = bullet

    # Fallback: use title as description if still missing
    if "description" not in parsed and "title" in parsed:
        parsed["description"] = parsed["title"]

    missing = _REQUIRED_METADATA_FIELDS - parsed.keys()
    if missing:
        raise ValueError(
            f"METADATA.MD is missing required field(s): {sorted(missing)}. "
            f"Found: {sorted(parsed.keys())}"
        )

    return parsed
