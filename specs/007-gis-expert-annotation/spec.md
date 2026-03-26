# Feature Specification: Human-in-the-Loop Habitat Mapping

**Feature Branch**: `007-gis-expert-annotation`  
**Created**: 2026-03-27  
**Status**: Draft  
**Input**: User description: "Human-in-the-loop habitat mapping: GIS-ready spatial analysis layers with expert annotation workflow for ML/DL refinement"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Automated Feature Engineering & GIS Layer Preparation (Priority: P1)

A marine scientist wants to visually inspect the spatial structure of benthic habitat classification before committing to a final ML prediction. The system generates GIS-ready layers (GeoTIFF rasters and GeoPackage vector layers) containing:

- Input rasters (bathymetry, backscatter) in their native CRS
- Derived products: depth zones, backscatter intensity zones, combined acoustic facies, kriging variogram surfaces, slope/curvature derivatives
- Spatial block/cluster boundaries as vector polygons with editable annotation fields
- Training point locations with class labels and prediction confidence

These outputs are placed in a structured directory that can be opened directly in QGIS.

**Why this priority**: Without GIS-ready outputs, the expert cannot perform visual analysis. This is the foundation for the entire workflow.

**Independent Test**: Open the output directory in QGIS; all layers load with correct CRS (EPSG:28355), symbology is interpretable, and the analyst can identify spatial patterns in the backscatter/bathymetry.

**Acceptance Scenarios**:

1. **Given** the MBES bathymetry and backscatter rasters and training CSV, **When** the preparation script runs, **Then** a directory `outputs/gis_layers/` contains GeoTIFF rasters and a GeoPackage with vector layers, all in EPSG:28355.
2. **Given** the output GeoPackage, **When** opened in QGIS, **Then** each vector layer includes an editable `expert_label` field (initially NULL) and a `confidence` field.
3. **Given** the output rasters, **When** inspected in QGIS, **Then** the depth zones, backscatter zones, and combined acoustic facies zones are clearly distinguishable as categorical rasters.

---

### User Story 2 - Expert Visual Annotation (Priority: P2)

The marine scientist opens the GIS layers in QGIS and visually annotates the spatial regions by editing the `expert_label` field in the polygon layer. The analyst compares the automated segmentation boundaries with the visible zonation in backscatter/bathymetry imagery and:

- Confirms regions where automated zones align with visible habitat boundaries
- Splits or merges zones where the automated segmentation is incorrect
- Assigns habitat interpretation labels (e.g., "hard reef", "sand plain", "mixed rubble") to zones based on domain knowledge
- Flags uncertain areas for further investigation

**Why this priority**: The expert annotation step is the core value proposition -- combining automated spatial analysis with human domain expertise.

**Independent Test**: Open the GeoPackage in QGIS, edit `expert_label` for several polygons, save, and re-open to confirm edits persist. Export as CSV to verify the annotation data is accessible.

**Acceptance Scenarios**:

1. **Given** the analyst has labelled regions in the GeoPackage, **When** the annotated file is saved, **Then** the `expert_label` column contains the analyst's interpretations alongside the original automated zone IDs.
2. **Given** some zones are split or merged by the analyst, **When** the annotation is exported, **Then** the new polygon geometries are preserved and associated with the correct labels.

---

### User Story 3 - ML/DL Refinement with Expert Annotations (Priority: P3)

After the expert returns the annotated GeoPackage, a refinement script ingests the expert labels as additional training signal. The system:

- Reads expert polygon annotations and assigns expert zone labels to training/test points
- Uses expert labels as features or stratification boundaries for kriging/GBDT
- Optionally trains a U-Net or similar spatial model on raster patches, seeded by expert annotations
- Produces a refined Kaggle submission and reproducibility report

**Why this priority**: This closes the loop -- expert knowledge improves the ML model, which should outperform either pure-ML or pure-visual approaches.

**Independent Test**: Run the refinement script with the annotated GeoPackage; the output submission has higher spatial-block CV weighted-F1 than the non-annotated baseline.

**Acceptance Scenarios**:

1. **Given** an annotated GeoPackage with expert labels, **When** the refinement script runs, **Then** it produces a Kaggle-format submission CSV and a reproducibility report in `reports/`.
2. **Given** the refined model, **When** evaluated by spatial-block CV, **Then** the weighted-F1 score is documented in the report alongside the baseline score.

---

### Edge Cases

- What happens when the analyst edits only part of the zones? The refinement script must handle partially-annotated data gracefully (NULL expert labels treated as "no override").
- How does the system handle if the analyst changes polygon geometries (splits/merges)? The refinement script uses spatial joins rather than zone IDs to associate points with expert labels.
- What if no GeoPackage annotation is returned? The pipeline falls back to fully-automated prediction (same as v3).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST produce GeoTIFF rasters for: depth zones (categorical), backscatter intensity zones (categorical), combined acoustic facies (categorical), slope, curvature, and VRM derivatives
- **FR-002**: System MUST produce a GeoPackage containing vector layers: zone polygons (with `zone_id`, `zone_type`, `mean_depth`, `mean_backscatter`, `dominant_class`, `confidence`, `expert_label` fields), training points (with class, predicted class, kriging probability), and test points (with predicted class)
- **FR-003**: All spatial outputs MUST use EPSG:28355 (GDA94 MGA Zone 55) coordinate reference system
- **FR-004**: The `expert_label` field in zone polygons MUST be editable (string type, initially NULL)
- **FR-005**: System MUST produce a reproducibility report for each submission, documenting: git revision, blend weights, spatial CV score, feature set, and method description
- **FR-006**: The refinement script MUST accept an annotated GeoPackage and produce a Kaggle-format submission CSV
- **FR-007**: Output directory structure MUST follow: `outputs/gis_layers/{rasters/, vectors/, reports/}`
- **FR-008**: Reports MUST be produced in Markdown format consistent with the `benthic-terrain-model-kaggle` repo convention (metrics JSON + reproducibility MD)

### Key Entities

- **Acoustic Facies Zone**: A contiguous spatial region with homogeneous backscatter/depth properties, represented as a polygon with zone metadata and an editable expert label
- **Reproducibility Report**: A structured document recording the full provenance of a submission (code version, configuration, scores, method description)
- **Submission Artefact**: A Kaggle-format CSV (ID, class) with associated metrics and provenance trail

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All GIS layers load correctly in QGIS with no CRS warnings or projection issues
- **SC-002**: At least 80% of the study area is covered by zone polygons suitable for expert annotation
- **SC-003**: The expert annotation workflow (open, label, save) can be completed for all zones in under 30 minutes
- **SC-004**: The expert-refined submission achieves at least the same Kaggle weighted-F1 as the best automated approach (>= 0.764)
- **SC-005**: Every submission is accompanied by a reproducibility report that enables re-creation of results from the same inputs

## Assumptions

- The analyst has access to QGIS 3.x (LTR) and is familiar with basic GIS operations (attribute editing, layer styling)
- The MBES rasters (bathymetry.tif, backscatter.tif) are in EPSG:28355 and cover the full study area
- The competition data (train.csv, test.csv) use the same CRS as the rasters
- Deep learning (U-Net) is a candidate for Step 3 refinement but not mandatory -- tree-ensemble methods with expert zone features are the primary approach
- The benthic-terrain-model-kaggle repo report structure (metrics JSON + reproducibility MD) is adopted for consistency
- Fiona and GDAL Python bindings are available via rasterio/geopandas for GeoPackage I/O
