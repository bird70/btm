"""BTM feature extraction for ML pipelines.

Exports BTM terrain derivatives (BPI, slope, VRM, surface ratio) as a
point-sampled DataFrame that is directly compatible with the ``benthic_model``
feature engineering pipeline.

Typical usage
-------------
::

    from btm.features.extract import extract_btm_features

    # sample_points: DataFrame with columns ID, x, y (geographic CRS)
    btm_feats = extract_btm_features(
        sample_points,
        bathymetry_tif="data/bathy.tif",
        broad_inner=10, broad_outer=30,
        fine_inner=1,  fine_outer=5,
    )
    # btm_feats has original columns + btm_broad_std, btm_fine_std,
    # btm_slope, btm_vrm, btm_surface_ratio, ...

    # Merge with benthic_model's extract_mbes_features output
    combined = mbes_feats.merge(btm_feats.drop(columns=["x", "y"]), on="ID")
"""

from btm.features.extract import extract_btm_features, sample_raster_at_points

__all__ = ["extract_btm_features", "sample_raster_at_points"]
