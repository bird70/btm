"""README for reference output files used in numerical parity tests.

Reference outputs are generated from BTM 3.0 (ArcGIS Pro) using the
Fagatele Bay test dataset with these parameters:

    bathy_input : tests/data/bathy5m_clip.tif
    broad_inner : 10   (cells)
    broad_outer : 30   (cells)
    fine_inner  : 1    (cells)
    fine_outer  : 5    (cells)
    classdict   : tests/data/fagatelebay.csv

To generate (requires ArcGIS Pro + BTM 3.0 toolbox):

    1. Run the Full BTM Model tool in ArcGIS Pro with the above parameters.
    2. Export each output raster to NumPy with:

        import arcpy, numpy as np
        for name, path in [("broad_bpi_ref", "broad_bpi.tif"), ...]:
            arr = arcpy.RasterToNumPyArray(path)
            np.save(f"tests/data/reference/{name}.npy", arr)

Expected files:
broad_bpi_ref.npy
fine_bpi_ref.npy
broad_std_ref.npy
fine_std_ref.npy
slope_ref.npy
classified_zones_ref.npy
"""
