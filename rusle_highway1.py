"""
================================================================================
RUSLE Soil Erosion Risk Mapping — Highway 1 Corridor, California
================================================================================
Author:      [Your Name]
Date:        [Date]
Description: This script computes a spatially distributed RUSLE erosion risk
             map (A = R × K × LS × C) along the Highway 1 corridor in 
             Central California at 10m resolution using:
               - Sentinel-2 L2A imagery (pre-downloaded .SAFE folders)
               - USGS 3DEP DEM tiles (pre-downloaded GeoTIFFs)
               - OpenStreetMap Highway 1 geometry (pre-downloaded GeoJSON)

Requirements:
    - QGIS Python environment (GDAL, OGR, processing, qgis.core)
    - scipy (pip install scipy --break-system-packages)
    - NumPy (included with QGIS)

Usage:
    Run from the QGIS Python console, or via the QGIS Python executable:
    "C:/Program Files/QGIS 3.44.9/apps/Python312/python.exe" rusle_highway1.py

Inputs (edit CONFIGURATION block below):
    SENTINEL_DIR   — folder containing Sentinel-2 .SAFE subfolders
    DEM_TILES      — list of pre-downloaded DEM GeoTIFF file paths
    OSM_HWY1       — path to Highway 1 GeoJSON (OpenStreetMap export)
    OUT_DIR        — output directory for all intermediate and final rasters

Outputs:
    mosaic_B04.tif          — Sentinel-2 Band 4 (Red) mosaic
    mosaic_B08.tif          — Sentinel-2 Band 8 (NIR) mosaic
    ndvi.tif                — NDVI raster
    C_factor.tif            — Cover management factor
    dem_merged_wgs84.tif    — Merged DEM in WGS84
    dem_utm10n_10m.tif      — DEM reprojected to UTM Zone 10N at 10m
    slope_deg.tif           — Slope in degrees
    LS_factor.tif           — Slope length and steepness factor
    R_factor.tif            — Rainfall erosivity factor (uniform)
    K_factor.tif            — Soil erodibility factor (uniform)
    C_factor_aligned.tif    — C factor resampled to DEM grid
    hwy1_buffer_utm.gpkg    — 2km Highway 1 corridor buffer (UTM 10N)
    RUSLE_final.tif         — Final clipped RUSLE erosion risk raster
================================================================================
"""

import os
import sys
import glob
import json
import numpy as np
from scipy.ndimage import uniform_filter
from osgeo import gdal, ogr, osr

# Ensure GDAL exceptions are raised rather than returning None silently
gdal.UseExceptions()

# ================================================================================
# CONFIGURATION — Edit these paths before running
# ================================================================================

# Directory containing Sentinel-2 .SAFE folders
SENTINEL_DIR = r"C:\Users\YourName\ErosionRisk\Sentinel"

# List of pre-downloaded DEM GeoTIFF file paths (USGS 3DEP tiles)
DEM_TILES = [
    r"C:\Users\YourName\ErosionRisk\DEM\dem_tile_1.tif",
    r"C:\Users\YourName\ErosionRisk\DEM\dem_tile_2.tif",
    r"C:\Users\YourName\ErosionRisk\DEM\dem_tile_3.tif",
    r"C:\Users\YourName\ErosionRisk\DEM\dem_tile_4.tif",
]

# Path to Highway 1 GeoJSON (from OpenStreetMap)
OSM_HWY1 = r"C:\Users\YourName\ErosionRisk\hwy1_lines.geojson"

# Output directory — all results will be written here
OUT_DIR = r"C:\Users\YourName\ErosionRisk\outputs"

# RUSLE factor values
R_VALUE = 250.0   # Rainfall erosivity (MJ·mm / ha·h·yr) — CA Central Coast estimate
K_VALUE = 0.28    # Soil erodibility (t·h / MJ·mm) — coastal range approximation
# Note: Replace with spatially distributed PRISM R and SSURGO K for improved accuracy

# Corridor buffer distance in degrees (~2km at this latitude)
BUFFER_DEGREES = 0.018

# Target CRS
TARGET_EPSG = 32610   # UTM Zone 10N
TARGET_RES  = 10      # metres

# ================================================================================
# SETUP
# ================================================================================

os.makedirs(OUT_DIR, exist_ok=True)

def path(filename):
    """Helper: return full output path for a given filename."""
    return os.path.join(OUT_DIR, filename)


def copy_grid_properties(reference_path):
    """
    Read spatial properties (geotransform, projection, cols, rows)
    from a reference raster. Used to create matching output rasters.
    """
    ds = gdal.Open(reference_path)
    gt   = ds.GetGeoTransform()
    proj = ds.GetProjection()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ds = None
    return gt, proj, cols, rows


def write_float32(array, reference_path, output_path, nodata=-9999.0):
    """
    Write a NumPy float32 array to a compressed GeoTIFF, using the
    spatial properties of an existing reference raster.
    """
    gt, proj, cols, rows = copy_grid_properties(reference_path)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        output_path, cols, rows, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(array.astype(np.float32))
    ds.FlushCache()
    ds = None
    print(f"  Written: {output_path}")


print("=" * 72)
print("RUSLE Erosion Risk Mapping — Highway 1, California")
print("=" * 72)

# ================================================================================
# STEP 1 — Discover Sentinel-2 Band Files
# ================================================================================
# Sentinel-2 L2A .SAFE folders follow this structure:
#   <TILE>.SAFE/GRANULE/<granule_id>/IMG_DATA/R10m/<tile>_B04_10m.jp2
# We search for all B04 and B08 10m files recursively.

print("\n[Step 1] Locating Sentinel-2 Band 4 and Band 8 files...")

b04_files = sorted(glob.glob(
    os.path.join(SENTINEL_DIR, "**", "*_B04_10m.jp2"), recursive=True
))
b08_files = sorted(glob.glob(
    os.path.join(SENTINEL_DIR, "**", "*_B08_10m.jp2"), recursive=True
))

if not b04_files or not b08_files:
    sys.exit(
        "ERROR: No Sentinel-2 Band 4 or Band 8 files found.\n"
        "Check that SENTINEL_DIR points to a folder containing .SAFE subfolders\n"
        "and that the imagery includes 10m bands."
    )

print(f"  Found {len(b04_files)} B04 file(s) and {len(b08_files)} B08 file(s)")
for f in b04_files:
    print(f"    B04: {os.path.basename(f)}")
for f in b08_files:
    print(f"    B08: {os.path.basename(f)}")

# ================================================================================
# STEP 2 — Mosaic Sentinel-2 Bands
# ================================================================================
# Multiple tiles are merged into a single seamless raster using gdal.Warp.
# This handles any overlap between adjacent tiles automatically.

print("\n[Step 2] Mosaicking Sentinel-2 bands...")

for band_name, band_files, out_name in [
    ("B04", b04_files, "mosaic_B04.tif"),
    ("B08", b08_files, "mosaic_B08.tif"),
]:
    out_path = path(out_name)
    if os.path.exists(out_path):
        print(f"  {out_name} already exists, skipping.")
        continue

    print(f"  Mosaicking {band_name} ({len(band_files)} tiles)...")
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        srcNodata=0,
        dstNodata=0,
        resampleAlg="bilinear",
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    )
    gdal.Warp(out_path, band_files, options=warp_opts)
    print(f"  Done: {out_name}")

# ================================================================================
# STEP 3 — Compute NDVI
# ================================================================================
# NDVI = (NIR - Red) / (NIR + Red)
# A small epsilon (0.0001) is added to the denominator to prevent division by zero
# in pixels where both bands are zero (water, shadow, cloud).

print("\n[Step 3] Computing NDVI...")

b04_ds  = gdal.Open(path("mosaic_B04.tif"))
b08_ds  = gdal.Open(path("mosaic_B08.tif"))
b04_arr = b04_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
b08_arr = b08_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
b04_ds  = None
b08_ds  = None

ndvi = (b08_arr - b04_arr) / (b08_arr + b04_arr + 0.0001)

# Mask invalid pixels (where both bands are zero — ocean/NoData)
ndvi[b04_arr == 0] = -9999.0

write_float32(ndvi, path("mosaic_B04.tif"), path("ndvi.tif"))

# ================================================================================
# STEP 4 — Derive C Factor from NDVI
# ================================================================================
# The Cover Management Factor (C) is derived using the exponential decay
# relationship of van der Knijff et al. (1999):
#
#     C = exp(-2.5 × NDVI)
#
# Values are clamped to [0.05, 1.0]:
#   - C = 1.0 : bare soil, no vegetation protection
#   - C = 0.05: dense vegetation, maximum protection
#
# Reference: van der Knijff, J.M., Jones, R.J.A., & Montanarella, L. (1999).
#            Soil Erosion Risk Assessment in Italy. European Soil Bureau, JRC.

print("\n[Step 4] Deriving C factor from NDVI...")

valid_mask = ndvi != -9999.0
c_factor   = np.full(ndvi.shape, -9999.0, dtype=np.float32)
c_factor[valid_mask] = np.clip(
    np.exp(-2.5 * ndvi[valid_mask]), 0.05, 1.0
)

write_float32(c_factor, path("mosaic_B04.tif"), path("C_factor.tif"))

# ================================================================================
# STEP 5 — Merge and Reproject DEM
# ================================================================================
# DEM tiles are merged into a single raster and reprojected to UTM Zone 10N
# at 10m resolution to match the Sentinel-2 grid.
#
# A 5×5 uniform smoothing filter (scipy.ndimage.uniform_filter) is applied
# after reprojection to reduce tile boundary seam artifacts that would otherwise
# appear as grid lines in the slope and LS factor outputs.

print("\n[Step 5] Merging and reprojecting DEM...")

# 5a. Merge tiles in WGS84
dem_wgs_path = path("dem_merged_wgs84.tif")
if not os.path.exists(dem_wgs_path):
    print("  Merging DEM tiles...")
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        srcNodata=-9999,
        dstNodata=-9999,
        resampleAlg="bilinear",
        creationOptions=["COMPRESS=LZW", "TILED=YES"]
    )
    gdal.Warp(dem_wgs_path, DEM_TILES, options=warp_opts)
    print("  DEM tiles merged.")
else:
    print("  dem_merged_wgs84.tif already exists, skipping.")

# 5b. Reproject to UTM Zone 10N at 10m
dem_utm_path = path("dem_utm10n_10m.tif")
if not os.path.exists(dem_utm_path):
    print("  Reprojecting DEM to UTM Zone 10N at 10m...")
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        srcSRS="EPSG:4326",
        dstSRS=f"EPSG:{TARGET_EPSG}",
        xRes=TARGET_RES,
        yRes=TARGET_RES,
        srcNodata=-9999,
        dstNodata=-9999,
        resampleAlg="bilinear",
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    )
    gdal.Warp(dem_utm_path, dem_wgs_path, options=warp_opts)
    print("  DEM reprojected.")
else:
    print("  dem_utm10n_10m.tif already exists, skipping.")

# 5c. Apply smoothing to remove tile boundary seams
dem_smooth_path = path("dem_utm10n_smooth.tif")
if not os.path.exists(dem_smooth_path):
    print("  Applying 5×5 smoothing filter to DEM...")
    ds  = gdal.Open(dem_utm_path)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nd  = ds.GetRasterBand(1).GetNoDataValue()
    ds  = None

    mask     = arr == nd
    arr[mask] = 0.0
    smoothed  = uniform_filter(arr, size=5).astype(np.float32)
    smoothed[mask] = -9999.0

    write_float32(smoothed, dem_utm_path, dem_smooth_path)
    print("  DEM smoothing complete.")
else:
    print("  dem_utm10n_smooth.tif already exists, skipping.")

# ================================================================================
# STEP 6 — Compute Slope
# ================================================================================
# Slope in degrees is derived from the smoothed DEM using GDAL's built-in
# slope algorithm. Edge pixels are computed to avoid NoData borders.

print("\n[Step 6] Computing slope in degrees...")

slope_path = path("slope_deg.tif")
if not os.path.exists(slope_path):
    dem_options = gdal.DEMProcessingOptions(
        format="GTiff",
        computeEdges=True,
        alg="Horn",        # Horn (1981) — standard neighbourhood algorithm
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    )
    gdal.DEMProcessing(slope_path, dem_smooth_path, "slope", options=dem_options)
    print("  Slope computed.")
else:
    print("  slope_deg.tif already exists, skipping.")

# ================================================================================
# STEP 7 — Compute LS Factor
# ================================================================================
# The LS factor combines slope length (L) and slope steepness (S) into a
# single dimensionless topographic factor. The simplified formulation of
# Moore and Burch (1986) is used:
#
#     LS = (cell_size / 22.13)^0.4 × (sin(θ) / 0.0896)^1.3
#
# where:
#   - cell_size = 10m (raster resolution)
#   - 22.13m    = standard RUSLE plot length reference
#   - 0.0896    = sin(5.143°), the standard RUSLE plot slope reference
#   - θ         = slope angle in radians
#
# Slope values are clipped to [0.01°, 89°] to prevent numerical errors
# at perfectly flat (0°) or vertical (90°) pixels.
#
# Note: This slope-only formulation does not account for upslope contributing
# area (flow accumulation). The Desmet and Govers (1996) approach, which
# incorporates flow routing, would provide improved accuracy in convergent
# terrain but requires hydrological preprocessing of the DEM.
#
# Reference: Moore, I.D. & Burch, G.J. (1986). Physical basis of the
#            length-slope factor in the Universal Soil Loss Equation.
#            Soil Science Society of America Journal, 50(5), 1294–1298.

print("\n[Step 7] Computing LS factor (Moore & Burch 1986)...")

ls_path = path("LS_factor.tif")
if not os.path.exists(ls_path):
    ds        = gdal.Open(slope_path)
    slope_arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nd        = ds.GetRasterBand(1).GetNoDataValue()
    ds        = None

    valid     = (slope_arr != nd) & (slope_arr >= 0)
    ls        = np.full(slope_arr.shape, -9999.0, dtype=np.float32)

    slope_clipped      = np.clip(slope_arr[valid], 0.01, 89.0)
    slope_rad          = np.deg2rad(slope_clipped)
    ls[valid]          = (
        (TARGET_RES / 22.13) ** 0.4
        * (np.sin(slope_rad) / 0.0896) ** 1.3
    )

    write_float32(ls, slope_path, ls_path)
    print("  LS factor computed.")
else:
    print("  LS_factor.tif already exists, skipping.")

# ================================================================================
# STEP 8 — Create R and K Factor Rasters
# ================================================================================
# Spatially uniform R and K factor rasters are created matching the DEM grid.
# Each pixel receives a constant value across the study area.
#
# R = 250 MJ·mm / ha·h·yr
#   Approximation for Central California Coast (Mediterranean climate,
#   ~500–1000mm annual precipitation concentrated in winter months).
#   For improved accuracy, derive from PRISM 30-year climatological normals
#   or the NOAA R-Factor dataset (Office for Coastal Management, 2026).
#
# K = 0.28 t·h / MJ·mm
#   Approximation for mixed clay loam and silty clay soils of the Monterey
#   Formation and coastal range (Renard et al., 1997). For improved accuracy,
#   derive from USDA SSURGO database via Web Soil Survey.

print("\n[Step 8] Creating R and K factor rasters...")

gt, proj, cols, rows = copy_grid_properties(ls_path)
driver = gdal.GetDriverByName("GTiff")

for factor_val, out_name, description in [
    (R_VALUE, "R_factor.tif", "Rainfall Erosivity (R)"),
    (K_VALUE, "K_factor.tif", "Soil Erodibility (K)"),
]:
    out_path = path(out_name)
    if not os.path.exists(out_path):
        ds = driver.Create(
            out_path, cols, rows, 1, gdal.GDT_Float32,
            options=["COMPRESS=LZW", "TILED=YES"]
        )
        ds.SetGeoTransform(gt)
        ds.SetProjection(proj)
        band = ds.GetRasterBand(1)
        band.Fill(factor_val)
        band.SetNoDataValue(-9999.0)
        ds.FlushCache()
        ds = None
        print(f"  {description}: {factor_val} → {out_name}")
    else:
        print(f"  {out_name} already exists, skipping.")

# ================================================================================
# STEP 9 — Align C Factor to DEM Grid
# ================================================================================
# The C factor was derived from Sentinel-2 imagery which may have a different
# spatial extent than the DEM. All RUSLE inputs must share an identical grid
# (same CRS, extent, and resolution) before multiplication.
# Bilinear resampling is used for the continuous C factor values.

print("\n[Step 9] Aligning C factor to DEM grid...")

c_aligned_path = path("C_factor_aligned.tif")
if not os.path.exists(c_aligned_path):
    # Get target extent from LS factor
    gt, proj, cols, rows = copy_grid_properties(ls_path)
    xmin = gt[0]
    xmax = gt[0] + cols * gt[1]
    ymin = gt[3] + rows * gt[5]
    ymax = gt[3]

    warp_opts = gdal.WarpOptions(
        format="GTiff",
        srcSRS=f"EPSG:{TARGET_EPSG}",
        dstSRS=f"EPSG:{TARGET_EPSG}",
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=TARGET_RES,
        yRes=TARGET_RES,
        srcNodata=-9999,
        dstNodata=-9999,
        resampleAlg="bilinear",
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    )
    gdal.Warp(c_aligned_path, path("C_factor.tif"), options=warp_opts)
    print("  C factor aligned to DEM grid.")
else:
    print("  C_factor_aligned.tif already exists, skipping.")

# ================================================================================
# STEP 10 — Create Highway 1 Corridor Buffer
# ================================================================================
# The Highway 1 GeoJSON (pre-downloaded from OpenStreetMap) is buffered by
# approximately 2km (0.018 degrees at this latitude) and reprojected to
# UTM Zone 10N for use as a clip mask.

print("\n[Step 10] Creating Highway 1 corridor buffer...")

buffer_wgs_path = path("hwy1_buffer_wgs84.gpkg")
buffer_utm_path = path("hwy1_buffer_utm.gpkg")

if not os.path.exists(buffer_utm_path):
    # Open the Highway 1 GeoJSON
    hwy_ds  = ogr.Open(OSM_HWY1)
    hwy_lyr = hwy_ds.GetLayer()

    # Create output GeoPackage for the buffer
    gpkg_driver = ogr.GetDriverByName("GPKG")
    buf_ds  = gpkg_driver.CreateDataSource(buffer_wgs_path)
    buf_lyr = buf_ds.CreateLayer(
        "hwy1_buffer", srs=hwy_lyr.GetSpatialRef(), geom_type=ogr.wkbPolygon
    )

    # Buffer each feature and union into a single dissolved polygon
    union_geom = ogr.Geometry(ogr.wkbPolygon)
    for feat in hwy_lyr:
        geom = feat.GetGeometryRef()
        if geom:
            buffered = geom.Buffer(BUFFER_DEGREES)
            union_geom = union_geom.Union(buffered)

    # Write dissolved buffer as a single feature
    out_feat = ogr.Feature(buf_lyr.GetLayerDefn())
    out_feat.SetGeometry(union_geom)
    buf_lyr.CreateFeature(out_feat)
    buf_ds.FlushCache()
    buf_ds  = None
    hwy_ds  = None

    # Reproject buffer to UTM Zone 10N
    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(4326)
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(TARGET_EPSG)

    buf_ds    = ogr.Open(buffer_wgs_path)
    buf_lyr   = buf_ds.GetLayer()
    utm_ds    = gpkg_driver.CreateDataSource(buffer_utm_path)
    utm_lyr   = utm_ds.CreateLayer(
        "hwy1_buffer_utm", srs=target_srs, geom_type=ogr.wkbPolygon
    )
    transform = osr.CoordinateTransformation(source_srs, target_srs)

    for feat in buf_lyr:
        geom = feat.GetGeometryRef().Clone()
        geom.Transform(transform)
        out_feat = ogr.Feature(utm_lyr.GetLayerDefn())
        out_feat.SetGeometry(geom)
        utm_lyr.CreateFeature(out_feat)

    utm_ds.FlushCache()
    utm_ds = None
    buf_ds = None
    print("  Highway 1 buffer created and reprojected.")
else:
    print("  hwy1_buffer_utm.gpkg already exists, skipping.")

# ================================================================================
# STEP 11 — Compute RUSLE  (A = R × K × LS × C)
# ================================================================================
# All four factor rasters are loaded as NumPy arrays and multiplied cell-by-cell.
# Pixels where any factor is NoData (-9999) are excluded from the result.
#
# This step is performed outside the QGIS raster calculator to avoid memory
# and timeout constraints associated with operating on large raster grids
# (~350 million pixels for the full study area extent).
#
# Output units: t ha⁻¹ yr⁻¹ (tonnes per hectare per year)

print("\n[Step 11] Computing RUSLE = R × K × LS × C ...")

rusle_path = path("RUSLE.tif")
if not os.path.exists(rusle_path):
    print("  Loading factor arrays...")
    ls_ds  = gdal.Open(ls_path)
    c_ds   = gdal.Open(c_aligned_path)
    ls_arr = ls_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    c_arr  = c_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ls_ds  = None
    c_ds   = None

    # R and K are uniform scalars — no need to load full arrays
    nd = -9999.0
    print("  Multiplying factors...")
    rusle = np.where(
        (ls_arr != nd) & (c_arr != nd) & (c_arr >= 0),
        R_VALUE * K_VALUE * ls_arr * c_arr,
        nd
    )

    write_float32(rusle, ls_path, rusle_path)
    print("  RUSLE computed.")

    # Report summary statistics
    valid = rusle[(rusle > 0) & (rusle != nd)]
    if len(valid) > 0:
        print(f"\n  Summary statistics (full extent):")
        print(f"    Mean:   {valid.mean():.1f} t/ha/yr")
        print(f"    Median: {np.median(valid):.1f} t/ha/yr")
        print(f"    Max:    {valid.max():.1f} t/ha/yr")
        print(f"    p90:    {np.percentile(valid, 90):.1f} t/ha/yr")
else:
    print("  RUSLE.tif already exists, skipping.")

# ================================================================================
# STEP 12 — Clip RUSLE to Highway 1 Corridor
# ================================================================================
# The full-extent RUSLE raster is clipped to the 2km Highway 1 buffer polygon.
# CROP_TO_CUTLINE removes all pixels outside the buffer boundary.

print("\n[Step 12] Clipping RUSLE to Highway 1 corridor...")

rusle_final_path = path("RUSLE_final.tif")
if not os.path.exists(rusle_final_path):
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=buffer_utm_path,
        cropToCutline=True,
        srcNodata=-9999,
        dstNodata=-9999,
        srcSRS=f"EPSG:{TARGET_EPSG}",
        dstSRS=f"EPSG:{TARGET_EPSG}",
        xRes=TARGET_RES,
        yRes=TARGET_RES,
        resampleAlg="near",
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    )
    gdal.Warp(rusle_final_path, rusle_path, options=warp_opts)
    print("  RUSLE clipped to corridor.")

    # Final statistics
    ds    = gdal.Open(rusle_final_path)
    arr   = ds.GetRasterBand(1).ReadAsArray()
    ds    = None
    valid = arr[(arr > 0) & (arr != -9999)]

    if len(valid) > 0:
        pixel_ha = TARGET_RES * TARGET_RES / 10000.0  # 10m × 10m = 0.01 ha
        print(f"\n  Final corridor statistics:")
        print(f"    Valid land pixels:  {len(valid):,}")
        print(f"    Total land area:    {len(valid)*pixel_ha:,.1f} ha "
              f"({len(valid)*pixel_ha/100:.1f} km²)")
        print(f"    Mean erosion rate:  {valid.mean():.1f} t/ha/yr")
        print(f"    Median:            {np.median(valid):.1f} t/ha/yr")
        print(f"    Maximum:           {valid.max():.1f} t/ha/yr")
        print(f"    90th percentile:   {np.percentile(valid, 90):.1f} t/ha/yr")

        print(f"\n  Risk class distribution:")
        classes = [
            ("Very Low",  0,    5),
            ("Low",       5,    20),
            ("Moderate",  20,   50),
            ("High",      50,   100),
            ("Very High", 100,  200),
            ("Extreme",   200,  99999),
        ]
        total = len(valid)
        for name, lo, hi in classes:
            count = int(np.sum((valid >= lo) & (valid < hi)))
            area  = count * pixel_ha
            pct   = count / total * 100
            print(f"    {name:<12} ({lo:>3}–{hi if hi < 99999 else '>200':>4} t/ha/yr): "
                  f"{area:>10,.1f} ha  ({pct:.1f}%)")
else:
    print("  RUSLE_final.tif already exists, skipping.")

# ================================================================================
# COMPLETE
# ================================================================================

print("\n" + "=" * 72)
print("Processing complete.")
print(f"All outputs written to: {OUT_DIR}")
print("\nKey output files:")
print(f"  Final RUSLE map:      {rusle_final_path}")
print(f"  Highway 1 corridor:   {buffer_utm_path}")
print(f"  LS factor:            {ls_path}")
print(f"  C factor (aligned):   {c_aligned_path}")
print("=" * 72)
