import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CREATING HABITAT SUITABILITY RASTER")
print("=" * 80)

species_name = 'vanessa-atalanta'

# Load the trained model
print("\n[STEP 1] Loading trained model...")
rf_model = joblib.load(f'{species_name}_rf_model.pkl')
print(f"  ✓ Model loaded")

# Load climate averages (we need to recalculate them)
print("\n[STEP 2] Loading climate data...")

climate_base = Path('climate-data')
growing_season_months = [5, 6, 7, 8, 9]

climate_vars = {
    'tmax': climate_base / 'Maximum Temperature',
    'tmin': climate_base / 'Minimum Temperature',
    'ppt': climate_base / 'Precipitation'
}

climate_averages = {}

for var_name, var_path in climate_vars.items():
    print(f"  Processing {var_name}...")
    
    all_data = []
    
    for year in range(2015, 2025):
        nc_file = var_path / f'TerraClimate_{var_name}_{year}.nc'
        
        if not nc_file.exists():
            continue
        
        ds = xr.open_dataset(nc_file)
        ds_season = ds.sel(time=ds.time.dt.month.isin(growing_season_months))
        year_mean = ds_season[var_name].mean(dim='time')
        all_data.append(year_mean)
        ds.close()
    
    if all_data:
        climate_avg = sum(all_data) / len(all_data)
        climate_averages[var_name] = climate_avg

# Define study area extent
print("\n[STEP 3] Setting up prediction grid...")

# You can adjust this to match your species range or make it smaller for faster processing
extent = {
    'lat_min': 25,   # Adjust based on your species range
    'lat_max': 55,
    'lon_min': -130,
    'lon_max': -65
}

# Use climate resolution (~4km)
lat_res = abs(float(climate_averages['tmax'].lat[1] - climate_averages['tmax'].lat[0]))
lon_res = abs(float(climate_averages['tmax'].lon[1] - climate_averages['tmax'].lon[0]))

print(f"  Extent: {extent}")
print(f"  Resolution: ~{lat_res:.4f} degrees")

# Create prediction grid matching climate data
lats = climate_averages['tmax'].lat.values
lons = climate_averages['tmax'].lon.values

lat_mask = (lats >= extent['lat_min']) & (lats <= extent['lat_max'])
lon_mask = (lons >= extent['lon_min']) & (lons <= extent['lon_max'])

pred_lats = lats[lat_mask]
pred_lons = lons[lon_mask]

print(f"  Grid size: {len(pred_lats)} x {len(pred_lons)} = {len(pred_lats) * len(pred_lons):,} pixels")

# Extract climate data for prediction grid
print("\n[STEP 4] Extracting climate values for prediction grid...")

tmax_grid = climate_averages['tmax'].sel(lat=pred_lats, lon=pred_lons).values
tmin_grid = climate_averages['tmin'].sel(lat=pred_lats, lon=pred_lons).values
ppt_grid = climate_averages['ppt'].sel(lat=pred_lats, lon=pred_lons).values

# For now, use a constant land cover value (since it had 0 importance anyway)
# Or set to 0 to exclude it
landcover_grid = np.zeros_like(tmax_grid)

print(f"  ✓ Climate grids extracted")

# Reshape for prediction
print("\n[STEP 5] Preparing data for prediction...")

n_pixels = len(pred_lats) * len(pred_lons)
tmax_flat = tmax_grid.flatten()
tmin_flat = tmin_grid.flatten()
ppt_flat = ppt_grid.flatten()
landcover_flat = landcover_grid.flatten()

# Create feature dataframe
pred_df = pd.DataFrame({
    'tmax': tmax_flat,
    'tmin': tmin_flat,
    'ppt': ppt_flat,
    'landcover': landcover_flat
})

# Remove NaN values (ocean, etc.)
valid_mask = ~pred_df.isnull().any(axis=1)
valid_indices = np.where(valid_mask)[0]

print(f"  Valid pixels: {valid_mask.sum():,}/{n_pixels:,}")

# Predict only on valid pixels
print("\n[STEP 6] Predicting habitat suitability...")

predictions = np.full(n_pixels, np.nan)
predictions[valid_indices] = rf_model.predict_proba(pred_df[valid_mask])[:, 1]

# Reshape to grid
suitability_grid = predictions.reshape((len(pred_lats), len(pred_lons)))

print(f"  ✓ Predictions complete")
print(f"  Suitability range: {np.nanmin(suitability_grid):.3f} to {np.nanmax(suitability_grid):.3f}")

# Save as GeoTIFF
print("\n[STEP 7] Saving raster...")

output_file = f'{species_name}_habitat_suitability.tif'

transform = from_bounds(
    pred_lons.min(), pred_lats.min(),
    pred_lons.max(), pred_lats.max(),
    len(pred_lons), len(pred_lats)
)

with rasterio.open(
    output_file,
    'w',
    driver='GTiff',
    height=suitability_grid.shape[0],
    width=suitability_grid.shape[1],
    count=1,
    dtype=suitability_grid.dtype,
    crs='EPSG:4326',
    transform=transform,
    compress='lzw'
) as dst:
    dst.write(suitability_grid, 1)

print(f"  ✓ Raster saved: {output_file}")

print("\n" + "=" * 80)
print("✓ RASTER CREATION COMPLETE!")
print("=" * 80)
print(f"\nYou can now open '{output_file}' in QGIS")
print("Suitability values range from 0 (unsuitable) to 1 (highly suitable)")