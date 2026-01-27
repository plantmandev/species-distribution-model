import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.mask import mask
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SPECIES DISTRIBUTION MODEL - VANESSA ATALANTA")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD OCCURRENCE DATA
# ============================================================================
print("\n[STEP 1] Loading occurrence data...")

species_name = 'vanessa-atalanta'
occurrence_file = Path(f'occurrence-data/{species_name}') / f'{species_name}-gbif.gpkg'

gdf = gpd.read_file(occurrence_file)
gdf['eventDate'] = pd.to_datetime(gdf['eventDate'], errors='coerce')
gdf['year'] = gdf['eventDate'].dt.year

# Filter to 2015-2025
gdf = gdf[(gdf['year'] >= 2015) & (gdf['year'] <= 2025)]
gdf = gdf.dropna(subset=['decimalLatitude', 'decimalLongitude'])

print(f"  Total occurrences (2015-2025): {len(gdf)}")
print(f"  Geographic extent:")
print(f"    Lat: {gdf['decimalLatitude'].min():.2f} to {gdf['decimalLatitude'].max():.2f}")
print(f"    Lon: {gdf['decimalLongitude'].min():.2f} to {gdf['decimalLongitude'].max():.2f}")

# ============================================================================
# STEP 2: CALCULATE GROWING SEASON CLIMATE AVERAGES (2015-2024)
# ============================================================================
print("\n[STEP 2] Calculating growing season climate averages (May-September, 2015-2024)...")

climate_base = Path('climate-data')
growing_season_months = [5, 6, 7, 8, 9]  # May-September

climate_vars = {
    'tmax': climate_base / 'Maximum Temperature',
    'tmin': climate_base / 'Minimum Temperature',
    'ppt': climate_base / 'Precipitation'
}

climate_averages = {}

for var_name, var_path in climate_vars.items():
    print(f"  Processing {var_name}...")
    
    all_data = []
    
    # Load all years (2015-2024)
    for year in range(2015, 2025):
        nc_file = var_path / f'TerraClimate_{var_name}_{year}.nc'
        
        if not nc_file.exists():
            print(f"    Warning: {nc_file.name} not found, skipping...")
            continue
        
        ds = xr.open_dataset(nc_file)
        
        # Select growing season months (May-September)
        ds_season = ds.sel(time=ds.time.dt.month.isin(growing_season_months))
        
        # Calculate mean across months for this year
        year_mean = ds_season[var_name].mean(dim='time')
        all_data.append(year_mean)
        
        ds.close()
    
    # Average across all years
    if all_data:
        climate_avg = sum(all_data) / len(all_data)
        climate_averages[var_name] = climate_avg
        print(f"    ✓ Averaged {len(all_data)} years")
    else:
        print(f"    ✗ No data found for {var_name}")

print(f"  Climate variables prepared: {list(climate_averages.keys())}")

# ============================================================================
# STEP 3: EXTRACT CLIMATE VALUES AT OCCURRENCE POINTS
# ============================================================================
print("\n[STEP 3] Extracting climate values at occurrence points...")

def extract_climate_value(lat, lon, climate_data):
    """Extract climate value at a given lat/lon"""
    try:
        # Find nearest grid cell
        lat_idx = np.abs(climate_data.lat.values - lat).argmin()
        lon_idx = np.abs(climate_data.lon.values - lon).argmin()
        value = float(climate_data.values[lat_idx, lon_idx])
        return value if not np.isnan(value) else None
    except:
        return None

# Extract climate for each occurrence
for var_name, climate_data in climate_averages.items():
    print(f"  Extracting {var_name}...")
    gdf[var_name] = gdf.apply(
        lambda row: extract_climate_value(row['decimalLatitude'], row['decimalLongitude'], climate_data),
        axis=1
    )

# Remove rows with missing climate data
gdf_clean = gdf.dropna(subset=list(climate_averages.keys()))
print(f"  Occurrences with complete climate data: {len(gdf_clean)}/{len(gdf)}")

# ============================================================================
# STEP 4: EXTRACT LAND COVER AT OCCURRENCE POINTS
# ============================================================================
print("\n[STEP 4] Extracting land cover at occurrence points...")

landcover_file = Path('land-cover-data/data/NA_NALCMS_landcover_2020v2_30m.tif')

def extract_landcover(lat, lon, raster_path):
    """Extract land cover value at a given lat/lon"""
    try:
        with rasterio.open(raster_path) as src:
            # Convert lat/lon to raster coordinates
            row, col = src.index(lon, lat)
            # Read the value
            value = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
            return int(value) if value > 0 else None
    except:
        return None

print(f"  Extracting from {landcover_file.name}...")
gdf_clean['landcover'] = gdf_clean.apply(
    lambda row: extract_landcover(row['decimalLatitude'], row['decimalLongitude'], landcover_file),
    axis=1
)

gdf_clean = gdf_clean.dropna(subset=['landcover'])
print(f"  Occurrences with complete environmental data: {len(gdf_clean)}")

# ============================================================================
# STEP 5: GENERATE BACKGROUND POINTS
# ============================================================================
print("\n[STEP 5] Generating background points across North America...")

n_background = 10000

# Define North America extent (approximate)
na_extent = {
    'lat_min': 15,  # Southern Mexico
    'lat_max': 70,  # Northern Canada
    'lon_min': -170,  # Western Alaska
    'lon_max': -50   # Eastern Canada
}

# Generate random points
np.random.seed(42)
background_lats = np.random.uniform(na_extent['lat_min'], na_extent['lat_max'], n_background)
background_lons = np.random.uniform(na_extent['lon_min'], na_extent['lon_max'], n_background)

background_df = pd.DataFrame({
    'decimalLatitude': background_lats,
    'decimalLongitude': background_lons,
    'presence': 0
})

print(f"  Extracting climate for {n_background} background points...")
for var_name, climate_data in climate_averages.items():
    background_df[var_name] = background_df.apply(
        lambda row: extract_climate_value(row['decimalLatitude'], row['decimalLongitude'], climate_data),
        axis=1
    )

print(f"  Extracting land cover for background points...")
background_df['landcover'] = background_df.apply(
    lambda row: extract_landcover(row['decimalLatitude'], row['decimalLongitude'], landcover_file),
    axis=1
)

background_clean = background_df.dropna()
print(f"  Background points with complete data: {len(background_clean)}/{n_background}")

# ============================================================================
# STEP 6: PREPARE TRAINING DATA
# ============================================================================
print("\n[STEP 6] Preparing training data...")

# Add presence label
gdf_clean['presence'] = 1

# Combine presence and background
feature_cols = list(climate_averages.keys()) + ['landcover']
presence_data = gdf_clean[feature_cols + ['presence']].copy()
background_data = background_clean[feature_cols + ['presence']].copy()

training_data = pd.concat([presence_data, background_data], ignore_index=True)

print(f"  Total training samples: {len(training_data)}")
print(f"    Presences: {len(presence_data)}")
print(f"    Background: {len(background_data)}")
print(f"  Features: {feature_cols}")

# ============================================================================
# STEP 7: TRAIN RANDOM FOREST MODEL
# ============================================================================
print("\n[STEP 7] Training Random Forest model...")

X = training_data[feature_cols]
y = training_data['presence']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)

# ============================================================================
# STEP 8: EVALUATE MODEL
# ============================================================================
print("\n[STEP 8] Evaluating model performance...")

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC Score: {auc_score:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# ============================================================================
# STEP 9: SAVE MODEL AND RESULTS
# ============================================================================
print("\n[STEP 9] Saving results...")

# Save model
import joblib
model_file = f'{species_name}_rf_model.pkl'
joblib.dump(rf_model, model_file)
print(f"  ✓ Model saved: {model_file}")

# Save training data
training_file = f'{species_name}_training_data.csv'
training_data.to_csv(training_file, index=False)
print(f"  ✓ Training data saved: {training_file}")

# Save feature importance
importance_file = f'{species_name}_feature_importance.csv'
feature_importance.to_csv(importance_file, index=False)
print(f"  ✓ Feature importance saved: {importance_file}")

# Save presence points with predictions
gdf_clean['predicted_suitability'] = rf_model.predict_proba(gdf_clean[feature_cols])[:, 1]
presence_file = f'{species_name}_occurrences_with_predictions.gpkg'
gdf_clean.to_file(presence_file, driver='GPKG')
print(f"  ✓ Occurrence data with predictions saved: {presence_file}")

print("\n" + "=" * 80)
print("✓ SDM PIPELINE COMPLETE!")
print("=" * 80)
print(f"\nModel Performance Summary:")
print(f"  AUC Score: {auc_score:.3f}")
print(f"  Training samples: {len(training_data)}")
print(f"  Top 3 most important features:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"    {row['feature']}: {row['importance']:.3f}")