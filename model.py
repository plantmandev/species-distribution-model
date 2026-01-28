import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SPECIES DISTRIBUTION MODEL - VANESSA ATALANTA")
print("=" * 80)

print("\n[STEP 1] Loading occurrence data...")

species_name = 'vanessa-atalanta'
occurrence_file = Path(f'occurrence-data/{species_name}') / f'{species_name}-gbif.gpkg'

gdf = gpd.read_file(occurrence_file)
gdf['eventDate'] = pd.to_datetime(gdf['eventDate'], errors='coerce')
gdf['year'] = gdf['eventDate'].dt.year

gdf = gdf[(gdf['year'] >= 2015) & (gdf['year'] <= 2025)]
gdf = gdf.dropna(subset=['decimalLatitude', 'decimalLongitude'])

print(f"  Total occurrences: {len(gdf)}")
print(f"  Geographic extent:")
print(f"    Lat: {gdf['decimalLatitude'].min():.2f} to {gdf['decimalLatitude'].max():.2f}")
print(f"    Lon: {gdf['decimalLongitude'].min():.2f} to {gdf['decimalLongitude'].max():.2f}")

print("\n[STEP 2] Calculating growing season climate averages...")

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
        print(f"    ✓ Averaged {len(all_data)} years")

print(f"  Climate variables ready: {list(climate_averages.keys())}")

print("\n[STEP 3] Extracting climate values at occurrence points...")

def extract_climate_batch(gdf, climate_data):
    """Vectorized climate extraction"""
    lats = gdf['decimalLatitude'].values
    lons = gdf['decimalLongitude'].values
    
    lat_indices = np.searchsorted(climate_data.lat.values[::-1], lats)
    lon_indices = np.searchsorted(climate_data.lon.values, lons)
    
    lat_indices = np.clip(lat_indices, 0, len(climate_data.lat) - 1)
    lon_indices = np.clip(lon_indices, 0, len(climate_data.lon) - 1)
    
    values = climate_data.values[lat_indices, lon_indices]
    values[np.isnan(values)] = None
    
    return values

for var_name, climate_data in climate_averages.items():
    print(f"  Extracting {var_name}...")
    gdf[var_name] = extract_climate_batch(gdf, climate_data)

gdf_clean = gdf.dropna(subset=list(climate_averages.keys()))
print(f"  Occurrences with climate data: {len(gdf_clean)}/{len(gdf)}")

print("\n[STEP 4] Extracting land cover (batch mode)...")

landcover_file = Path('land-cover-data/data/NA_NALCMS_landcover_2020v2_30m.tif')

def extract_landcover_batch(gdf, raster_path):
    """Batch extraction using rasterio.sample"""
    print(f"  Opening {raster_path.name}...")
    
    with rasterio.open(raster_path) as src:
        coords = [(row['decimalLongitude'], row['decimalLatitude']) 
                  for idx, row in gdf.iterrows()]
        
        print(f"  Sampling {len(coords)} points...")
        values = [x[0] if x[0] > 0 else None for x in src.sample(coords)]
    
    return values

gdf_clean['landcover'] = extract_landcover_batch(gdf_clean, landcover_file)
gdf_clean = gdf_clean.dropna(subset=['landcover'])
print(f"  Occurrences with complete data: {len(gdf_clean)}")

print("\n[STEP 5] Generating background points...")

n_background = 10000

na_extent = {
    'lat_min': 15,
    'lat_max': 70,
    'lon_min': -170,
    'lon_max': -50
}

np.random.seed(42)
background_df = pd.DataFrame({
    'decimalLatitude': np.random.uniform(na_extent['lat_min'], na_extent['lat_max'], n_background),
    'decimalLongitude': np.random.uniform(na_extent['lon_min'], na_extent['lon_max'], n_background),
    'presence': 0
})

print(f"  Extracting climate for background points...")
for var_name, climate_data in climate_averages.items():
    background_df[var_name] = extract_climate_batch(background_df, climate_data)

print(f"  Extracting land cover for background points...")
background_df['landcover'] = extract_landcover_batch(background_df, landcover_file)

background_clean = background_df.dropna()
print(f"  Background points with data: {len(background_clean)}/{n_background}")

print("\n[STEP 6] Preparing training data...")

gdf_clean['presence'] = 1

feature_cols = list(climate_averages.keys()) + ['landcover']
presence_data = gdf_clean[feature_cols + ['presence']].copy()
background_data = background_clean[feature_cols + ['presence']].copy()

training_data = pd.concat([presence_data, background_data], ignore_index=True)

print(f"  Total training samples: {len(training_data)}")
print(f"    Presences: {len(presence_data)}")
print(f"    Background: {len(background_data)}")

print("\n[STEP 7] Training Random Forest model...")

X = training_data[feature_cols]
y = training_data['presence']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

print("\n[STEP 8] Evaluating model...")

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC Score: {auc_score:.3f}")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

print("\n[STEP 9] Saving results...")

import joblib
joblib.dump(rf_model, f'{species_name}_rf_model.pkl')
training_data.to_csv(f'{species_name}_training_data.csv', index=False)
feature_importance.to_csv(f'{species_name}_feature_importance.csv', index=False)

gdf_clean['predicted_suitability'] = rf_model.predict_proba(gdf_clean[feature_cols])[:, 1]
gdf_clean.to_file(f'{species_name}_occurrences_with_predictions.gpkg', driver='GPKG')

print("\n" + "=" * 80)
print("✓ COMPLETE!")
print("=" * 80)
print(f"AUC Score: {auc_score:.3f}")
print(f"Training samples: {len(training_data)}")
print("\nTop 3 features:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")