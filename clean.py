import pandas as pd
import geopandas as gpd

df = pd.read_csv("vanessa_cardui_gbif.csv", low_memory=False) # low_memory=False removes error message

# GBIF = ['decimalLatitude', 'decimalLongitude', 'eventDate']
# iNaturalist = [latitude, longitude, observed_on]
keep_columns = ['decimalLatitude', 'decimalLongitude', 'eventDate'] 
cleaned_df = df[keep_columns].copy()

# Parse dates with UTC to avoid timezone warning
cleaned_df['eventDate'] = pd.to_datetime(cleaned_df['eventDate'], format='ISO8601', errors='coerce', utc=True)

# Remove rows with missing coordinates
cleaned_df = cleaned_df.dropna(subset=['decimalLatitude', 'decimalLongitude', 'eventDate'])

cleaned_df = cleaned_df[cleaned_df['eventDate'].dt.year >= 2017]

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    cleaned_df,
    geometry=gpd.points_from_xy(cleaned_df.decimalLongitude, cleaned_df.decimalLatitude),
    crs="EPSG:4326"
)

# GeoJSON saves dates as strings, GeoPackage does not 
gdf.to_file('vanessa-cardui-gbif.gpkg', driver='GPKG')
print(f"Saved {len(gdf)} records to GeoPackage")