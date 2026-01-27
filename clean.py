import pandas as pd
import geopandas as gpd
from pathlib import Path

# Get the occurrence-data directory (same level as where script is running)
occurrence_dir = Path('occurrence-data')

print(f"Looking for occurrence data in: {occurrence_dir.absolute()}")

if not occurrence_dir.exists():
    print(f"ERROR: Directory does not exist: {occurrence_dir.absolute()}")
    exit(1)

# Loop through all subdirectories in occurrence-data
for species_dir in occurrence_dir.iterdir():
    if not species_dir.is_dir():
        continue
    
    print(f"\nProcessing: {species_dir.name}")
    
    # Look for CSV files in this directory
    csv_files = list(species_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"  No CSV files found in {species_dir.name}")
        continue
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"  Reading: {csv_file.name}")
        
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Check if required columns exist
            keep_columns = ['decimalLatitude', 'decimalLongitude', 'eventDate']
            if not all(col in df.columns for col in keep_columns):
                print(f"  Skipping {csv_file.name}: missing required columns")
                continue
            
            cleaned_df = df[keep_columns].copy()
            
            # Parse dates with UTC to avoid timezone warning
            cleaned_df['eventDate'] = pd.to_datetime(cleaned_df['eventDate'], format='ISO8601', errors='coerce', utc=True)
            
            # Remove rows with missing coordinates or dates
            cleaned_df = cleaned_df.dropna(subset=['decimalLatitude', 'decimalLongitude', 'eventDate'])
            
            if len(cleaned_df) == 0:
                print(f"  No valid records found in {csv_file.name}")
                continue
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                cleaned_df,
                geometry=gpd.points_from_xy(cleaned_df.decimalLongitude, cleaned_df.decimalLatitude),
                crs="EPSG:4326"
            )
            
            # Create output filename (replace .csv with .gpkg)
            output_file = csv_file.with_suffix('.gpkg')
            
            # Save to GeoPackage
            gdf.to_file(output_file, driver='GPKG')
            print(f"  ✓ Saved {len(gdf)} records to {output_file.name}")
            
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")

print("\n✓ Processing complete!")