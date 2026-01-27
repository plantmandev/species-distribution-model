import xarray as xr
from pathlib import Path

# Define the climate data directory structure
climate_base = Path('climate-data')

variable_folders = {
    'tmax': climate_base / 'Maximum Temperature',
    'tmin': climate_base / 'Minimum Temperature', 
    'ppt': climate_base / 'Precipitation'
}

print("=" * 60)
print("TERRACLIMATE DATA INSPECTION")
print("=" * 60)

for var_name, folder_path in variable_folders.items():
    print(f"\n{var_name.upper()} - {folder_path.name}")
    print("-" * 60)
    
    # Get all .nc files in this folder
    nc_files = sorted(folder_path.glob('*.nc'))
    print(f"Files found: {len(nc_files)}")
    
    if nc_files:
        # Inspect the first file
        first_file = nc_files[0]
        print(f"Inspecting: {first_file.name}")
        
        try:
            ds = xr.open_dataset(first_file)
            
            print(f"\nVariables in file:")
            for var in ds.data_vars:
                print(f"  - {var}: {ds[var].long_name if 'long_name' in ds[var].attrs else 'no description'}")
                print(f"    Units: {ds[var].units if 'units' in ds[var].attrs else 'unknown'}")
                print(f"    Shape: {ds[var].shape}")
                print(f"    Data type: {ds[var].dtype}")
            
            print(f"\nCoordinates:")
            print(f"  Latitude range: {float(ds.lat.min()):.2f} to {float(ds.lat.max()):.2f}")
            print(f"  Longitude range: {float(ds.lon.min()):.2f} to {float(ds.lon.max()):.2f}")
            print(f"  Resolution: ~{abs(float(ds.lat[1] - ds.lat[0])):.4f} degrees (~4km)")
            
            if 'time' in ds.coords:
                print(f"  Time steps: {len(ds.time)}")
                print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
            
            ds.close()
            
        except Exception as e:
            print(f"  Error reading file: {e}")
    else:
        print("  No .nc files found in this folder!")