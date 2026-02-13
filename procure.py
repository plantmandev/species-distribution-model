from pygbif import occurrences as occ
from pygbif import species as species_api
import pandas as pd
import geopandas as gpd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse
from tqdm import tqdm
import warnings

# Suppress pandas FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Paths
DATA_DIR = Path('occurrence-data')
METADATA_FILE = DATA_DIR / 'species-metadata.csv'

# Country code to UN subregion mapping
COUNTRY_TO_SUBREGION = {
    # Northern Africa
    'DZ': 'Northern Africa', 'EG': 'Northern Africa', 'LY': 'Northern Africa', 'MA': 'Northern Africa', 
    'SD': 'Northern Africa', 'TN': 'Northern Africa', 'EH': 'Northern Africa',
    
    # Sub-Saharan Africa
    'AO': 'Sub-Saharan Africa', 'BJ': 'Sub-Saharan Africa', 'BW': 'Sub-Saharan Africa', 'BF': 'Sub-Saharan Africa',
    'BI': 'Sub-Saharan Africa', 'CM': 'Sub-Saharan Africa', 'CV': 'Sub-Saharan Africa', 'CF': 'Sub-Saharan Africa',
    'TD': 'Sub-Saharan Africa', 'KM': 'Sub-Saharan Africa', 'CG': 'Sub-Saharan Africa', 'CD': 'Sub-Saharan Africa',
    'CI': 'Sub-Saharan Africa', 'DJ': 'Sub-Saharan Africa', 'GQ': 'Sub-Saharan Africa', 'ER': 'Sub-Saharan Africa',
    'ET': 'Sub-Saharan Africa', 'GA': 'Sub-Saharan Africa', 'GM': 'Sub-Saharan Africa', 'GH': 'Sub-Saharan Africa',
    'GN': 'Sub-Saharan Africa', 'GW': 'Sub-Saharan Africa', 'KE': 'Sub-Saharan Africa', 'LS': 'Sub-Saharan Africa',
    'LR': 'Sub-Saharan Africa', 'MG': 'Sub-Saharan Africa', 'MW': 'Sub-Saharan Africa', 'ML': 'Sub-Saharan Africa',
    'MR': 'Sub-Saharan Africa', 'MU': 'Sub-Saharan Africa', 'YT': 'Sub-Saharan Africa', 'MZ': 'Sub-Saharan Africa',
    'NA': 'Sub-Saharan Africa', 'NE': 'Sub-Saharan Africa', 'NG': 'Sub-Saharan Africa', 'RE': 'Sub-Saharan Africa',
    'RW': 'Sub-Saharan Africa', 'ST': 'Sub-Saharan Africa', 'SN': 'Sub-Saharan Africa', 'SC': 'Sub-Saharan Africa',
    'SL': 'Sub-Saharan Africa', 'SO': 'Sub-Saharan Africa', 'ZA': 'Sub-Saharan Africa', 'SS': 'Sub-Saharan Africa',
    'SZ': 'Sub-Saharan Africa', 'TZ': 'Sub-Saharan Africa', 'TG': 'Sub-Saharan Africa', 'UG': 'Sub-Saharan Africa',
    'ZM': 'Sub-Saharan Africa', 'ZW': 'Sub-Saharan Africa',
    
    # Northern America
    'BM': 'Northern America', 'CA': 'Northern America', 'GL': 'Northern America', 'PM': 'Northern America',
    'US': 'Northern America',
    
    # Central America
    'BZ': 'Central America', 'CR': 'Central America', 'SV': 'Central America', 'GT': 'Central America',
    'HN': 'Central America', 'MX': 'Central America', 'NI': 'Central America', 'PA': 'Central America',
    
    # South America
    'AR': 'South America', 'BO': 'South America', 'BR': 'South America', 'CL': 'South America',
    'CO': 'South America', 'EC': 'South America', 'FK': 'South America', 'GF': 'South America',
    'GY': 'South America', 'PY': 'South America', 'PE': 'South America', 'SR': 'South America',
    'UY': 'South America', 'VE': 'South America',
    
    # Caribbean
    'AI': 'Caribbean', 'AG': 'Caribbean', 'AW': 'Caribbean', 'BS': 'Caribbean', 'BB': 'Caribbean',
    'BQ': 'Caribbean', 'VG': 'Caribbean', 'KY': 'Caribbean', 'CU': 'Caribbean', 'CW': 'Caribbean',
    'DM': 'Caribbean', 'DO': 'Caribbean', 'GD': 'Caribbean', 'GP': 'Caribbean', 'HT': 'Caribbean',
    'JM': 'Caribbean', 'MQ': 'Caribbean', 'MS': 'Caribbean', 'PR': 'Caribbean', 'BL': 'Caribbean',
    'KN': 'Caribbean', 'LC': 'Caribbean', 'MF': 'Caribbean', 'VC': 'Caribbean', 'SX': 'Caribbean',
    'TT': 'Caribbean', 'TC': 'Caribbean', 'VI': 'Caribbean',
    
    # Central Asia
    'KZ': 'Central Asia', 'KG': 'Central Asia', 'TJ': 'Central Asia', 'TM': 'Central Asia', 'UZ': 'Central Asia',
    
    # Eastern Asia
    'CN': 'Eastern Asia', 'HK': 'Eastern Asia', 'MO': 'Eastern Asia', 'KP': 'Eastern Asia', 'KR': 'Eastern Asia',
    'MN': 'Eastern Asia', 'JP': 'Eastern Asia', 'TW': 'Eastern Asia',
    
    # South-Eastern Asia
    'BN': 'South-Eastern Asia', 'KH': 'South-Eastern Asia', 'ID': 'South-Eastern Asia', 'LA': 'South-Eastern Asia',
    'MY': 'South-Eastern Asia', 'MM': 'South-Eastern Asia', 'PH': 'South-Eastern Asia', 'SG': 'South-Eastern Asia',
    'TH': 'South-Eastern Asia', 'TL': 'South-Eastern Asia', 'VN': 'South-Eastern Asia',
    
    # Southern Asia
    'AF': 'Southern Asia', 'BD': 'Southern Asia', 'BT': 'Southern Asia', 'IN': 'Southern Asia',
    'IR': 'Southern Asia', 'MV': 'Southern Asia', 'NP': 'Southern Asia', 'PK': 'Southern Asia', 'LK': 'Southern Asia',
    
    # Western Asia
    'AM': 'Western Asia', 'AZ': 'Western Asia', 'BH': 'Western Asia', 'CY': 'Western Asia', 'GE': 'Western Asia',
    'IQ': 'Western Asia', 'IL': 'Western Asia', 'JO': 'Western Asia', 'KW': 'Western Asia', 'LB': 'Western Asia',
    'OM': 'Western Asia', 'PS': 'Western Asia', 'QA': 'Western Asia', 'SA': 'Western Asia', 'SY': 'Western Asia',
    'TR': 'Western Asia', 'AE': 'Western Asia', 'YE': 'Western Asia',
    
    # Eastern Europe
    'BY': 'Eastern Europe', 'BG': 'Eastern Europe', 'CZ': 'Eastern Europe', 'HU': 'Eastern Europe',
    'PL': 'Eastern Europe', 'MD': 'Eastern Europe', 'RO': 'Eastern Europe', 'RU': 'Eastern Europe',
    'SK': 'Eastern Europe', 'UA': 'Eastern Europe',
    
    # Northern Europe
    'AX': 'Northern Europe', 'DK': 'Northern Europe', 'EE': 'Northern Europe', 'FO': 'Northern Europe',
    'FI': 'Northern Europe', 'GG': 'Northern Europe', 'IS': 'Northern Europe', 'IE': 'Northern Europe',
    'IM': 'Northern Europe', 'JE': 'Northern Europe', 'LV': 'Northern Europe', 'LT': 'Northern Europe',
    'NO': 'Northern Europe', 'SJ': 'Northern Europe', 'SE': 'Northern Europe', 'GB': 'Northern Europe',
    
    # Southern Europe
    'AL': 'Southern Europe', 'AD': 'Southern Europe', 'BA': 'Southern Europe', 'HR': 'Southern Europe',
    'GI': 'Southern Europe', 'GR': 'Southern Europe', 'VA': 'Southern Europe', 'IT': 'Southern Europe',
    'MT': 'Southern Europe', 'ME': 'Southern Europe', 'MK': 'Southern Europe', 'PT': 'Southern Europe',
    'SM': 'Southern Europe', 'RS': 'Southern Europe', 'SI': 'Southern Europe', 'ES': 'Southern Europe',
    
    # Western Europe
    'AT': 'Western Europe', 'BE': 'Western Europe', 'FR': 'Western Europe', 'DE': 'Western Europe',
    'LI': 'Western Europe', 'LU': 'Western Europe', 'MC': 'Western Europe', 'NL': 'Western Europe', 'CH': 'Western Europe',
    
    # Australia and New Zealand
    'AU': 'Australia and New Zealand', 'CX': 'Australia and New Zealand', 'CC': 'Australia and New Zealand',
    'HM': 'Australia and New Zealand', 'NZ': 'Australia and New Zealand', 'NF': 'Australia and New Zealand',
    
    # Melanesia
    'FJ': 'Melanesia', 'NC': 'Melanesia', 'PG': 'Melanesia', 'SB': 'Melanesia', 'VU': 'Melanesia',
    
    # Micronesia
    'GU': 'Micronesia', 'KI': 'Micronesia', 'MH': 'Micronesia', 'FM': 'Micronesia', 'NR': 'Micronesia',
    'MP': 'Micronesia', 'PW': 'Micronesia', 'UM': 'Micronesia',
    
    # Polynesia
    'AS': 'Polynesia', 'CK': 'Polynesia', 'PF': 'Polynesia', 'NU': 'Polynesia', 'PN': 'Polynesia',
    'WS': 'Polynesia', 'TK': 'Polynesia', 'TO': 'Polynesia', 'TV': 'Polynesia', 'WF': 'Polynesia',
    
    # Antarctica
    'AQ': 'Antarctica', 'BV': 'Antarctica', 'TF': 'Antarctica', 'GS': 'Antarctica',
}

def get_countries_and_subregion_from_geojson(gdf):
    """Extract unique country codes and determine primary subregion from GeoJSON data"""
    try:
        # Check if countryCode column exists in the data
        if 'countryCode' in gdf.columns:
            countries = gdf['countryCode'].dropna().unique().tolist()
            countries = [c for c in countries if c]  # Remove empty strings
        else:
            countries = []
        
        if not countries:
            return '', ''
        
        # Sort for consistency
        countries_str = ','.join(sorted(countries))
        
        # Determine primary subregion (most common)
        subregions = [COUNTRY_TO_SUBREGION.get(c, '') for c in countries]
        subregions = [s for s in subregions if s]  # Remove empty
        
        if subregions:
            # Get most common subregion
            from collections import Counter
            subregion = Counter(subregions).most_common(1)[0][0]
        else:
            subregion = ''
        
        return countries_str, subregion
        
    except Exception as e:
        return '', ''

# Metadata CSV functions
def create_metadata_template():
    """Create a template metadata CSV file"""
    template = pd.DataFrame({
        'species_name': ['Morpho', 'Panthera tigris'],
        'common_name': ['Blue Morpho Butterflies', 'Tiger'],
        'status': ['pending', 'pending'],
        'extent': ['', ''],
        'temporal_range': ['', ''],
        'expected_obs': [0, 0],
        'actual_obs': [0, 0],
        'last_updated': ['', ''],
        'taxonomic_rank': ['', ''],
        'gbif_key': ['', ''],
        'data_quality': [0.0, 0.0],
        'countries_observed': ['', ''],
        'subregion': ['', ''],
        'color': ['#0066FF', '#FF6600'],
        'notes': ['', '']
    })
    
    DATA_DIR.mkdir(exist_ok=True)
    template.to_csv(METADATA_FILE, index=False)
    return template

def read_metadata():
    """Read metadata CSV, create if doesn't exist"""
    if not METADATA_FILE.exists():
        print(f"\n{'='*60}")
        print("NO METADATA FILE FOUND")
        print(f"{'='*60}")
        print(f"\nCreating template at: {METADATA_FILE}")
        create_metadata_template()
        print("\n✓ Template created with example species")
        print("\nNext steps:")
        print("1. Edit species-metadata.csv in occurrence-data/")
        print("2. Add your species to the 'species_name' column")
        print("3. Optionally fill in 'common_name' and 'color'")
        print("4. Run 'python procure.py' again")
        print(f"\n{'='*60}\n")
        return None
    
    return pd.read_csv(METADATA_FILE, dtype={'gbif_key': str})


def write_metadata(df):
    """Write metadata CSV"""
    df.to_csv(METADATA_FILE, index=False)

def update_species_metadata(species_name, updates):
    """Update metadata for a specific species"""
    df = pd.read_csv(METADATA_FILE)
    
    # Find the species row
    mask = df['species_name'] == species_name
    
    if not mask.any():
        # Species not in metadata, add it
        new_row = {
            'species_name': species_name,
            'common_name': '',
            'status': 'pending',
            'extent': '',
            'temporal_range': '',
            'expected_obs': 0,
            'actual_obs': 0,
            'last_updated': '',
            'taxonomic_rank': '',
            'gbif_key': '',
            'data_quality': 0.0,
            'countries_observed': '',
            'subregion': '',
            'color': '',
            'notes': ''
        }
        new_row.update(updates)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # Update existing row
        for key, value in updates.items():
            # Convert to string if it's gbif_key to avoid dtype warning
            if key == 'gbif_key':
                value = str(value)
            df.loc[mask, key] = value
    
    write_metadata(df)

def scan_existing_geojsons():
    """Scan for existing GeoJSON files and populate metadata retroactively"""
    if not DATA_DIR.exists():
        return
    
    geojson_files = list(DATA_DIR.glob('*.geojson'))
    if not geojson_files:
        return
    
    print(f"\nScanning {len(geojson_files)} existing GeoJSON files...")
    
    df = read_metadata()
    if df is None:
        return
    
    updated_count = 0
    added_count = 0
    
    for geojson_file in geojson_files:
        # Extract species name from filename (e.g., "morpho-gbif.geojson" -> "morpho")
        filename_base = geojson_file.stem.replace('-gbif', '')
        
        # Try to match with existing metadata entries
        found_match = False
        matched_species_name = None
        
        for idx, row in df.iterrows():
            species_name = row['species_name']
            safe_name = species_name.replace(' ', '-').lower()
            
            if safe_name == filename_base:
                found_match = True
                matched_species_name = species_name
                
                # Check if already has complete metadata
                if pd.notna(row['extent']) and row['extent'] != '':
                    continue
                
                try:
                    # Read the GeoJSON to get metadata
                    gdf = gpd.read_file(geojson_file)
                    
                    if len(gdf) == 0:
                        continue
                    
                    # Calculate extent
                    bounds = gdf.total_bounds
                    extent = f"[{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}]"
                    
                    # Calculate temporal range
                    if 'eventDate' in gdf.columns:
                        gdf['eventDate'] = pd.to_datetime(gdf['eventDate'], errors='coerce')
                        min_date = gdf['eventDate'].min().strftime('%Y-%m-%d')
                        max_date = gdf['eventDate'].max().strftime('%Y-%m-%d')
                        temporal_range = f"{min_date} to {max_date}"
                    else:
                        temporal_range = "unknown"
                    
                    # Determine rank if not set
                    rank = row['taxonomic_rank'] if pd.notna(row['taxonomic_rank']) and row['taxonomic_rank'] != '' else ('GENUS' if len(species_name.split()) == 1 else 'SPECIES')
                    
                    # Extract countries and subregion
                    countries_str, subregion = get_countries_and_subregion_from_geojson(gdf)
                    
                    updates = {
                        'status': 'complete',
                        'extent': extent,
                        'temporal_range': temporal_range,
                        'actual_obs': len(gdf),
                        'last_updated': time.strftime('%Y-%m-%d'),
                        'taxonomic_rank': rank,
                        'data_quality': 100.0,
                        'countries_observed': countries_str,
                        'subregion': subregion
                    }
                    
                    update_species_metadata(species_name, updates)
                    print(f"  ✓ Updated: {species_name} ({len(gdf):,} records)")
                    updated_count += 1
                    
                except Exception as e:
                    print(f"  ✗ Error reading {geojson_file.name}: {e}")
                break
        
        # If no match found, this is an untracked GeoJSON file - add it!
        if not found_match:
            try:
                # Read the GeoJSON to get metadata
                gdf = gpd.read_file(geojson_file)
                
                if len(gdf) == 0:
                    continue
                
                # Reconstruct species name from filename
                # Convert "morpho" -> "Morpho", "panthera-tigris" -> "Panthera tigris"
                species_name = filename_base.replace('-', ' ').title()
                
                # Calculate extent
                bounds = gdf.total_bounds
                extent = f"[{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}]"
                
                # Calculate temporal range
                if 'eventDate' in gdf.columns:
                    gdf['eventDate'] = pd.to_datetime(gdf['eventDate'], errors='coerce')
                    min_date = gdf['eventDate'].min().strftime('%Y-%m-%d')
                    max_date = gdf['eventDate'].max().strftime('%Y-%m-%d')
                    temporal_range = f"{min_date} to {max_date}"
                else:
                    temporal_range = "unknown"
                
                # Determine rank
                rank = 'GENUS' if len(species_name.split()) == 1 else 'SPECIES'
                
                # Extract countries and subregion
                countries_str, subregion = get_countries_and_subregion_from_geojson(gdf)
                
                # Add new entry
                new_entry = {
                    'species_name': species_name,
                    'common_name': '',
                    'status': 'complete',
                    'extent': extent,
                    'temporal_range': temporal_range,
                    'expected_obs': 0,  # Unknown since we didn't track the download
                    'actual_obs': len(gdf),
                    'last_updated': time.strftime('%Y-%m-%d'),
                    'taxonomic_rank': rank,
                    'gbif_key': '',
                    'data_quality': 100.0,  # Assume complete since file exists
                    'countries_observed': countries_str,
                    'subregion': subregion,
                    'color': '',
                    'notes': 'Auto-added from existing GeoJSON file'
                }
                
                # Read current metadata and append
                current_df = pd.read_csv(METADATA_FILE)
                new_df = pd.concat([current_df, pd.DataFrame([new_entry])], ignore_index=True)
                write_metadata(new_df)
                
                print(f"  + Added: {species_name} ({len(gdf):,} records)")
                added_count += 1
                
            except Exception as e:
                print(f"  ✗ Error processing untracked file {geojson_file.name}: {e}")
    
    if updated_count > 0 or added_count > 0:
        print(f"\n✓ Updated: {updated_count} | Added: {added_count} species\n")

# Genus/species functions
def is_genus_level(name):
    """Check if a name appears to be genus-level (single word)"""
    return len(name.split()) == 1

def get_genus_species(genus_name):
    """Get all species in a genus from GBIF"""
    try:
        # Search for the genus
        genus_search = species_api.name_backbone(genus_name)
        
        if not genus_search:
            return []
        
        # The response is nested - check 'usage' key
        if 'usage' in genus_search:
            usage = genus_search['usage']
            genus_key = usage.get('key')
        else:
            genus_key = genus_search.get('usageKey')
        
        if not genus_key:
            return []
        
        # Get all species in this genus
        species_list = []
        offset = 0
        limit = 100
        
        while True:
            result = species_api.name_usage(key=genus_key, data='children', limit=limit, offset=offset)
            
            if not result or 'results' not in result:
                break
            
            for sp in result['results']:
                if sp.get('rank') == 'SPECIES' and sp.get('canonicalName'):
                    species_list.append(sp.get('canonicalName'))
            
            if result.get('endOfRecords', True):
                break
            
            offset += limit
        
        return species_list
        
    except Exception as e:
        print(f"  Error getting genus species: {e}")
        return []

# Download functions
def download_year(species, year, countries=None):
    """Download up to 100k records for one year with retry logic"""
    records = []
    offset = 0
    
    while offset < 100000:
        retries = 0
        max_retries = 5
        
        while retries < max_retries:
            try:
                search_params = {
                    'scientificName': species,
                    'hasCoordinate': True,
                    'hasGeospatialIssue': False,
                    'year': str(year),
                    'limit': min(300, 100000 - offset),
                    'offset': offset
                }
                
                if countries:
                    search_params['country'] = countries
                
                batch = occ.search(**search_params)
                
                if not batch.get('results'):
                    return records
                
                records.extend([
                    r for r in batch['results']
                    if r.get('decimalLatitude') and r.get('decimalLongitude') and r.get('eventDate')
                ])
                
                offset += 300
                time.sleep(0.1)
                break
                
            except Exception as e:
                if '429' in str(e):
                    retries += 1
                    if retries < max_retries:
                        wait = 2 ** retries
                        time.sleep(wait)
                    else:
                        return records
                else:
                    return records
    
    return records

def convert_to_geojson(csv_file):
    """Convert CSV to GeoJSON and return metadata"""
    try:
        if csv_file.stat().st_size == 0:
            csv_file.unlink()
            return None

        df = pd.read_csv(csv_file, low_memory=False)

        # Add this — header-only file passes the size check but has no rows
        if df.empty:
            csv_file.unlink()
            return None
        
        keep_columns = ['decimalLatitude', 'decimalLongitude', 'eventDate']
        if 'countryCode' in df.columns:
            keep_columns.append('countryCode')
        
        if not all(col in df.columns for col in ['decimalLatitude', 'decimalLongitude', 'eventDate']):
            return None
        
        cleaned_df = df[keep_columns].copy()
        cleaned_df['eventDate'] = pd.to_datetime(cleaned_df['eventDate'], format='ISO8601', errors='coerce', utc=True)
        cleaned_df = cleaned_df.dropna(subset=['decimalLatitude', 'decimalLongitude', 'eventDate'])
        
        if len(cleaned_df) == 0:
            return None
        
        gdf = gpd.GeoDataFrame(
            cleaned_df,
            geometry=gpd.points_from_xy(cleaned_df.decimalLongitude, cleaned_df.decimalLatitude),
            crs="EPSG:4326"
        )
        
        output_file = csv_file.with_suffix('.geojson')
        gdf.to_file(output_file, driver='GeoJSON')
        csv_file.unlink()
        
        bounds = gdf.total_bounds
        extent = f"[{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}]"
        
        min_date = gdf['eventDate'].min().strftime('%Y-%m-%d')
        max_date = gdf['eventDate'].max().strftime('%Y-%m-%d')
        temporal_range = f"{min_date} to {max_date}"
        
        countries_str, subregion = get_countries_and_subregion_from_geojson(gdf)
        
        return {
            'extent': extent,
            'temporal_range': temporal_range,
            'actual_obs': len(gdf),
            'countries_observed': countries_str,
            'subregion': subregion
        }
        
    except Exception as e:
        print(f"  ✗ Conversion error: {e}")
        return None
def download_species(species, year_from, countries=None, convert=True):
    """Download occurrence data for a species or genus"""
    
    DATA_DIR.mkdir(exist_ok=True)
    safe_name = species.replace(' ', '-').lower()
    output = DATA_DIR / f'{safe_name}-gbif.csv'
    
    # Get GBIF key and determine rank
    try:
        backbone = species_api.name_backbone(species)
        if backbone and 'usage' in backbone:
            gbif_key = backbone['usage'].get('key', '')
            rank = backbone['usage'].get('rank', '')
        else:
            gbif_key = ''
            rank = 'GENUS' if is_genus_level(species) else 'SPECIES'
    except:
        gbif_key = ''
        rank = 'GENUS' if is_genus_level(species) else 'SPECIES'
    
    # Handle genus-level downloads
    if is_genus_level(species):
        species_in_genus = get_genus_species(species)
        
        if not species_in_genus:
            print(f"[{species}] ✗ Could not find species in genus")
            update_species_metadata(species, {
                'status': 'error',
                'last_updated': time.strftime('%Y-%m-%d'),
                'taxonomic_rank': 'GENUS',
                'gbif_key': str(gbif_key)
            })
            return 0
        
        print(f"[{species}] Found {len(species_in_genus)} species")
        
        # Get total expected
        total_expected = 0
        for sp in species_in_genus:
            try:
                params = {'scientificName': sp, 'hasCoordinate': True, 'year': f'{year_from},2025', 'limit': 1}
                if countries:
                    params['country'] = countries
                total_expected += occ.search(**params).get('count', 0)
            except:
                pass
        
        # Download all species
        all_records = []
        pbar = tqdm(total=total_expected, desc=f"{species} (genus)", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]',
                   ncols=80)
        
        try:
            for sp in species_in_genus:
                for year in range(year_from, 2026):
                    year_data = download_year(sp, year, countries)
                    all_records.extend(year_data)
                    pbar.update(len(year_data))
            pbar.close()
            
            if all_records:
                df = pd.DataFrame(all_records)
                df.to_csv(output, index=False)
                print(f"[{species}] ✓ {len(df):,} records")
                
                if convert:
                    metadata = convert_to_geojson(output)
                    if metadata:
                        update_species_metadata(species, {
                            'status': 'complete',
                            'expected_obs': total_expected,
                            'actual_obs': metadata['actual_obs'],
                            'extent': metadata['extent'],
                            'temporal_range': metadata['temporal_range'],
                            'last_updated': time.strftime('%Y-%m-%d'),
                            'taxonomic_rank': 'GENUS',
                            'gbif_key': str(gbif_key),
                            'data_quality': round(metadata['actual_obs'] / total_expected * 100, 1) if total_expected > 0 else 100.0,
                            'countries_observed': metadata.get('countries_observed', ''),
                            'subregion': metadata.get('subregion', '')
                        })
                        return metadata['actual_obs']
            
            if output.exists():
                output.unlink()
            return 0
            
        except Exception as e:
            pbar.close()
            print(f"[{species}] ✗ {e}")
            if output.exists():
                output.unlink()
            return 0
    
    # Regular species download
    try:
        params = {'scientificName': species, 'hasCoordinate': True, 'year': f'{year_from},2025', 'limit': 1}
        if countries:
            params['country'] = countries
        total = occ.search(**params).get('count', 0)
    except:
        total = 0
    
    all_records = []
    pbar = tqdm(total=total, desc=species[:30], 
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]',
               ncols=80)
    
    try:
        for year in range(year_from, 2026):
            year_data = download_year(species, year, countries)
            all_records.extend(year_data)
            pbar.update(len(year_data))
        pbar.close()
        
        df = pd.DataFrame(all_records)
        df.to_csv(output, index=False)
        
        print(f"[{species}] ✓ {len(df):,} records")
        
        if convert:
            metadata = convert_to_geojson(output)
            if metadata:
                update_species_metadata(species, {
                    'status': 'complete',
                    'expected_obs': total,
                    'actual_obs': metadata['actual_obs'],
                    'extent': metadata['extent'],
                    'temporal_range': metadata['temporal_range'],
                    'last_updated': time.strftime('%Y-%m-%d'),
                    'taxonomic_rank': rank,
                    'gbif_key': str(gbif_key),
                    'data_quality': round(metadata['actual_obs'] / total * 100, 1) if total > 0 else 100.0,
                    'countries_observed': metadata.get('countries_observed', ''),
                    'subregion': metadata.get('subregion', '')
                })
                return metadata['actual_obs']
            else:
                if output.exists():
                    output.unlink()
        
        if len(df) == 0 and output.exists():
            output.unlink()
        
        return len(df)
        
    except Exception as e:
        pbar.close()
        print(f"[{species}] ✗ {e}")
        if output.exists():
            output.unlink()
        return 0

def convert_existing():
    """Convert existing CSV files to GeoJSON"""
    print(f"\n{'='*60}")
    print("CONVERTING EXISTING CSV FILES TO GEOJSON")
    print(f"{'='*60}\n")
    
    if not DATA_DIR.exists():
        print(f"ERROR: Directory does not exist")
        return
    
    csv_files = list(DATA_DIR.glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found")
        return
    
    converted = 0
    for csv_file in csv_files:
        print(f"\n{csv_file.name}:")
        if convert_to_geojson(csv_file):
            converted += 1
    
    print(f"\n✓ Converted: {converted}")

def run_batch(year_from, countries, workers, force=False, skip_conversion=False):
    """Process species from metadata CSV"""
    
    # Read or create metadata
    df = read_metadata()
    if df is None:
        return
    
    # Scan existing GeoJSON files
    scan_existing_geojsons()
    
    # Reload metadata after scan
    df = read_metadata()
    
    # Determine what to download
    if force:
        pending = df['species_name'].tolist()
        skipped = []
        print("🔄 Force mode: Re-downloading all species")
    else:
        pending = df[df['status'] != 'complete']['species_name'].tolist()
        skipped = df[df['status'] == 'complete']['species_name'].tolist()
    
    if not pending:
        print(f"\n✓ All {len(df)} species up to date!")
        return
    
    # Show status
    country_display = ', '.join(countries) if countries else 'worldwide'
    print(f"\n{'='*60}")
    print(f"Species: {len(pending)} to download | {len(skipped)} complete")
    print(f"Settings: {year_from}-2025 | {country_display} | {workers} workers")
    print(f"{'='*60}\n")
    
    # Download
    total_records = 0
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_species, sp, year_from, countries, not skip_conversion): sp 
            for sp in pending
        }
        
        for future in as_completed(futures):
            try:
                records = future.result()
                if records > 0:
                    success_count += 1
                    total_records += records
            except KeyboardInterrupt:
                print("\n⏸ Interrupted")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as e:
                sp = futures[future]
                print(f"[{sp}] ✗ {e}")
    
    print(f"\n{'='*60}")
    print(f"✓ {success_count}/{len(pending)} successful | {total_records:,} total records")
    print(f"{'='*60}")

def show_status():
    """Show metadata status"""
    df = read_metadata()
    if df is None:
        return
    
    print(f"\n{'='*60}")
    print("METADATA STATUS")
    print(f"{'='*60}\n")
    
    for status in ['complete', 'pending', 'error']:
        subset = df[df['status'] == status]
        if len(subset) > 0:
            print(f"{status.upper()} ({len(subset)}):")
            for _, row in subset.iterrows():
                if status == 'complete':
                    print(f"  ✓ {row['species_name']}: {row['actual_obs']:,} obs ({row['data_quality']:.1f}%)")
                else:
                    print(f"  • {row['species_name']}")
            print()
    
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description='Download GBIF occurrence data for species',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-w', '--workers', type=int, default=3, help='Parallel downloads (default: 3)')
    parser.add_argument('-y', '--year', type=int, default=2015, help='Start year (default: 2015)')
    parser.add_argument('-c', '--countries', nargs='+', default=None, help='Country codes (default: worldwide)')
    parser.add_argument('--no-convert', action='store_true', help='Skip GeoJSON conversion')
    parser.add_argument('--convert-existing', action='store_true', help='Convert existing CSV files')
    parser.add_argument('--status', action='store_true', help='Show metadata status')
    parser.add_argument('--force', action='store_true', help='Re-download all species')
    
    args = parser.parse_args()
    
    if args.workers > 5:
        print(f"⚠️  Warning: {args.workers} workers may cause rate limiting\n")
        time.sleep(2)
    
    if args.convert_existing:
        convert_existing()
        return
    
    if args.status:
        show_status()
        return
    
    try:
        run_batch(args.year, args.countries, args.workers, args.force, args.no_convert)
    except KeyboardInterrupt:
        print("\n⏸ Interrupted")

if __name__ == '__main__':
    main()