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

warnings.filterwarnings('ignore', category=FutureWarning)

DATA_DIR      = Path('occurrence-data')
METADATA_FILE = DATA_DIR / 'species-metadata.csv'

COUNTRY_TO_SUBREGION = {
    'DZ': 'Northern Africa', 'EG': 'Northern Africa', 'LY': 'Northern Africa',
    'MA': 'Northern Africa', 'SD': 'Northern Africa', 'TN': 'Northern Africa',
    'EH': 'Northern Africa',
    'AO': 'Sub-Saharan Africa', 'BJ': 'Sub-Saharan Africa', 'BW': 'Sub-Saharan Africa',
    'BF': 'Sub-Saharan Africa', 'BI': 'Sub-Saharan Africa', 'CM': 'Sub-Saharan Africa',
    'CV': 'Sub-Saharan Africa', 'CF': 'Sub-Saharan Africa', 'TD': 'Sub-Saharan Africa',
    'KM': 'Sub-Saharan Africa', 'CG': 'Sub-Saharan Africa', 'CD': 'Sub-Saharan Africa',
    'CI': 'Sub-Saharan Africa', 'DJ': 'Sub-Saharan Africa', 'GQ': 'Sub-Saharan Africa',
    'ER': 'Sub-Saharan Africa', 'ET': 'Sub-Saharan Africa', 'GA': 'Sub-Saharan Africa',
    'GM': 'Sub-Saharan Africa', 'GH': 'Sub-Saharan Africa', 'GN': 'Sub-Saharan Africa',
    'GW': 'Sub-Saharan Africa', 'KE': 'Sub-Saharan Africa', 'LS': 'Sub-Saharan Africa',
    'LR': 'Sub-Saharan Africa', 'MG': 'Sub-Saharan Africa', 'MW': 'Sub-Saharan Africa',
    'ML': 'Sub-Saharan Africa', 'MR': 'Sub-Saharan Africa', 'MU': 'Sub-Saharan Africa',
    'YT': 'Sub-Saharan Africa', 'MZ': 'Sub-Saharan Africa', 'NA': 'Sub-Saharan Africa',
    'NE': 'Sub-Saharan Africa', 'NG': 'Sub-Saharan Africa', 'RE': 'Sub-Saharan Africa',
    'RW': 'Sub-Saharan Africa', 'ST': 'Sub-Saharan Africa', 'SN': 'Sub-Saharan Africa',
    'SC': 'Sub-Saharan Africa', 'SL': 'Sub-Saharan Africa', 'SO': 'Sub-Saharan Africa',
    'ZA': 'Sub-Saharan Africa', 'SS': 'Sub-Saharan Africa', 'SZ': 'Sub-Saharan Africa',
    'TZ': 'Sub-Saharan Africa', 'TG': 'Sub-Saharan Africa', 'UG': 'Sub-Saharan Africa',
    'ZM': 'Sub-Saharan Africa', 'ZW': 'Sub-Saharan Africa',
    'BM': 'Northern America', 'CA': 'Northern America', 'GL': 'Northern America',
    'PM': 'Northern America', 'US': 'Northern America',
    'BZ': 'Central America', 'CR': 'Central America', 'SV': 'Central America',
    'GT': 'Central America', 'HN': 'Central America', 'MX': 'Central America',
    'NI': 'Central America', 'PA': 'Central America',
    'AR': 'South America', 'BO': 'South America', 'BR': 'South America',
    'CL': 'South America', 'CO': 'South America', 'EC': 'South America',
    'FK': 'South America', 'GF': 'South America', 'GY': 'South America',
    'PY': 'South America', 'PE': 'South America', 'SR': 'South America',
    'UY': 'South America', 'VE': 'South America',
    'AI': 'Caribbean', 'AG': 'Caribbean', 'AW': 'Caribbean', 'BS': 'Caribbean',
    'BB': 'Caribbean', 'BQ': 'Caribbean', 'VG': 'Caribbean', 'KY': 'Caribbean',
    'CU': 'Caribbean', 'CW': 'Caribbean', 'DM': 'Caribbean', 'DO': 'Caribbean',
    'GD': 'Caribbean', 'GP': 'Caribbean', 'HT': 'Caribbean', 'JM': 'Caribbean',
    'MQ': 'Caribbean', 'MS': 'Caribbean', 'PR': 'Caribbean', 'BL': 'Caribbean',
    'KN': 'Caribbean', 'LC': 'Caribbean', 'MF': 'Caribbean', 'VC': 'Caribbean',
    'SX': 'Caribbean', 'TT': 'Caribbean', 'TC': 'Caribbean', 'VI': 'Caribbean',
    'KZ': 'Central Asia', 'KG': 'Central Asia', 'TJ': 'Central Asia',
    'TM': 'Central Asia', 'UZ': 'Central Asia',
    'CN': 'Eastern Asia', 'HK': 'Eastern Asia', 'MO': 'Eastern Asia',
    'KP': 'Eastern Asia', 'KR': 'Eastern Asia', 'MN': 'Eastern Asia',
    'JP': 'Eastern Asia', 'TW': 'Eastern Asia',
    'BN': 'South-Eastern Asia', 'KH': 'South-Eastern Asia', 'ID': 'South-Eastern Asia',
    'LA': 'South-Eastern Asia', 'MY': 'South-Eastern Asia', 'MM': 'South-Eastern Asia',
    'PH': 'South-Eastern Asia', 'SG': 'South-Eastern Asia', 'TH': 'South-Eastern Asia',
    'TL': 'South-Eastern Asia', 'VN': 'South-Eastern Asia',
    'AF': 'Southern Asia', 'BD': 'Southern Asia', 'BT': 'Southern Asia',
    'IN': 'Southern Asia', 'IR': 'Southern Asia', 'MV': 'Southern Asia',
    'NP': 'Southern Asia', 'PK': 'Southern Asia', 'LK': 'Southern Asia',
    'AM': 'Western Asia', 'AZ': 'Western Asia', 'BH': 'Western Asia',
    'CY': 'Western Asia', 'GE': 'Western Asia', 'IQ': 'Western Asia',
    'IL': 'Western Asia', 'JO': 'Western Asia', 'KW': 'Western Asia',
    'LB': 'Western Asia', 'OM': 'Western Asia', 'PS': 'Western Asia',
    'QA': 'Western Asia', 'SA': 'Western Asia', 'SY': 'Western Asia',
    'TR': 'Western Asia', 'AE': 'Western Asia', 'YE': 'Western Asia',
    'BY': 'Eastern Europe', 'BG': 'Eastern Europe', 'CZ': 'Eastern Europe',
    'HU': 'Eastern Europe', 'PL': 'Eastern Europe', 'MD': 'Eastern Europe',
    'RO': 'Eastern Europe', 'RU': 'Eastern Europe', 'SK': 'Eastern Europe',
    'UA': 'Eastern Europe',
    'AX': 'Northern Europe', 'DK': 'Northern Europe', 'EE': 'Northern Europe',
    'FO': 'Northern Europe', 'FI': 'Northern Europe', 'GG': 'Northern Europe',
    'IS': 'Northern Europe', 'IE': 'Northern Europe', 'IM': 'Northern Europe',
    'JE': 'Northern Europe', 'LV': 'Northern Europe', 'LT': 'Northern Europe',
    'NO': 'Northern Europe', 'SJ': 'Northern Europe', 'SE': 'Northern Europe',
    'GB': 'Northern Europe',
    'AL': 'Southern Europe', 'AD': 'Southern Europe', 'BA': 'Southern Europe',
    'HR': 'Southern Europe', 'GI': 'Southern Europe', 'GR': 'Southern Europe',
    'VA': 'Southern Europe', 'IT': 'Southern Europe', 'MT': 'Southern Europe',
    'ME': 'Southern Europe', 'MK': 'Southern Europe', 'PT': 'Southern Europe',
    'SM': 'Southern Europe', 'RS': 'Southern Europe', 'SI': 'Southern Europe',
    'ES': 'Southern Europe',
    'AT': 'Western Europe', 'BE': 'Western Europe', 'FR': 'Western Europe',
    'DE': 'Western Europe', 'LI': 'Western Europe', 'LU': 'Western Europe',
    'MC': 'Western Europe', 'NL': 'Western Europe', 'CH': 'Western Europe',
    'AU': 'Australia and New Zealand', 'CX': 'Australia and New Zealand',
    'CC': 'Australia and New Zealand', 'HM': 'Australia and New Zealand',
    'NZ': 'Australia and New Zealand', 'NF': 'Australia and New Zealand',
    'FJ': 'Melanesia', 'NC': 'Melanesia', 'PG': 'Melanesia',
    'SB': 'Melanesia', 'VU': 'Melanesia',
    'GU': 'Micronesia', 'KI': 'Micronesia', 'MH': 'Micronesia',
    'FM': 'Micronesia', 'NR': 'Micronesia', 'MP': 'Micronesia',
    'PW': 'Micronesia', 'UM': 'Micronesia',
    'AS': 'Polynesia', 'CK': 'Polynesia', 'PF': 'Polynesia', 'NU': 'Polynesia',
    'PN': 'Polynesia', 'WS': 'Polynesia', 'TK': 'Polynesia', 'TO': 'Polynesia',
    'TV': 'Polynesia', 'WF': 'Polynesia',
    'AQ': 'Antarctica', 'BV': 'Antarctica', 'TF': 'Antarctica', 'GS': 'Antarctica',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv_meta(path):
    """Single place to read metadata CSV — always enforces gbif_key as str."""
    return pd.read_csv(path, dtype={'gbif_key': str})


def get_countries_and_subregion_from_geojson(gdf):
    """Extract unique country codes and determine primary subregion."""
    try:
        if 'countryCode' in gdf.columns:
            countries = gdf['countryCode'].dropna().unique().tolist()
            countries = [c for c in countries if c]
        else:
            countries = []

        if not countries:
            return '', ''

        countries_str = ','.join(sorted(countries))
        subregions    = [COUNTRY_TO_SUBREGION.get(c, '') for c in countries]
        subregions    = [s for s in subregions if s]

        if subregions:
            from collections import Counter
            subregion = Counter(subregions).most_common(1)[0][0]
        else:
            subregion = ''

        return countries_str, subregion

    except Exception:
        return '', ''


def get_downloaded_years(species_name):
    """
    Read existing GeoJSON and return the set of years already present.
    Returns (set_of_years, existing_geodataframe_or_None).
    """
    safe_name    = species_name.replace(' ', '-').lower()
    geojson_file = DATA_DIR / f'{safe_name}-gbif.geojson'

    if not geojson_file.exists():
        return set(), None

    try:
        gdf = gpd.read_file(geojson_file)
        if 'eventDate' not in gdf.columns or len(gdf) == 0:
            return set(), gdf
        gdf['eventDate'] = pd.to_datetime(gdf['eventDate'], errors='coerce', utc=True)
        years = set(gdf['eventDate'].dropna().dt.year.astype(int).unique())
        return years, gdf
    except Exception:
        return set(), None


def flatten_gdf(gdf):
    """Convert a GeoDataFrame back to a flat DataFrame for merging."""
    flat = pd.DataFrame({
        'decimalLatitude':  gdf.geometry.y,
        'decimalLongitude': gdf.geometry.x,
        'eventDate':        gdf['eventDate'].astype(str),
    })
    if 'countryCode' in gdf.columns:
        flat['countryCode'] = gdf['countryCode'].values
    return flat


# ---------------------------------------------------------------------------
# Metadata CSV
# ---------------------------------------------------------------------------

def create_metadata_template():
    template = pd.DataFrame({
        'species_name':       ['Morpho', 'Panthera tigris'],
        'common_name':        ['Blue Morpho Butterflies', 'Tiger'],
        'status':             ['pending', 'pending'],
        'extent':             ['', ''],
        'temporal_range':     ['', ''],
        'expected_obs':       [0, 0],
        'actual_obs':         [0, 0],
        'last_updated':       ['', ''],
        'taxonomic_rank':     ['', ''],
        'gbif_key':           ['', ''],
        'data_quality':       [0.0, 0.0],
        'countries_observed': ['', ''],
        'subregion':          ['', ''],
        'color':              ['#0066FF', '#FF6600'],
        'notes':              ['', '']
    })
    DATA_DIR.mkdir(exist_ok=True)
    template.to_csv(METADATA_FILE, index=False)
    return template


def read_metadata():
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
    return read_csv_meta(METADATA_FILE)


def write_metadata(df):
    df.to_csv(METADATA_FILE, index=False)


def update_species_metadata(species_name, updates):
    df   = read_csv_meta(METADATA_FILE)
    mask = df['species_name'] == species_name

    if not mask.any():
        new_row = {
            'species_name': species_name, 'common_name': '', 'status': 'pending',
            'extent': '', 'temporal_range': '', 'expected_obs': 0, 'actual_obs': 0,
            'last_updated': '', 'taxonomic_rank': '', 'gbif_key': '',
            'data_quality': 0.0, 'countries_observed': '', 'subregion': '',
            'color': '', 'notes': ''
        }
        new_row.update(updates)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        for key, value in updates.items():
            if key == 'gbif_key':
                value = str(value)
            df.loc[mask, key] = value

    write_metadata(df)


def scan_existing_geojsons():
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
    added_count   = 0

    for geojson_file in geojson_files:
        filename_base = geojson_file.stem.replace('-gbif', '')
        found_match   = False

        for idx, row in df.iterrows():
            species_name = row['species_name']
            safe_name    = species_name.replace(' ', '-').lower()

            if safe_name == filename_base:
                found_match = True

                if pd.notna(row['extent']) and row['extent'] != '':
                    break

                try:
                    gdf = gpd.read_file(geojson_file)
                    if len(gdf) == 0:
                        break

                    bounds         = gdf.total_bounds
                    extent         = f"[{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}]"

                    if 'eventDate' in gdf.columns:
                        gdf['eventDate'] = pd.to_datetime(gdf['eventDate'], errors='coerce')
                        min_date         = gdf['eventDate'].min().strftime('%Y-%m-%d')
                        max_date         = gdf['eventDate'].max().strftime('%Y-%m-%d')
                        temporal_range   = f"{min_date} to {max_date}"
                    else:
                        temporal_range = "unknown"

                    rank = (row['taxonomic_rank']
                            if pd.notna(row['taxonomic_rank']) and row['taxonomic_rank'] != ''
                            else ('GENUS' if len(species_name.split()) == 1 else 'SPECIES'))

                    countries_str, subregion = get_countries_and_subregion_from_geojson(gdf)

                    update_species_metadata(species_name, {
                        'status': 'complete', 'extent': extent,
                        'temporal_range': temporal_range, 'actual_obs': len(gdf),
                        'last_updated': time.strftime('%Y-%m-%d'), 'taxonomic_rank': rank,
                        'data_quality': 100.0, 'countries_observed': countries_str,
                        'subregion': subregion
                    })
                    print(f"  ✓ Updated: {species_name} ({len(gdf):,} records)")
                    updated_count += 1

                except Exception as e:
                    print(f"  ✗ Error reading {geojson_file.name}: {e}")
                break

        if not found_match:
            try:
                gdf = gpd.read_file(geojson_file)
                if len(gdf) == 0:
                    continue

                species_name   = filename_base.replace('-', ' ').title()
                bounds         = gdf.total_bounds
                extent         = f"[{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}]"

                if 'eventDate' in gdf.columns:
                    gdf['eventDate'] = pd.to_datetime(gdf['eventDate'], errors='coerce')
                    min_date         = gdf['eventDate'].min().strftime('%Y-%m-%d')
                    max_date         = gdf['eventDate'].max().strftime('%Y-%m-%d')
                    temporal_range   = f"{min_date} to {max_date}"
                else:
                    temporal_range = "unknown"

                rank                     = 'GENUS' if len(species_name.split()) == 1 else 'SPECIES'
                countries_str, subregion = get_countries_and_subregion_from_geojson(gdf)

                new_entry = {
                    'species_name': species_name, 'common_name': '', 'status': 'complete',
                    'extent': extent, 'temporal_range': temporal_range, 'expected_obs': 0,
                    'actual_obs': len(gdf), 'last_updated': time.strftime('%Y-%m-%d'),
                    'taxonomic_rank': rank, 'gbif_key': '', 'data_quality': 100.0,
                    'countries_observed': countries_str, 'subregion': subregion,
                    'color': '', 'notes': 'Auto-added from existing GeoJSON file'
                }

                current_df = read_csv_meta(METADATA_FILE)
                new_df     = pd.concat([current_df, pd.DataFrame([new_entry])], ignore_index=True)
                write_metadata(new_df)

                print(f"  + Added: {species_name} ({len(gdf):,} records)")
                added_count += 1

            except Exception as e:
                print(f"  ✗ Error processing untracked file {geojson_file.name}: {e}")

    if updated_count > 0 or added_count > 0:
        print(f"\n✓ Updated: {updated_count} | Added: {added_count} species\n")


# ---------------------------------------------------------------------------
# Genus helpers
# ---------------------------------------------------------------------------

def is_genus_level(name):
    return len(name.split()) == 1


def get_genus_species(genus_name):
    try:
        genus_search = species_api.name_backbone(genus_name)
        if not genus_search:
            return []

        genus_key = (genus_search['usage'].get('key')
                     if 'usage' in genus_search
                     else genus_search.get('usageKey'))

        if not genus_key:
            return []

        species_list  = []
        offset, limit = 0, 100

        while True:
            result = species_api.name_usage(key=genus_key, data='children',
                                            limit=limit, offset=offset)
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


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_year(species, year, countries=None):
    """Download up to 100k records for one year with retry/backoff."""
    records, offset = [], 0

    while offset < 100000:
        retries = 0
        while retries < 5:
            try:
                params = {
                    'scientificName':     species,
                    'hasCoordinate':      True,
                    'hasGeospatialIssue': False,
                    'year':               str(year),
                    'limit':              min(300, 100000 - offset),
                    'offset':             offset
                }
                if countries:
                    params['country'] = countries

                batch = occ.search(**params)
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
                    if retries < 5:
                        time.sleep(2 ** retries)
                    else:
                        return records
                else:
                    return records

    return records


def convert_to_geojson(csv_file):
    """Convert a CSV to GeoJSON, return metadata dict or None."""
    try:
        if csv_file.stat().st_size == 0:
            csv_file.unlink()
            return None

        df = pd.read_csv(csv_file, low_memory=False)

        if df.empty:
            csv_file.unlink()
            return None

        keep_columns = ['decimalLatitude', 'decimalLongitude', 'eventDate']
        if 'countryCode' in df.columns:
            keep_columns.append('countryCode')

        if not all(col in df.columns for col in ['decimalLatitude', 'decimalLongitude', 'eventDate']):
            return None

        cleaned             = df[keep_columns].copy()
        cleaned['eventDate'] = pd.to_datetime(
            cleaned['eventDate'], format='ISO8601', errors='coerce', utc=True
        )
        cleaned = cleaned.dropna(subset=['decimalLatitude', 'decimalLongitude', 'eventDate'])

        if len(cleaned) == 0:
            return None

        gdf = gpd.GeoDataFrame(
            cleaned,
            geometry=gpd.points_from_xy(cleaned.decimalLongitude, cleaned.decimalLatitude),
            crs="EPSG:4326"
        )

        out = csv_file.with_suffix('.geojson')
        gdf.to_file(out, driver='GeoJSON')
        csv_file.unlink()

        bounds         = gdf.total_bounds
        extent         = f"[{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}]"
        min_date       = gdf['eventDate'].min().strftime('%Y-%m-%d')
        max_date       = gdf['eventDate'].max().strftime('%Y-%m-%d')
        temporal_range = f"{min_date} to {max_date}"
        countries_str, subregion = get_countries_and_subregion_from_geojson(gdf)

        return {
            'extent': extent, 'temporal_range': temporal_range,
            'actual_obs': len(gdf), 'countries_observed': countries_str,
            'subregion': subregion
        }

    except Exception as e:
        print(f"  ✗ Conversion error: {e}")
        return None


# ---------------------------------------------------------------------------
# Download — resume-aware (species and genus)
# ---------------------------------------------------------------------------

def download_species(species, year_from, countries=None, convert=True):
    """
    Download GBIF occurrences for a species or genus.
    Checks what years already exist in the GeoJSON and only fetches
    the missing ones, then merges with existing data.
    """
    DATA_DIR.mkdir(exist_ok=True)
    safe_name   = species.replace(' ', '-').lower()
    output      = DATA_DIR / f'{safe_name}-gbif.csv'
    geojson_out = DATA_DIR / f'{safe_name}-gbif.geojson'

    # GBIF backbone lookup
    try:
        backbone = species_api.name_backbone(species)
        if backbone and 'usage' in backbone:
            gbif_key = backbone['usage'].get('key', '')
            rank     = backbone['usage'].get('rank', '')
        else:
            gbif_key = ''
            rank     = 'GENUS' if is_genus_level(species) else 'SPECIES'
    except Exception:
        gbif_key = ''
        rank     = 'GENUS' if is_genus_level(species) else 'SPECIES'

    # -----------------------------------------------------------------------
    # Genus-level path
    # -----------------------------------------------------------------------
    if is_genus_level(species):
        species_in_genus = get_genus_species(species)

        if not species_in_genus:
            print(f"[{species}] ✗ Could not find species in genus")
            update_species_metadata(species, {
                'status': 'error', 'last_updated': time.strftime('%Y-%m-%d'),
                'taxonomic_rank': 'GENUS', 'gbif_key': str(gbif_key)
            })
            return 0

        print(f"[{species}] Found {len(species_in_genus)} species in genus")

        downloaded_years, existing_gdf = get_downloaded_years(species)
        missing_years = sorted(set(range(year_from, 2026)) - downloaded_years)

        if not missing_years:
            print(f"[{species}] ✓ All years present, nothing to download")
            return len(existing_gdf) if existing_gdf is not None else 0

        if downloaded_years:
            print(f"[{species}] Resuming — missing years: {missing_years}")

        total_expected = 0
        for sp in species_in_genus:
            try:
                p = {'scientificName': sp, 'hasCoordinate': True,
                     'year': f'{year_from},2025', 'limit': 1}
                if countries:
                    p['country'] = countries
                total_expected += occ.search(**p).get('count', 0)
            except Exception:
                pass

        new_records = []
        pbar = tqdm(total=total_expected, desc=f"{species} (genus)",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]',
                    ncols=80)

        try:
            for sp in species_in_genus:
                for year in missing_years:
                    data = download_year(sp, year, countries)
                    new_records.extend(data)
                    pbar.update(len(data))
            pbar.close()

            new_df = pd.DataFrame(new_records) if new_records else pd.DataFrame()

            if existing_gdf is not None and len(existing_gdf) > 0:
                merged_df = (pd.concat([flatten_gdf(existing_gdf), new_df], ignore_index=True)
                             if not new_df.empty else flatten_gdf(existing_gdf))
            else:
                merged_df = new_df

            if merged_df.empty:
                if output.exists():
                    output.unlink()
                return 0

            merged_df.to_csv(output, index=False)
            existing_count = len(existing_gdf) if existing_gdf is not None else 0
            print(f"[{species}] ✓ {len(new_records):,} new + {existing_count:,} existing = {len(merged_df):,} total")

            if convert:
                if geojson_out.exists():
                    geojson_out.unlink()
                metadata = convert_to_geojson(output)
                if metadata:
                    update_species_metadata(species, {
                        'status': 'complete', 'expected_obs': total_expected,
                        'actual_obs': metadata['actual_obs'], 'extent': metadata['extent'],
                        'temporal_range': metadata['temporal_range'],
                        'last_updated': time.strftime('%Y-%m-%d'), 'taxonomic_rank': 'GENUS',
                        'gbif_key': str(gbif_key),
                        'data_quality': round(metadata['actual_obs'] / total_expected * 100, 1)
                                        if total_expected > 0 else 100.0,
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

    # -----------------------------------------------------------------------
    # Species-level path — resume-aware
    # -----------------------------------------------------------------------
    downloaded_years, existing_gdf = get_downloaded_years(species)
    missing_years = sorted(set(range(year_from, 2026)) - downloaded_years)

    if not missing_years:
        print(f"[{species}] ✓ All years present, nothing to download")
        return len(existing_gdf) if existing_gdf is not None else 0

    if downloaded_years:
        print(f"[{species}] Resuming — missing years: {missing_years}")

    # Expected count for missing years (for progress bar)
    try:
        p = {
            'scientificName': species, 'hasCoordinate': True,
            'year': f'{min(missing_years)},{max(missing_years)}', 'limit': 1
        }
        if countries:
            p['country'] = countries
        total = occ.search(**p).get('count', 0)
    except Exception:
        total = 0

    new_records = []
    pbar = tqdm(total=total, desc=species[:30],
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]',
                ncols=80)

    try:
        for year in missing_years:
            data = download_year(species, year, countries)
            new_records.extend(data)
            pbar.update(len(data))
        pbar.close()

        new_df = pd.DataFrame(new_records) if new_records else pd.DataFrame()

        if existing_gdf is not None and len(existing_gdf) > 0:
            merged_df = (pd.concat([flatten_gdf(existing_gdf), new_df], ignore_index=True)
                         if not new_df.empty else flatten_gdf(existing_gdf))
        else:
            merged_df = new_df

        if merged_df.empty:
            if output.exists():
                output.unlink()
            return 0

        merged_df.to_csv(output, index=False)
        existing_count = len(existing_gdf) if existing_gdf is not None else 0
        print(f"[{species}] ✓ {len(new_records):,} new + {existing_count:,} existing = {len(merged_df):,} total")

        if convert:
            if geojson_out.exists():
                geojson_out.unlink()
            metadata = convert_to_geojson(output)
            if metadata:
                # Full expected count for data_quality calculation
                try:
                    full_expected = occ.search(
                        scientificName=species, hasCoordinate=True,
                        year=f'{year_from},2025', limit=1
                    ).get('count', 0)
                except Exception:
                    full_expected = 0

                update_species_metadata(species, {
                    'status': 'complete', 'expected_obs': full_expected,
                    'actual_obs': metadata['actual_obs'], 'extent': metadata['extent'],
                    'temporal_range': metadata['temporal_range'],
                    'last_updated': time.strftime('%Y-%m-%d'), 'taxonomic_rank': rank,
                    'gbif_key': str(gbif_key),
                    'data_quality': round(metadata['actual_obs'] / full_expected * 100, 1)
                                    if full_expected > 0 else 100.0,
                    'countries_observed': metadata.get('countries_observed', ''),
                    'subregion': metadata.get('subregion', '')
                })
                return metadata['actual_obs']
            else:
                if output.exists():
                    output.unlink()

        if len(merged_df) == 0 and output.exists():
            output.unlink()

        return len(merged_df)

    except Exception as e:
        pbar.close()
        print(f"[{species}] ✗ {e}")
        if output.exists():
            output.unlink()
        return 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def convert_existing():
    print(f"\n{'='*60}")
    print("CONVERTING EXISTING CSV FILES TO GEOJSON")
    print(f"{'='*60}\n")

    if not DATA_DIR.exists():
        print("ERROR: Directory does not exist")
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
    df = read_metadata()
    if df is None:
        return

    scan_existing_geojsons()
    df = read_metadata()

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

    country_display = ', '.join(countries) if countries else 'worldwide'
    print(f"\n{'='*60}")
    print(f"Species: {len(pending)} to download | {len(skipped)} complete")
    print(f"Settings: {year_from}-2025 | {country_display} | {workers} workers")
    print(f"{'='*60}\n")

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Download GBIF occurrence data for species',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-w', '--workers',    type=int,  default=3,    help='Parallel downloads (default: 3)')
    parser.add_argument('-y', '--year',       type=int,  default=2015, help='Start year (default: 2015)')
    parser.add_argument('-c', '--countries',  nargs='+', default=None, help='Country codes (default: worldwide)')
    parser.add_argument('--no-convert',       action='store_true',     help='Skip GeoJSON conversion')
    parser.add_argument('--convert-existing', action='store_true',     help='Convert existing CSV files')
    parser.add_argument('--status',           action='store_true',     help='Show metadata status')
    parser.add_argument('--force',            action='store_true',     help='Re-download all species')
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
