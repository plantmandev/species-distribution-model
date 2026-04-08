import pandas as pd
import geopandas as gpd
from pathlib import Path
from pygbif import occurrences as occ
from pygbif import species as sp
import time

DATA_DIR = Path('occurrence-data')
METADATA_FILE = DATA_DIR / 'species-metadata.csv'

COMPLETE_THRESHOLD = 95.0
PARTIAL_THRESHOLD  = 80.0


def read_csv_meta(path):
    """Always read gbif_key as str to avoid dtype errors."""
    return pd.read_csv(path, dtype={'gbif_key': str})


def is_empty(value):
    """Return True if a metadata field is missing, NaN, or blank."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() in ('', 'nan')


def fetch_gbif_expected(species_name):
    """Query GBIF for the current total record count for a species."""
    try:
        result = occ.search(
            scientificName=species_name,
            hasCoordinate=True,
            hasGeospatialIssue=False,
            year='2015,2025',
            limit=1
        )
        return result.get('count', 0)
    except Exception as e:
        print(f"  ⚠ Could not fetch GBIF count for {species_name}: {e}")
        return None


def get_actual_count(species_name):
    """Read actual record count from GeoJSON file."""
    safe_name    = species_name.replace(' ', '-').lower()
    geojson_file = DATA_DIR / f'{safe_name}-gbif.geojson'

    if not geojson_file.exists():
        return None, False

    try:
        gdf = gpd.read_file(geojson_file)
        return len(gdf), True
    except Exception as e:
        print(f"  ⚠ Error reading {geojson_file.name}: {e}")
        return 0, True


def safe_int(value):
    """Convert a metadata field to int, returning 0 for NaN/None."""
    if is_empty(value):
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def classify(actual, expected):
    """Return (status_string, percent) based on completion."""
    if expected == 0:
        return 'complete', 100.0
    pct = actual / expected * 100
    if pct >= COMPLETE_THRESHOLD:
        return 'complete', pct
    elif pct >= PARTIAL_THRESHOLD:
        return 'partial', pct
    else:
        return 'incomplete', pct


# ---------------------------------------------------------------------------
# GBIF key population
# ---------------------------------------------------------------------------

def populate_gbif_keys():
    """
    Populate gbif_key only for rows where it is missing.
    Skips rows that already have a key.
    """
    df = read_csv_meta(METADATA_FILE)

    if 'gbif_key' not in df.columns:
        df['gbif_key'] = ''

    to_fill = [(idx, row) for idx, row in df.iterrows() if is_empty(row.get('gbif_key'))]

    if not to_fill:
        print("  ✓ All GBIF keys present, skipping.\n")
        return

    updated = 0

    print(f"\n{'='*60}")
    print(f"POPULATING GBIF KEYS ({len(to_fill)} missing)")
    print(f"{'='*60}\n")

    for idx, row in to_fill:
        species_name = row['species_name']

        try:
            result = sp.name_backbone(species_name)

            key = (
                (result.get('usage') or {}).get('key') or
                result.get('usageKey') or
                result.get('speciesKey') or
                result.get('acceptedUsageKey')
            )

            if key:
                df.at[idx, 'gbif_key'] = str(key)
                print(f"  ✓ {species_name:40s} → {key}")
                updated += 1
            else:
                df.at[idx, 'gbif_key'] = ''
                print(f"  ⚠ {species_name:40s}   (no match found)")

        except Exception as e:
            print(f"  ⚠ Error for {species_name}: {e}")

        time.sleep(0.2)

    df.to_csv(METADATA_FILE, index=False)
    print(f"\n  ✓ Populated {updated} GBIF keys.\n")


# ---------------------------------------------------------------------------
# Common name population
# ---------------------------------------------------------------------------

def populate_common_names():
    """
    Populate common_name only for rows where it is missing.
    Skips rows that already have a name.
    """
    df = read_csv_meta(METADATA_FILE)

    if 'common_name' not in df.columns:
        df['common_name'] = ''

    to_fill = [(idx, row) for idx, row in df.iterrows() if is_empty(row.get('common_name'))]

    if not to_fill:
        print("  ✓ All common names present, skipping.\n")
        return

    updated = 0

    print(f"\n{'='*60}")
    print(f"POPULATING COMMON NAMES ({len(to_fill)} missing)")
    print(f"{'='*60}\n")

    for idx, row in to_fill:
        gbif_key = str(row.get('gbif_key', '')).strip()
        if is_empty(gbif_key):
            print(f"  ⚠ No gbif_key for {row['species_name']}, skipping")
            continue

        try:
            result = sp.name_usage(key=gbif_key, data='vernacularNames')
            names  = result.get('results', [])

            english = [n['vernacularName'] for n in names if n.get('language') == 'eng']
            chosen  = english[0] if english else (names[0]['vernacularName'] if names else None)

            if chosen:
                df.at[idx, 'common_name'] = chosen
                print(f"  ✓ {row['species_name']:40s} → {chosen}")
                updated += 1
            else:
                df.at[idx, 'common_name'] = ''

        except Exception as e:
            print(f"  ⚠ Error for {row['species_name']}: {e}")

        time.sleep(0.2)

    df.to_csv(METADATA_FILE, index=False)
    print(f"\n  ✓ Populated {updated} common names.\n")


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def run_check():
    """
    Check all species in metadata against local GeoJSON files.
    Only fetches expected_obs from GBIF if it is missing.
    Marks incomplete/missing species as pending for procure.py.
    """
    df = read_csv_meta(METADATA_FILE)

    problems   = []
    complete   = 0
    ingested   = 0
    total_rows = len(df)

    print(f"\n{'='*60}")
    print(f"CHECKING {total_rows} SPECIES")
    print(f"{'='*60}\n")

    for idx, row in df.iterrows():
        species_name = row['species_name']
        prior_status = str(row.get('status', '')).strip()

        if prior_status == 'ingested':
            ingested += 1
            continue

        actual, file_exists = get_actual_count(species_name)

        if not file_exists:
            problems.append({
                'species':      species_name,
                'status':       'missing',
                'prior_status': prior_status,
                'actual':       0,
                'expected':     safe_int(row.get('expected_obs')),
                'pct':          0.0
            })
            df.at[idx, 'status'] = 'pending'
            continue

        # Only fetch expected count from GBIF if not already recorded
        if is_empty(row.get('expected_obs')):
            new_expected = fetch_gbif_expected(species_name)
            expected = new_expected if new_expected is not None else 0
            df.at[idx, 'expected_obs'] = expected
            time.sleep(0.2)
        else:
            expected = safe_int(row.get('expected_obs'))

        status, pct = classify(actual, expected)

        if status == 'complete':
            complete += 1
            df.at[idx, 'actual_obs']   = actual
            df.at[idx, 'status']       = 'complete'
            if expected > 0:
                df.at[idx, 'data_quality'] = round(pct, 1)
        else:
            problems.append({
                'species':      species_name,
                'status':       status,
                'prior_status': prior_status,
                'actual':       actual,
                'expected':     expected,
                'pct':          pct
            })
            df.at[idx, 'status'] = 'pending'

    df.to_csv(METADATA_FILE, index=False)

    missing    = sum(1 for p in problems if p['status'] == 'missing')
    incomplete = sum(1 for p in problems if p['status'] == 'incomplete')
    partial    = sum(1 for p in problems if p['status'] == 'partial')

    print(f"  ✓ Complete  : {complete}")
    print(f"  ✓ Ingested  : {ingested}  (in database, files removed)")
    print(f"  ⚠ Partial   : {partial}")
    print(f"  ✗ Incomplete: {incomplete}")
    print(f"  ✗ Missing   : {missing}")
    print(f"\n  Total species: {total_rows}")

    if problems:
        print(f"\n{'─'*60}")
        print("SPECIES NEEDING ATTENTION")
        print(f"{'─'*60}")
        for p in problems:
            sym  = '✗' if p['status'] in ('missing', 'incomplete') else '⚠'
            name = p['species']

            if p['status'] == 'missing':
                print(f"  {sym} {name:40s} ← marked pending")
            elif p['expected'] == 0:
                print(f"  {sym} {name:40s} {p['actual']:>7,} records  (no expected count)")
            else:
                print(f"  {sym} {name:40s} {p['actual']:>7,} / {p['expected']:>7,}  ({p['pct']:.1f}%)")

    if problems:
        print(f"\n  → {len(problems)} species marked pending. Run 'python3 procure.py' to download.")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if not METADATA_FILE.exists():
        print("Error: species-metadata.csv not found.")
    else:
        populate_gbif_keys()
        populate_common_names()
        run_check()