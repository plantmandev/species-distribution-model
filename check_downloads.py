import pandas as pd
import geopandas as gpd
from pathlib import Path
from pygbif import occurrences as occ
import argparse
import time

DATA_DIR = Path('occurrence-data')
METADATA_FILE = DATA_DIR / 'species-metadata.csv'

COMPLETE_THRESHOLD = 95.0
PARTIAL_THRESHOLD  = 80.0


def read_csv_meta(path):
    """Always read gbif_key as str to avoid dtype errors."""
    return pd.read_csv(path, dtype={'gbif_key': str})


def fetch_gbif_expected(species_name, year_from=2015):
    """Query GBIF for the current total record count for a species."""
    try:
        result = occ.search(
            scientificName=species_name,
            hasCoordinate=True,
            hasGeospatialIssue=False,
            year=f'{year_from},2025',
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
    if pd.isna(value) if value is not None else True:
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
# Purge
# ---------------------------------------------------------------------------

def purge_dead_entries():
    """
    Remove metadata rows where no GeoJSON file exists and status is
    'error', 'pending', or blank — i.e. bad names or 0-record species.
    Prints what it removed before writing.
    """
    if not METADATA_FILE.exists():
        print("Error: species-metadata.csv not found.")
        return

    df        = read_csv_meta(METADATA_FILE)
    to_remove = []

    for idx, row in df.iterrows():
        _, file_exists = get_actual_count(row['species_name'])
        if not file_exists:
            prior = str(row.get('status', '')).strip()
            if prior in ('error', 'pending', ''):
                to_remove.append((idx, row['species_name'], prior))

    if not to_remove:
        print("Nothing to purge.")
        return

    print(f"\nPurging {len(to_remove)} dead entries:")
    indices_to_drop = []
    for idx, name, status in to_remove:
        print(f"  ✗ {name}  (status: '{status}')")
        indices_to_drop.append(idx)

    df = df.drop(indices_to_drop)
    df.to_csv(METADATA_FILE, index=False)
    print(f"\n✓ Removed {len(to_remove)} entries from metadata.")
    print("  Fix any typos and re-add them to the CSV to retry.\n")


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def run_check(refresh_expected=False, mark_pending=False, year_from=2015):
    """
    Check all species in metadata against local GeoJSON files.

    refresh_expected  — query GBIF live and update expected_obs in the CSV
    mark_pending      — set status=pending for incomplete/missing species
                        so procure.py will pick them up automatically
    """
    if not METADATA_FILE.exists():
        print("Error: species-metadata.csv not found. Run procure.py first.")
        return

    df = read_csv_meta(METADATA_FILE)

    problems   = []
    complete   = 0
    ingested   = 0
    total_rows = len(df)

    print(f"\n{'='*60}")
    print(f"CHECKING {total_rows} SPECIES")
    if refresh_expected:
        print("Mode: refreshing expected counts from GBIF")
    print(f"{'='*60}\n")

    for idx, row in df.iterrows():
        species_name = row['species_name']
        prior_status = str(row.get('status', '')).strip()

        # Already in the database — file deletion is expected, not a problem
        if prior_status == 'ingested':
            ingested += 1
            continue

        actual, file_exists = get_actual_count(species_name)

        # --- File not found ---
        if not file_exists:
            problems.append({
                'species':      species_name,
                'status':       'missing',
                'prior_status': prior_status,
                'actual':       0,
                'expected':     safe_int(row.get('expected_obs')),
                'pct':          0.0
            })
            if mark_pending:
                df.at[idx, 'status'] = 'pending'
            continue

        # --- Expected count ---
        if refresh_expected:
            new_expected = fetch_gbif_expected(species_name, year_from)
            if new_expected is not None:
                df.at[idx, 'expected_obs'] = new_expected
                expected = new_expected
            else:
                expected = safe_int(row.get('expected_obs'))
            time.sleep(0.2)
        else:
            expected = safe_int(row.get('expected_obs'))

        status, pct = classify(actual, expected)

        if status == 'complete':
            complete += 1
            df.at[idx, 'actual_obs'] = actual
            df.at[idx, 'status']     = 'complete'
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
            if mark_pending:
                df.at[idx, 'status'] = 'pending'

    # Write updated metadata
    df.to_csv(METADATA_FILE, index=False)

    # --- Summary ---
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
                prior = p.get('prior_status', '')
                if prior == 'error':
                    hint = '← bad name / 0 GBIF records  (remove with --purge)'
                elif prior in ('pending', ''):
                    hint = '← never downloaded  (run with --fix)'
                else:
                    hint = f'← was {prior}'
                print(f"  {sym} {name:35s} {hint}")

            elif p['expected'] == 0:
                print(f"  {sym} {name:35s} {p['actual']:>7,} records  (no expected count)")

            else:
                print(f"  {sym} {name:35s} {p['actual']:>7,} / {p['expected']:>7,}  ({p['pct']:.1f}%)")

    if mark_pending:
        print(f"\n  → Marked {len(problems)} species as pending.")
        print("    Run 'python procure.py' to re-download them.")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Check GBIF download completeness against metadata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_downloads.py                   # quick local check
  python check_downloads.py --refresh         # update expected counts from GBIF
  python check_downloads.py --refresh --fix   # refresh + mark incomplete as pending
                                              # then run: python procure.py
  python check_downloads.py --purge           # remove dead metadata entries (bad names, 0-record species)
        """
    )
    parser.add_argument(
        '--refresh', action='store_true',
        help='Query GBIF live to update expected_obs in metadata CSV'
    )
    parser.add_argument(
        '--fix', action='store_true',
        help='Mark incomplete/missing species as pending so procure.py re-downloads them'
    )
    parser.add_argument(
        '--purge', action='store_true',
        help='Remove metadata rows with no file and status error/pending (bad names, 0-record species)'
    )
    parser.add_argument(
        '-y', '--year', type=int, default=2015,
        help='Start year for GBIF count queries (default: 2015)'
    )
    args = parser.parse_args()

    if args.purge:
        purge_dead_entries()
        return

    run_check(
        refresh_expected=args.refresh,
        mark_pending=args.fix,
        year_from=args.year
    )


if __name__ == '__main__':
    main()