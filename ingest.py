"""
ingest.py — Load occurrence GeoJSON files into the PostGIS database.

Reads species-metadata.csv, routes each species to the correct table
based on the 'category' column, links host plants, then deletes the
GeoJSON file and marks the row as 'ingested'.

If a row's 'notes' column contains '#drop', the species is deleted from
the database and removed from the metadata CSV entirely.

Duplicates are prevented using stable GBIF occurrence IDs as source_id.

Prerequisites:
  - setup_database.sql has been run
  - species-metadata.csv has 'category' and 'host_plants' columns
  - category values: lepidoptera | host_plant
  - GeoJSON files exist in occurrence-data/

Usage:
  export DB_APP_PASSWORD="yourpassword"
  python ingest.py                                  # ingest all pending species
  python ingest.py --dry-run                        # preview without writing
  python ingest.py --species "Morpho"               # ingest one species only
  python ingest.py --no-delete                      # ingest but keep GeoJSON files
  python ingest.py --populate-hosts resource.csv    # populate host plants then ingest
  python ingest.py --populate-hosts resource.csv --hosts-only  # populate only
"""

import os
import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd
import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR           = Path('occurrence-data')
METADATA_FILE      = DATA_DIR / 'species-metadata.csv'
HOSTS_CLEANED_FILE = DATA_DIR / 'hosts-cleaned.csv'
HOSTS_RAW_FILE     = DATA_DIR / 'host-plant-info.csv'

DB_CONFIG = {
    'host':     os.environ.get('DB_HOST',     'localhost'),
    'port':     int(os.environ.get('DB_PORT', 5432)),
    'dbname':   os.environ.get('DB_NAME',     'lepidoptera_data'),
    'user':     os.environ.get('DB_USER',     'lepidoptera_app'),
    'password': '',   # set in main() from DB_APP_PASSWORD env var
}

BATCH_SIZE = 500

FAMILY_NORMALIZE = {
    'Leguminosae (C)':  'Fabaceae',
    'Leguminosae (M)':  'Fabaceae',
    'Leguminosae (P)':  'Fabaceae',
    'Leguminosae':      'Fabaceae',
    'Compositae':       'Asteraceae',
    'Cruciferae':       'Brassicaceae',
    'Gramineae':        'Poaceae',
    'Labiatae':         'Lamiaceae',
    'Umbelliferae':     'Apiaceae',
    'Guttiferae':       'Hypericaceae',
    'Palmae':           'Arecaceae',
    'Chenopodiaceae':   'Amaranthaceae',
    'Scrophulariaceae': 'Plantaginaceae',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv_meta(path):
    return pd.read_csv(path, dtype={'gbif_key': str})


def write_csv_meta(df, path):
    df.to_csv(path, index=False)


def get_geojson_path(species_name):
    safe = species_name.replace(' ', '-').lower()
    return DATA_DIR / f'{safe}-gbif.geojson'


def parse_host_plants(value):
    """Parse comma-separated host plant names, return list of stripped strings."""
    if pd.isna(value) or str(value).strip() == '':
        return []
    return [s.strip() for s in str(value).split(',') if s.strip()]


def connect():
    return psycopg2.connect(**DB_CONFIG)


def has_drop_note(row):
    """Return True if the row's notes field contains '#drop'."""
    notes = str(row.get('notes', '')).strip()
    return '#drop' in notes.lower()


# ---------------------------------------------------------------------------
# Drop species from database and metadata
# ---------------------------------------------------------------------------

def drop_flagged_species(df, conn, dry_run=False):
    """
    For any row with '#drop' in notes:
      - Delete all occurrences from the database
      - Delete the species row (lepidoptera only)
      - Delete the GeoJSON file if it still exists
      - Remove the row from the metadata DataFrame

    Returns the cleaned DataFrame.
    """
    flagged = [(idx, row) for idx, row in df.iterrows() if has_drop_note(row)]

    if not flagged:
        return df

    print(f"\n{'='*60}")
    print(f"DROPPING {len(flagged)} FLAGGED SPECIES (#drop)")
    print(f"{'='*60}\n")

    indices_to_drop = []

    for idx, row in flagged:
        species_name = row['species_name']
        category     = str(row.get('category', '')).strip().lower()

        print(f"  ✗ {species_name}")

        if not dry_run:
            try:
                with conn.cursor() as cur:
                    if category == 'lepidoptera':
                        # Delete occurrences via species_id foreign key
                        cur.execute("""
                            DELETE FROM lepidoptera_occurrences
                            WHERE species_id = (
                                SELECT id FROM species WHERE scientific_name = %s
                            );
                        """, (species_name,))
                        cur.execute("""
                            DELETE FROM species_host_plants
                            WHERE species_id = (
                                SELECT id FROM species WHERE scientific_name = %s
                            );
                        """, (species_name,))
                        cur.execute(
                            "DELETE FROM species WHERE scientific_name = %s;",
                            (species_name,)
                        )
                    elif category == 'host_plant':
                        cur.execute(
                            "DELETE FROM host_plant_occurrences WHERE scientific_name = %s;",
                            (species_name,)
                        )
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"    ⚠ DB error for {species_name}: {e}")
                continue

        # Delete GeoJSON if still on disk
        geojson_path = get_geojson_path(species_name)
        if geojson_path.exists() and not dry_run:
            geojson_path.unlink()
            print(f"    deleted {geojson_path.name}")

        indices_to_drop.append(idx)

    if not dry_run:
        df = df.drop(indices_to_drop).reset_index(drop=True)
        print(f"\n  ✓ Removed {len(indices_to_drop)} rows from metadata.")
    else:
        print(f"\n  DRY RUN — {len(indices_to_drop)} rows would be removed.")

    print()
    return df


# ---------------------------------------------------------------------------
# NHM HOSTS database — clean and populate metadata
# ---------------------------------------------------------------------------

def clean_hosts_db(hosts_path):
    """
    Load, clean, and save the NHM HOSTS CSV.
    """
    print(f"\n{'='*60}")
    print("CLEANING NHM HOSTS DATABASE")
    print(f"{'='*60}\n")

    df = pd.read_csv(hosts_path)
    print(f"  Raw records: {len(df):,}")

    drop_cols = ['_id', 'HOSTS ID', 'Damage', 'Hostplant Subspecies/var',
                 'Insect Author', 'Insect Subspecies']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if 'Lab Rearing' in df.columns:
        before = len(df)
        df = df[df['Lab Rearing'].isna() | (df['Lab Rearing'] == '?')]
        df = df.drop(columns=['Lab Rearing'])
        print(f"  Dropped {before - len(df):,} lab-rearing-only records")

    df['Hostplant Family'] = df['Hostplant Family'].map(
        lambda x: FAMILY_NORMALIZE.get(str(x).strip(), str(x).strip())
        if pd.notna(x) else x
    )

    before = len(df)
    df = df[df['Hostplant Genus'].notna() & (df['Hostplant Genus'].str.strip() != '')]
    print(f"  Dropped {before - len(df):,} records with no hostplant genus")

    before = len(df)
    df = df[~df['Hostplant Family'].str.lower().isin(['polyphagous', 'detritophagous'])]
    print(f"  Dropped {before - len(df):,} polyphagous/detritophagous records")

    df['host_plant_name'] = (
        df['Hostplant Genus'].fillna('').str.strip() + ' ' +
        df['Hostplant Species'].fillna('').str.strip()
    ).str.strip()

    df['insect_name'] = (
        df['Insect Genus'].fillna('').str.strip() + ' ' +
        df['Insect Species'].fillna('').str.strip()
    ).str.strip()

    df = df.rename(columns={
        'Insect Family':     'insect_family',
        'Insect Genus':      'insect_genus',
        'Insect Species':    'insect_species',
        'Hostplant Family':  'plant_family',
        'Hostplant Genus':   'plant_genus',
        'Hostplant Species': 'plant_species',
        'Location':          'location',
    })

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(HOSTS_CLEANED_FILE, index=False)

    print(f"\n  Cleaned records:       {len(df):,}")
    print(f"  Unique insect species: {df['insect_name'].nunique():,}")
    print(f"  Unique host plants:    {df['host_plant_name'].nunique():,}")
    print(f"  Unique plant families: {df['plant_family'].nunique():,}")
    print(f"\n  ✓ Saved cleaned CSV → {HOSTS_CLEANED_FILE}")

    return df


def build_hosts_lookup(hosts_df):
    lookup = {}

    for genus, group in hosts_df.groupby('insect_genus'):
        plants   = sorted(set(p for p in group['host_plant_name'].dropna() if p))
        families = sorted(set(f for f in group['plant_family'].dropna() if f))
        locs     = sorted(set(l for l in group['location'].dropna() if l))
        lookup[genus.strip()] = {
            'host_plants':         ', '.join(plants),
            'host_plant_families': ', '.join(families),
            'location':            ', '.join(locs),
        }

    for name, group in hosts_df.groupby('insect_name'):
        name = name.strip()
        if not name or ' ' not in name:
            continue
        plants   = sorted(set(p for p in group['host_plant_name'].dropna() if p))
        families = sorted(set(f for f in group['plant_family'].dropna() if f))
        locs     = sorted(set(l for l in group['location'].dropna() if l))
        lookup[name] = {
            'host_plants':         ', '.join(plants),
            'host_plant_families': ', '.join(families),
            'location':            ', '.join(locs),
        }

    return lookup


def populate_host_plants(hosts_path, dry_run=False):
    hosts_df = clean_hosts_db(hosts_path)
    lookup   = build_hosts_lookup(hosts_df)

    print(f"\n{'='*60}")
    print("POPULATING METADATA HOST PLANTS")
    print(f"{'='*60}\n")

    meta = read_csv_meta(METADATA_FILE)

    for col in ['host_plants', 'host_plant_families', 'host_plant_status']:
        if col not in meta.columns:
            meta[col] = ''

    found   = 0
    missing = 0

    for idx, row in meta.iterrows():
        species_name = row['species_name'].strip()
        category     = str(row.get('category', '')).strip().lower()

        if category != 'lepidoptera':
            continue

        existing = str(row.get('host_plants', '')).strip()
        if existing and existing not in ('', 'nan'):
            print(f"  → {species_name:40s} already populated, skipping")
            continue

        data = lookup.get(species_name) or lookup.get(species_name.split()[0])

        if data and data['host_plants']:
            count = len(data['host_plants'].split(','))
            print(f"  ✓ {species_name:40s} {count} plants | {data['host_plant_families'][:50]}")
            if not dry_run:
                meta.at[idx, 'host_plants']         = data['host_plants']
                meta.at[idx, 'host_plant_families'] = data['host_plant_families']
                meta.at[idx, 'host_plant_status']   = 'complete'
            found += 1
        else:
            print(f"  ✗ {species_name:40s} no data — flagged as missing")
            if not dry_run:
                meta.at[idx, 'host_plant_status'] = 'missing'
            missing += 1

    if not dry_run:
        write_csv_meta(meta, METADATA_FILE)

    print(f"\n{'='*60}")
    print(f"  ✓ Populated : {found}")
    print(f"  ✗ Missing   : {missing}")
    if dry_run:
        print("  DRY RUN — no changes written")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Species table
# ---------------------------------------------------------------------------

def upsert_species(cur, row):
    """Insert or update species row. Returns the species id."""
    gbif_key = (
        int(row['gbif_key'])
        if pd.notna(row.get('gbif_key')) and str(row.get('gbif_key', '')).strip() not in ('', 'nan')
        else None
    )

    cur.execute("""
        INSERT INTO species (scientific_name, common_name, family, gbif_taxon_key)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (scientific_name)
        DO UPDATE SET
            common_name    = EXCLUDED.common_name,
            family         = EXCLUDED.family,
            gbif_taxon_key = COALESCE(EXCLUDED.gbif_taxon_key, species.gbif_taxon_key)
        RETURNING id;
    """, (
        row['species_name'],
        row.get('common_name') if pd.notna(row.get('common_name')) else None,
        row.get('family')      if pd.notna(row.get('family'))      else None,
        gbif_key,
    ))
    result = cur.fetchone()
    if result is None:
        raise RuntimeError(f"species upsert returned no id for '{row['species_name']}'")
    return result[0]


# ---------------------------------------------------------------------------
# Lepidoptera occurrences
# ---------------------------------------------------------------------------

def ingest_lepidoptera(conn, cur, species_id, gdf, dry_run=False):
    """
    Insert lepidoptera occurrences. Uses gbifID as stable source_id to
    prevent duplicate rows on re-ingest.
    """
    rows = []
    for _, feat in gdf.iterrows():
        lat = feat.geometry.y
        lng = feat.geometry.x

        # Use GBIF's own occurrence ID as the stable dedup key
        gbif_id = feat.get('gbifID') or feat.get('occurrenceID') or feat.get('id')
        if gbif_id is None:
            # Last resort: hash of coordinates + species_id (still better than row index)
            gbif_id = f'{species_id}-{lat:.6f}-{lng:.6f}'
        source_id = f'gbif-{gbif_id}'

        observed = feat.get('eventDate')
        if pd.notna(observed):
            try:
                observed = pd.to_datetime(observed, utc=True).date().isoformat()
            except Exception:
                observed = None
        else:
            observed = None

        rows.append((
            species_id,
            observed,
            'GBIF',
            source_id,
            1,
            f'SRID=4326;POINT({lng} {lat})',
        ))

    if dry_run:
        return len(rows)

    inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        psycopg2.extras.execute_values(cur, """
            INSERT INTO lepidoptera_occurrences
                (species_id, observed_date, source, source_id, data_tier, geom)
            VALUES %s
            ON CONFLICT (source, source_id) DO NOTHING
        """, batch, template="(%s, %s, %s, %s, %s, ST_GeomFromEWKT(%s))")
        inserted += cur.rowcount
        conn.commit()

    return inserted


# ---------------------------------------------------------------------------
# Host plant occurrences
# ---------------------------------------------------------------------------

def ingest_host_plant(conn, cur, scientific_name, gdf, dry_run=False):
    """
    Insert host plant occurrences. Uses gbifID as stable source_id to
    prevent duplicate rows on re-ingest.
    """
    rows = []
    safe_name = scientific_name.replace(' ', '-').lower()

    for _, feat in gdf.iterrows():
        lat = feat.geometry.y
        lng = feat.geometry.x

        gbif_id = feat.get('gbifID') or feat.get('occurrenceID') or feat.get('id')
        if gbif_id is None:
            gbif_id = f'{safe_name}-{lat:.6f}-{lng:.6f}'
        source_id = f'gbif-{gbif_id}'

        observed = feat.get('eventDate')
        if pd.notna(observed):
            try:
                observed = pd.to_datetime(observed, utc=True).date().isoformat()
            except Exception:
                observed = None
        else:
            observed = None

        rows.append((
            scientific_name,
            observed,
            'GBIF',
            source_id,
            f'SRID=4326;POINT({lng} {lat})',
        ))

    if dry_run:
        return len(rows)

    inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        psycopg2.extras.execute_values(cur, """
            INSERT INTO host_plant_occurrences
                (scientific_name, observed_date, source, source_id, geom)
            VALUES %s
            ON CONFLICT (source, source_id) DO NOTHING
        """, batch, template="(%s, %s, %s, %s, ST_GeomFromEWKT(%s))")
        inserted += cur.rowcount
        conn.commit()

    return inserted


# ---------------------------------------------------------------------------
# Host plant linkage
# ---------------------------------------------------------------------------

def link_host_plants(cur, species_id, host_plant_names, dry_run=False):
    if dry_run:
        return
    for plant_name in host_plant_names:
        cur.execute("""
            INSERT INTO species_host_plants (species_id, plant_scientific_name)
            VALUES (%s, %s)
            ON CONFLICT (species_id, plant_scientific_name) DO NOTHING;
        """, (species_id, plant_name))


# ---------------------------------------------------------------------------
# Refresh extents
# ---------------------------------------------------------------------------

def refresh_extents(cur, species_id, dry_run=False):
    if dry_run:
        return
    cur.execute("SELECT refresh_species_extents(%s);", (species_id,))


# ---------------------------------------------------------------------------
# Core ingest logic
# ---------------------------------------------------------------------------

def ingest_species(row, df, idx, conn, dry_run=False, delete_after=True):
    """
    Ingest a single species row. Handles lepidoptera and host_plant categories.
    Returns True on success.
    """
    species_name = row['species_name']
    category     = str(row.get('category', '')).strip().lower()
    geojson_path = get_geojson_path(species_name)

    if not geojson_path.exists():
        print(f"  ⚠ {species_name:40s} GeoJSON not found — skipping")
        return False

    try:
        gdf = gpd.read_file(geojson_path)
        if len(gdf) == 0:
            print(f"  ⚠ {species_name:40s} GeoJSON is empty — skipping")
            return False
    except Exception as e:
        print(f"  ✗ {species_name:40s} Could not read GeoJSON: {e}")
        return False

    try:
        with conn.cursor() as cur:

            if category == 'lepidoptera':
                species_id = upsert_species(cur, row)
                conn.commit()

                inserted = ingest_lepidoptera(conn, cur, species_id, gdf, dry_run)
                print(f"  ✓ {species_name:40s} {inserted:>7,} occurrences → lepidoptera_occurrences")

                host_plant_names = parse_host_plants(row.get('host_plants'))
                if host_plant_names:
                    link_host_plants(cur, species_id, host_plant_names, dry_run)
                    conn.commit()

                    for plant_name in host_plant_names:
                        plant_path = get_geojson_path(plant_name)
                        if plant_path.exists():
                            try:
                                plant_gdf      = gpd.read_file(plant_path)
                                plant_inserted = ingest_host_plant(conn, cur, plant_name, plant_gdf, dry_run)
                                print(f"    ↳ {plant_name:38s} {plant_inserted:>7,} occurrences → host_plant_occurrences")
                                if delete_after and not dry_run:
                                    plant_path.unlink()
                                    _mark_ingested(df, plant_name)
                            except Exception as e:
                                conn.rollback()
                                print(f"    ✗ {plant_name}: {e}")
                        else:
                            print(f"    ⚠ {plant_name:38s} no GeoJSON yet (run procure.py)")

                refresh_extents(cur, species_id, dry_run)
                conn.commit()

            elif category == 'host_plant':
                inserted = ingest_host_plant(conn, cur, species_name, gdf, dry_run)
                conn.commit()
                print(f"  ✓ {species_name:40s} {inserted:>7,} occurrences → host_plant_occurrences")

            else:
                print(f"  ⚠ {species_name:40s} unknown category '{category}' — skipping")
                return False

        if delete_after and not dry_run:
            geojson_path.unlink()
            _mark_ingested(df, species_name)

        return True

    except Exception as e:
        conn.rollback()
        print(f"  ✗ {species_name:40s} {e}")
        return False


def _mark_ingested(df, species_name):
    mask = df['species_name'] == species_name
    if mask.any():
        df.loc[mask, 'status'] = 'ingested'


# ---------------------------------------------------------------------------
# Main ingest runner
# ---------------------------------------------------------------------------

def run_ingest(target_species=None, dry_run=False, delete_after=True):
    if not METADATA_FILE.exists():
        print("Error: species-metadata.csv not found.")
        return

    df = read_csv_meta(METADATA_FILE)

    if 'category' not in df.columns:
        print("Error: metadata CSV missing 'category' column.")
        return

    conn = connect()

    # Handle #drop flags before ingesting anything
    df = drop_flagged_species(df, conn, dry_run=dry_run)
    if not dry_run:
        write_csv_meta(df, METADATA_FILE)

    if target_species:
        mask = df['species_name'].str.lower() == target_species.lower()
        if not mask.any():
            print(f"Error: '{target_species}' not found in metadata.")
            conn.close()
            return
        rows_to_process = df[mask]
    else:
        rows_to_process = df[df['status'] != 'ingested']

    if len(rows_to_process) == 0:
        print("\n✓ Nothing to ingest — all species already ingested.")
        conn.close()
        return

    print(f"\n{'='*60}")
    print(f"INGESTING {len(rows_to_process)} SPECIES")
    if dry_run:
        print("DRY RUN — no data will be written or deleted")
    print(f"{'='*60}\n")

    success = 0
    skipped = 0

    try:
        for idx, row in rows_to_process.iterrows():
            category = str(row.get('category', '')).strip().lower()

            if category == 'host_plant' and 'host_plants' in df.columns:
                referenced = df['host_plants'].dropna().apply(
                    lambda x: row['species_name'] in [s.strip() for s in str(x).split(',')]
                ).any()
                if referenced:
                    skipped += 1
                    continue

            result = ingest_species(row, df, idx, conn, dry_run, delete_after)
            if result:
                success += 1

    except KeyboardInterrupt:
        print("\n⏸ Interrupted")
    finally:
        conn.close()

    if not dry_run:
        write_csv_meta(df, METADATA_FILE)

    print(f"\n{'='*60}")
    print(f"✓ {success} ingested | {skipped} skipped (handled via lepidoptera) | {len(rows_to_process) - success - skipped} failed")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Ingest GeoJSON occurrence files into PostGIS database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  DB_APP_PASSWORD  (required)
  DB_HOST          (default: localhost)
  DB_PORT          (default: 5432)
  DB_NAME          (default: lepidoptera_data)
  DB_USER          (default: lepidoptera_app)

Examples:
  python ingest.py                                  # ingest all pending + process #drop flags
  python ingest.py --dry-run                        # preview without writing
  python ingest.py --species "Morpho"               # ingest one species
  python ingest.py --no-delete                      # ingest but keep GeoJSON files
  python ingest.py --populate-hosts resource.csv --hosts-only
  python ingest.py --populate-hosts resource.csv
        """
    )
    parser.add_argument('--dry-run',    action='store_true')
    parser.add_argument('--no-delete',  action='store_true')
    parser.add_argument('--species',    type=str, default=None)
    parser.add_argument('--populate-hosts', type=Path, nargs='?',
                        const=HOSTS_RAW_FILE, default=None, metavar='HOSTS_CSV')
    parser.add_argument('--hosts-only', action='store_true')
    args = parser.parse_args()

    if args.populate_hosts:
        if not args.populate_hosts.exists():
            print(f"Error: HOSTS CSV not found at {args.populate_hosts}")
            return
        populate_host_plants(args.populate_hosts, dry_run=args.dry_run)
        if args.hosts_only:
            return

    if not args.hosts_only:
        if 'DB_APP_PASSWORD' not in os.environ:
            print("Error: DB_APP_PASSWORD environment variable not set.")
            print("  export DB_APP_PASSWORD='yourpassword'")
            return

        DB_CONFIG['password'] = os.environ['DB_APP_PASSWORD']

        run_ingest(
            target_species=args.species,
            dry_run=args.dry_run,
            delete_after=not args.no_delete,
        )


if __name__ == '__main__':
    main()