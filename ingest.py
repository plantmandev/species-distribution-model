"""
ingest.py — Load occurrence GeoJSON files into the PostGIS database.

Reads species-metadata.csv, routes each species to the correct table
based on the 'category' column, links host plants, then deletes the
GeoJSON file and marks the row as 'ingested'.

Prerequisites:
  - setup_database.sql has been run
  - species-metadata.csv has 'category' and 'host_plants' columns
  - GeoJSON files exist in occurrence-data/

Usage:
  export DB_APP_PASSWORD="yourpassword"
  python ingest.py                        # ingest all pending species
  python ingest.py --dry-run              # preview without writing
  python ingest.py --species "Morpho"     # ingest one species only
  python ingest.py --no-delete            # ingest but keep GeoJSON files
"""

import os
import json
import argparse
import time
from pathlib import Path

import pandas as pd
import geopandas as gpd
import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR      = Path('occurrence-data')
METADATA_FILE = DATA_DIR / 'species-metadata.csv'

DB_CONFIG = {
    'host':     os.environ.get('DB_HOST',     'localhost'),
    'port':     int(os.environ.get('DB_PORT', 5432)),
    'dbname':   os.environ.get('DB_NAME',     'lepidoptera_data'),
    'user':     os.environ.get('DB_USER',     'lepidoptera_app'),
    'password': os.environ['DB_APP_PASSWORD'],
}

BATCH_SIZE = 500


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


# ---------------------------------------------------------------------------
# Species table
# ---------------------------------------------------------------------------

def upsert_species(cur, row):
    """
    Insert species row if not exists. Returns the species id.
    Updates common_name and family if already present.
    """
    gbif_key = int(row['gbif_key']) if pd.notna(row.get('gbif_key')) and str(row.get('gbif_key', '')).strip() not in ('', 'nan') else None

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
    return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Lepidoptera occurrences
# ---------------------------------------------------------------------------

def ingest_butterfly(conn, cur, species_id, gdf, dry_run=False):
    """
    Insert lepidoptera occurrences from a GeoDataFrame.
    Returns count of rows inserted.
    """
    rows = []
    for _, feat in gdf.iterrows():
        lat = feat.geometry.y
        lng = feat.geometry.x

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
            f'geojson-{species_id}-{len(rows)}',   # synthetic source_id for GeoJSON-origin records
            1,                                      # tier 1
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
    Insert host plant occurrences from a GeoDataFrame.
    Returns count of rows inserted.
    """
    rows = []
    for _, feat in gdf.iterrows():
        lat = feat.geometry.y
        lng = feat.geometry.x

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
            f'geojson-{scientific_name.replace(" ", "-")}-{len(rows)}',
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
    """Insert rows into species_host_plants for each host plant name."""
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
    Ingest a single species row. Handles both butterfly and host_plant categories.
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

            # ----------------------------------------------------------------
            # BUTTERFLY
            # ----------------------------------------------------------------
            if category == 'butterfly':
                species_id = upsert_species(cur, row)
                conn.commit()

                inserted = ingest_butterfly(conn, cur, species_id, gdf, dry_run)
                print(f"  ✓ {species_name:40s} {inserted:>7,} occurrences → lepidoptera_occurrences")

                # Link host plants
                host_plant_names = parse_host_plants(row.get('host_plants'))
                if host_plant_names:
                    link_host_plants(cur, species_id, host_plant_names, dry_run)
                    conn.commit()

                    # Ingest host plant GeoJSONs that exist
                    for plant_name in host_plant_names:
                        plant_path = get_geojson_path(plant_name)
                        if plant_path.exists():
                            try:
                                plant_gdf     = gpd.read_file(plant_path)
                                plant_inserted = ingest_host_plant(conn, cur, plant_name, plant_gdf, dry_run)
                                print(f"    ↳ {plant_name:38s} {plant_inserted:>7,} occurrences → host_plant_occurrences")
                                if delete_after and not dry_run:
                                    plant_path.unlink()
                                    _mark_ingested(df, plant_name)
                            except Exception as e:
                                print(f"    ✗ {plant_name}: {e}")
                        else:
                            print(f"    ⚠ {plant_name:38s} no GeoJSON yet (run procure.py to download)")

                refresh_extents(cur, species_id, dry_run)
                conn.commit()

            # ----------------------------------------------------------------
            # HOST PLANT (standalone — not triggered from a butterfly row)
            # ----------------------------------------------------------------
            elif category == 'host_plant':
                inserted = ingest_host_plant(conn, cur, species_name, gdf, dry_run)
                conn.commit()
                print(f"  ✓ {species_name:40s} {inserted:>7,} occurrences → host_plant_occurrences")

            else:
                print(f"  ⚠ {species_name:40s} unknown category '{category}' — skipping")
                return False

        # Success — delete file and mark metadata
        if delete_after and not dry_run:
            geojson_path.unlink()
            _mark_ingested(df, species_name)

        return True

    except Exception as e:
        conn.rollback()
        print(f"  ✗ {species_name:40s} {e}")
        return False


def _mark_ingested(df, species_name):
    """Update status to 'ingested' in the in-memory DataFrame."""
    mask = df['species_name'] == species_name
    if mask.any():
        df.loc[mask, 'status'] = 'ingested'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ingest(target_species=None, dry_run=False, delete_after=True):
    if not METADATA_FILE.exists():
        print("Error: species-metadata.csv not found.")
        return

    df = read_csv_meta(METADATA_FILE)

    # Validate category column exists
    if 'category' not in df.columns:
        print("Error: metadata CSV missing 'category' column.")
        print("Add a 'category' column with values: butterfly | host_plant")
        return

    # Filter to target species if specified
    if target_species:
        mask = df['species_name'].str.lower() == target_species.lower()
        if not mask.any():
            print(f"Error: '{target_species}' not found in metadata.")
            return
        rows_to_process = df[mask]
    else:
        # Skip already ingested and species with no GeoJSON
        rows_to_process = df[df['status'] != 'ingested']

    if len(rows_to_process) == 0:
        print("\n✓ Nothing to ingest — all species already marked as ingested.")
        print("  Use --species NAME to re-ingest a specific species.")
        return

    print(f"\n{'='*60}")
    print(f"INGESTING {len(rows_to_process)} SPECIES")
    if dry_run:
        print("DRY RUN — no data will be written or deleted")
    print(f"{'='*60}\n")

    conn = connect()
    success = 0
    skipped = 0

    try:
        for idx, row in rows_to_process.iterrows():
            category = str(row.get('category', '')).strip().lower()

            # Skip standalone host_plant rows — they get ingested when their
            # butterfly is processed. Only process them standalone if they have
            # no butterfly association in the CSV.
            if category == 'host_plant':
                # Check if any butterfly row references this plant
                if 'host_plants' in df.columns:
                    referenced = df['host_plants'].dropna().apply(
                        lambda x: row['species_name'] in [s.strip() for s in str(x).split(',')]
                    ).any()
                    if referenced:
                        skipped += 1
                        continue   # will be handled when the butterfly is ingested

            result = ingest_species(row, df, idx, conn, dry_run, delete_after)
            if result:
                success += 1

    except KeyboardInterrupt:
        print("\n⏸ Interrupted")
    finally:
        conn.close()

    # Write updated metadata (status = 'ingested' for completed rows)
    if not dry_run:
        write_csv_meta(df, METADATA_FILE)

    print(f"\n{'='*60}")
    print(f"✓ {success} ingested | {skipped} skipped (handled via butterfly) | {len(rows_to_process) - success - skipped} failed")
    print(f"{'='*60}\n")


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
  python ingest.py                        # ingest all pending species
  python ingest.py --dry-run              # preview without writing
  python ingest.py --species "Morpho"     # ingest one species
  python ingest.py --no-delete            # ingest but keep GeoJSON files
        """
    )
    parser.add_argument('--dry-run',   action='store_true', help='Preview without writing to DB or deleting files')
    parser.add_argument('--no-delete', action='store_true', help='Ingest but keep GeoJSON files on disk')
    parser.add_argument('--species',   type=str, default=None, help='Ingest a single species by name')
    args = parser.parse_args()

    if 'DB_APP_PASSWORD' not in os.environ:
        print("Error: DB_APP_PASSWORD environment variable not set.")
        print("  export DB_APP_PASSWORD='yourpassword'")
        return

    run_ingest(
        target_species=args.species,
        dry_run=args.dry_run,
        delete_after=not args.no_delete,
    )


if __name__ == '__main__':
    main()