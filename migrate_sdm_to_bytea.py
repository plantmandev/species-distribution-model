"""
migrate_sdm_to_bytea.py — One-time migration.

Reads the .tif and .geojson files already in plantmandev/public/species-distribution-model/
and uploads them as raw bytes into the sdm_outputs table in Neon.
Runs against the direct (non-pooler) Neon endpoint to avoid PgBouncer
message-size limits when inserting multi-MB bytea values.

Usage:
    python3 migrate_sdm_to_bytea.py
    python3 migrate_sdm_to_bytea.py --dry-run
"""

import os
import sys
import argparse
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

NEON_CONN    = os.environ['NEON_CONN']
PUBLIC_DIR   = Path('/home/denhac/Projects/plantmandev/public/species-distribution-model')

# Neon pooler URLs contain "-pooler." — strip it for the direct endpoint,
# which handles large binary payloads without PgBouncer in the way.
DIRECT_CONN  = (
    NEON_CONN
    .replace('-pooler.', '.')
    .replace('channel_binding=require&', '')
    .replace('&channel_binding=require', '')
    .replace('channel_binding=require', '')
)


def migrate(dry_run: bool) -> None:
    tif_files = sorted(PUBLIC_DIR.glob('*_suitability.tif'))
    if not tif_files:
        print("No .tif files found in", PUBLIC_DIR)
        return

    print(f"Found {len(tif_files)} suitability TIFs  (dry_run={dry_run})\n")

    if dry_run:
        for p in tif_files[:5]:
            print(f"  {p.name}  ({p.stat().st_size / 1_048_576:.1f} MB)")
        if len(tif_files) > 5:
            print(f"  ... and {len(tif_files) - 5} more")
        return

    conn = psycopg2.connect(DIRECT_CONN)
    uploaded = skipped = failed = 0

    try:
        for tif_path in tif_files:
            slug         = tif_path.stem.replace('_suitability', '')
            species_name = slug.replace('_', ' ').replace('-', ' ')
            geojson_path = PUBLIC_DIR / f"{slug}_range.geojson"

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM species WHERE LOWER(scientific_name) = LOWER(%s)",
                    (species_name,)
                )
                row = cur.fetchone()

            if not row:
                print(f"  ✗ {species_name}: not in species table")
                skipped += 1
                continue

            species_id = row[0]

            # Skip if bytea data already present
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT suitability_data IS NOT NULL FROM sdm_outputs WHERE species_id = %s",
                    (species_id,)
                )
                existing = cur.fetchone()

            if existing and existing[0]:
                skipped += 1
                continue

            try:
                suit_bytes  = tif_path.read_bytes()
                range_bytes = geojson_path.read_bytes() if geojson_path.exists() else None

                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO sdm_outputs
                            (species_id, suitability_data, range_data, status, completed_at)
                        VALUES (%s, %s, %s, 'completed', NOW())
                        ON CONFLICT (species_id) DO UPDATE SET
                            suitability_data = EXCLUDED.suitability_data,
                            range_data       = EXCLUDED.range_data,
                            status           = 'completed',
                            completed_at     = NOW()
                    """, (species_id, suit_bytes, range_bytes))
                conn.commit()

                mb = len(suit_bytes) / 1_048_576
                print(f"  ✓ {species_name}  ({mb:.1f} MB)")
                uploaded += 1

            except Exception as e:
                conn.rollback()
                print(f"  ✗ {species_name}: {e}")
                failed += 1

    finally:
        conn.close()

    print(f"\nuploaded: {uploaded}  |  skipped: {skipped}  |  failed: {failed}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    migrate(args.dry_run)
