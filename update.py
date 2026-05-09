"""
update.py — Push local database to Neon as the new live state, and upload
any pending SDM output files to the plantmandev public directory.

Usage:
  python3 update.py                # full run: upload SDM outputs + DB sync
  python3 update.py --upload-sdm   # upload SDM outputs only (skip DB sync)

DB sync logic:
  - If no save-state branches exist yet: restore dump directly to main.
  - If main has been pushed before: snapshot current main as
    save-state-YYYYMMDD, then restore the new dump to main.

SDM upload logic:
  Scans sdm-output/ for *_suitability.tif and *_range.gpkg pairs.
  For each pair:
    - Copies the .tif to plantmandev/public/species-distribution-model/
    - Converts the .gpkg to .geojson (via ogr2ogr) in the same destination
    - Upserts a row in sdm_outputs marking the species as completed
    - Deletes the local source files
"""

import argparse
import os
import sys
import json
import subprocess
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEON_PROJECT_ID = 'soft-recipe-26674693'
NEON_BRANCH     = 'main'
NEON_CONN       = os.environ.get('NEON_CONN', '')
LOCAL_CONN      = os.environ.get('LOCAL_CONN', 'postgresql://postgres@localhost:5432/lepidoptera_data')

TODAY     = date.today().strftime('%Y%m%d')
DUMP_FILE = Path(f'/tmp/lepidoptera_{TODAY}.dump')

SDM_SOURCE_DIR = Path('sdm-output')

_CREATE_SDM_TABLE = """
CREATE TABLE IF NOT EXISTS sdm_outputs (
    id               SERIAL PRIMARY KEY,
    species_id       INTEGER NOT NULL REFERENCES species(id) ON DELETE CASCADE,
    suitability_data BYTEA,
    range_data       BYTEA,
    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'completed')),
    created_at       TIMESTAMP DEFAULT NOW(),
    completed_at     TIMESTAMP,
    UNIQUE (species_id)
);
CREATE INDEX IF NOT EXISTS idx_sdm_species ON sdm_outputs (species_id);
CREATE INDEX IF NOT EXISTS idx_sdm_status  ON sdm_outputs (status);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, capture=False):
    """Run a shell command. Exits on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=capture, text=capture
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ Command failed: {e}")
        if capture and e.stderr:
            print(f"    {e.stderr.strip()}")
        sys.exit(1)


def neon_cmd(*args, capture=False):
    return run(
        ['neonctl', *args, '--output', 'json', '--project-id', NEON_PROJECT_ID],
        capture=capture
    )


def check_dependencies(upload_only=False):
    required = ['ogr2ogr']
    if not upload_only:
        required += ['pg_dump', 'pg_restore', 'neonctl']
    missing = [t for t in required if subprocess.run(['which', t], capture_output=True).returncode != 0]
    if missing:
        print(f"✗ Missing tools: {', '.join(missing)}")
        sys.exit(1)


def list_save_states():
    result = neon_cmd('branches', 'list', capture=True)
    branches = json.loads(result.stdout)
    return [b['name'] for b in branches if b['name'].startswith('save-state-')]


# ---------------------------------------------------------------------------
# DB sync steps
# ---------------------------------------------------------------------------

def create_save_state():
    branch_name = f'save-state-{TODAY}'
    print(f"\n  Creating save state: {branch_name}")
    neon_cmd('branches', 'create', '--name', branch_name, '--parent', NEON_BRANCH)
    print(f"  ✓ Save state created: {branch_name}")


def dump_local():
    run([
        'pg_dump', LOCAL_CONN,
        '--format=custom',
        '--no-owner',
        '--no-acl',
        f'--file={DUMP_FILE}',
    ])
    size_mb = DUMP_FILE.stat().st_size / (1024 * 1024)
    print(f"  ✓ Dump saved: {DUMP_FILE}  ({size_mb:.1f} MB)")


def restore_to_neon():
    run([
        'pg_restore',
        '--clean',
        '--if-exists',
        '--no-owner',
        '--no-acl',
        f'--dbname={NEON_CONN}',
        str(DUMP_FILE),
    ])
    print(f"  ✓ Neon main updated — site is now serving the new data.")


def cleanup_dump():
    if DUMP_FILE.exists():
        DUMP_FILE.unlink()
        print(f"  ✓ Removed {DUMP_FILE}")


# ---------------------------------------------------------------------------
# SDM upload step
# ---------------------------------------------------------------------------

def upload_sdm_outputs():
    """
    For each *_suitability.tif in sdm-output/:
      - Copy the .tif to plantmandev public directory
      - Convert the matching _range.gpkg to .geojson via ogr2ogr
      - Upsert sdm_outputs row in Neon (status = completed)
      - Delete the local source files
    """
    import psycopg2

    tif_files = sorted(SDM_SOURCE_DIR.glob('*_suitability.tif'))
    if not tif_files:
        print("  No pending SDM output files — skipping.")
        return

    conn = psycopg2.connect(NEON_CONN)
    try:
        with conn.cursor() as cur:
            cur.execute(_CREATE_SDM_TABLE)
        conn.commit()
    except Exception as e:
        conn.close()
        print(f"  ✗ Could not ensure sdm_outputs table: {e}")
        sys.exit(1)

    uploaded = failed = 0

    try:
        for tif_path in tif_files:
            slug         = tif_path.stem.replace('_suitability', '')
            species_name = slug.replace('_', ' ').replace('-', ' ')
            gpkg_path    = SDM_SOURCE_DIR / f"{slug}_range.gpkg"

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM species WHERE LOWER(scientific_name) = LOWER(%s)",
                    (species_name,)
                )
                row = cur.fetchone()

            if not row:
                print(f"  ✗ {species_name}: not found in species table — skipping")
                failed += 1
                continue

            species_id = row[0]

            try:
                suit_bytes = tif_path.read_bytes()

                range_bytes = None
                if gpkg_path.exists():
                    result = subprocess.run(
                        ['ogr2ogr', '-f', 'GeoJSON', '/vsistdout/', str(gpkg_path)],
                        check=True, capture_output=True
                    )
                    range_bytes = result.stdout

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

                tif_path.unlink()
                if gpkg_path.exists():
                    gpkg_path.unlink()

                mb = len(suit_bytes) / 1_048_576
                print(f"  ✓ {species_name}  ({mb:.1f} MB)")
                uploaded += 1

            except Exception as e:
                conn.rollback()
                print(f"  ✗ {species_name}: {e}")
                failed += 1

    finally:
        conn.close()

    print(f"\n  uploaded: {uploaded}  |  failed: {failed}")
    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--upload-sdm', action='store_true',
        help='Upload SDM outputs only; skip the database dump/restore cycle.'
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"UPDATE — {TODAY}")
    print(f"{'='*60}\n")

    if not NEON_CONN:
        print("✗ NEON_CONN environment variable not set.")
        print("  export NEON_CONN='postgresql://user:pass@host/db?sslmode=require'")
        sys.exit(1)

    check_dependencies(upload_only=args.upload_sdm)

    # ── SDM upload ────────────────────────────────────────────────────────────
    pending_tifs = list(SDM_SOURCE_DIR.glob('*_suitability.tif')) if SDM_SOURCE_DIR.exists() else []

    if pending_tifs:
        print(f"\n{'='*60}")
        print(f"STEP {'1' if args.upload_sdm else '0'} — Uploading SDM outputs ({len(pending_tifs)} species)")
        print(f"{'='*60}\n")
        upload_sdm_outputs()
    else:
        print("  No SDM outputs pending.")

    if args.upload_sdm:
        print(f"\n{'='*60}")
        print("✓ SDM upload complete.")
        print(f"{'='*60}\n")
        sys.exit(0)

    # ── DB sync ───────────────────────────────────────────────────────────────
    save_states = list_save_states()

    if save_states:
        print(f"  Existing save states: {', '.join(save_states)}")
        print(f"\n{'='*60}")
        print("STEP 1 — Snapshotting current main")
        print(f"{'='*60}\n")
        create_save_state()
    else:
        print("  No previous save states found — first push, skipping snapshot.\n")

    print(f"\n{'='*60}")
    print("STEP 2 — Dumping local database")
    print(f"{'='*60}\n")
    dump_local()

    print(f"\n{'='*60}")
    print("STEP 3 — Restoring to Neon main")
    print(f"{'='*60}\n")
    restore_to_neon()

    print(f"\n{'='*60}")
    print("STEP 4 — Cleanup")
    print(f"{'='*60}\n")
    cleanup_dump()

    print(f"\n{'='*60}")
    print("✓ Update complete.")
    print(f"{'='*60}\n")
