"""
update.py — Push local database to Neon as the new live state.

Logic:
  - If no save-state branches exist yet: restore dump directly to main.
  - If main has been pushed before: snapshot current main as
    save-state-YYYYMMDD, then restore the new dump to main.

Usage:
  python3 update.py
"""

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


def check_dependencies():
    missing = []
    for tool in ['pg_dump', 'pg_restore', 'neonctl']:
        if subprocess.run(['which', tool], capture_output=True).returncode != 0:
            missing.append(tool)
    if missing:
        print(f"✗ Missing tools: {', '.join(missing)}")
        sys.exit(1)


def list_save_states():
    """Return list of existing save-state branch names."""
    result = neon_cmd('branches', 'list', capture=True)
    branches = json.loads(result.stdout)
    return [b['name'] for b in branches if b['name'].startswith('save-state-')]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def create_save_state():
    """Snapshot current main as a dated save-state branch."""
    branch_name = f'save-state-{TODAY}'
    print(f"\n  Creating save state: {branch_name}")
    neon_cmd('branches', 'create',
             '--name', branch_name,
             '--parent', NEON_BRANCH)
    print(f"  ✓ Save state created: {branch_name}")


def dump_local():
    """Dump the local database to /tmp."""
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
    """Restore local dump to Neon main."""
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


def cleanup():
    if DUMP_FILE.exists():
        DUMP_FILE.unlink()
        print(f"  ✓ Removed {DUMP_FILE}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"UPDATE — {TODAY}")
    print(f"{'='*60}\n")

    if not NEON_CONN:
        print("✗ NEON_CONN environment variable not set.")
        print("  export NEON_CONN='postgresql://user:pass@host/db?sslmode=require'")
        sys.exit(1)

    check_dependencies()

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
    cleanup()

    print(f"\n{'='*60}")
    print("✓ Update complete.")
    print(f"{'='*60}\n")