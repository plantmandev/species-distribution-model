#!/usr/bin/env python3
"""
batch_sdm.py — Run sdm.py for all ingested lepidoptera species.

Reads the species list from PostGIS (or species-metadata.csv as fallback),
skips species with too few occurrence points, skips species that already
have output files, and calls sdm.py per species in a subprocess so that
per-species failures never abort the batch.

Results are appended to sdm-output/batch_results.csv after each run.

Usage:
    python batch_sdm.py                         # run all pending species
    python batch_sdm.py --dry-run               # preview without executing
    python batch_sdm.py --min-obs 30            # stricter occurrence filter
    python batch_sdm.py --validate              # include spatial CV + Boyce index
    python batch_sdm.py --workers 2             # parallel species (see note below)
    python batch_sdm.py --force                 # re-run even if output exists
    python batch_sdm.py --species "Danaus plexippus"  # single species

Note on --workers:
    sdm.py runs RandomForest with n_jobs=-1 (all CPU cores). Running multiple
    species in parallel will cause CPU contention. Only use --workers > 1 on
    machines with many cores or when speed matters more than per-run performance.
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR    = Path("sdm-output")
METADATA_FILE = Path("occurrence-data/species-metadata.csv")
RESULTS_LOG   = OUTPUT_DIR / "batch_results.csv"
SDM_SCRIPT    = Path(__file__).parent / "sdm.py"
PROJECT_DIR   = Path(__file__).parent

DEFAULT_MIN_OBS = 10
DEFAULT_TIMEOUT = 1800   # 30 min per species

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def slug(name: str) -> str:
    return name.replace("-", " ").replace(" ", "_").lower()


def output_exists(species_name: str, output_dir: Path) -> bool:
    s = slug(species_name)
    return (output_dir / f"{s}_suitability.tif").exists()


# ── Species list sources ───────────────────────────────────────────────────────

def get_species_from_db(min_obs: int) -> list[dict]:
    """
    Query PostGIS for all species that have at least min_obs georeferenced
    occurrences in lepidoptera_occurrences. Returns accurate live counts.
    """
    url = os.environ.get("NEON_CONN", "")
    try:
        import psycopg2

        conn = (
            psycopg2.connect(url)
            if url
            else psycopg2.connect(
                host=os.environ.get("DB_HOST", "localhost"),
                port=int(os.environ.get("DB_PORT", 5432)),
                dbname=os.environ.get("DB_NAME", "lepidoptera_data"),
                user=os.environ["DB_USER"],
                password=os.environ["DB_APP_PASSWORD"],
            )
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT s.scientific_name, COUNT(lo.id) AS obs_count
                    FROM species s
                    JOIN lepidoptera_occurrences lo ON lo.species_id = s.id
                    WHERE lo.geom IS NOT NULL
                    GROUP BY s.scientific_name
                    HAVING COUNT(lo.id) >= %s
                    ORDER BY s.scientific_name
                    """,
                    (min_obs,),
                )
                rows = [
                    {"species_name": r[0], "actual_obs": int(r[1])}
                    for r in cur.fetchall()
                ]
                log.info(f"Loaded {len(rows)} species from PostGIS (>= {min_obs} obs)")
                return rows
        finally:
            conn.close()
    except Exception as e:
        log.warning(f"PostGIS unavailable ({e}); falling back to metadata CSV")
        return []


def get_species_from_csv(min_obs: int) -> list[dict]:
    """Read ingested lepidoptera species from species-metadata.csv."""
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

    species = []
    with open(METADATA_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("category", "").strip() != "lepidoptera":
                continue
            if row.get("status", "").strip() != "ingested":
                continue
            try:
                obs = float(row.get("actual_obs") or 0)
            except ValueError:
                obs = 0
            if obs >= min_obs:
                species.append({"species_name": row["species_name"], "actual_obs": int(obs)})

    log.info(f"Loaded {len(species)} species from metadata CSV (>= {min_obs} obs)")
    return species


# ── Subprocess runner ──────────────────────────────────────────────────────────

def run_species(
    species_name: str,
    extra_args: list[str],
    timeout: int,
) -> dict:
    """
    Run sdm.py for one species in a subprocess.
    Returns a result dict with status, duration, and any error snippet.
    """
    start = time.time()
    cmd = [sys.executable, str(SDM_SCRIPT), species_name] + extra_args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_DIR,
        )
        duration = round(time.time() - start, 1)

        if result.returncode == 0:
            return {
                "species": species_name,
                "status": "success",
                "duration": duration,
                "error": "",
            }

        # Capture the last meaningful line from stderr/stdout for the log
        combined = (result.stderr or "") + (result.stdout or "")
        last_line = next(
            (ln.strip() for ln in reversed(combined.splitlines()) if ln.strip()),
            f"exit code {result.returncode}",
        )
        return {
            "species": species_name,
            "status": "failed",
            "duration": duration,
            "error": last_line[:200],
        }

    except subprocess.TimeoutExpired:
        return {
            "species": species_name,
            "status": "timeout",
            "duration": timeout,
            "error": f"exceeded {timeout}s limit",
        }
    except Exception as e:
        return {
            "species": species_name,
            "status": "error",
            "duration": round(time.time() - start, 1),
            "error": str(e)[:200],
        }


# ── Results log ────────────────────────────────────────────────────────────────

def init_results_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "species", "status", "duration_s", "error"])


def append_result(path: Path, result: dict) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result["species"],
            result["status"],
            result["duration"],
            result["error"],
        ])


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-run sdm.py for all ingested lepidoptera species.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--min-obs", type=int, default=DEFAULT_MIN_OBS,
        help=f"Skip species with fewer occurrences than this (default: {DEFAULT_MIN_OBS})",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel species to run (default: 1; see module docstring)",
    )
    p.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Per-species timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run species that already have a suitability.tif in output-dir",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the run plan without executing anything",
    )
    p.add_argument(
        "--species", type=str, default=None,
        help="Run a single species by name (overrides list from DB / CSV)",
    )

    # ── sdm.py pass-through flags ─────────────────────────────────────────────
    sdm = p.add_argument_group("sdm.py options (passed through)")
    sdm.add_argument("--validate",    action="store_true", help="Spatial CV + Boyce index")
    sdm.add_argument("--preview",     action="store_true", help="Save PNG previews instead of TIFFs")
    sdm.add_argument("--target-group", action="store_true", help="Use target-group background points")
    sdm.add_argument("--resolution",  type=float, default=None, help="Output resolution in degrees")
    sdm.add_argument("--n-absences",  type=int,   default=None, help="Pseudo-absence count")
    sdm.add_argument("--block-size",  type=float, default=None, help="CV block size in degrees")
    sdm.add_argument("--range-coverage", type=float, default=None, help="Range polygon coverage fraction")
    sdm.add_argument("--output-dir",  type=Path,  default=OUTPUT_DIR, help=f"Output directory (default: {OUTPUT_DIR})")
    sdm.add_argument("--db-url",      type=str,   default=None, help="PostgreSQL connection URL")

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.db_url:
        os.environ["NEON_CONN"] = args.db_url

    if args.workers > 1:
        log.warning(
            f"--workers {args.workers}: each sdm.py run uses all CPU cores for Random Forest. "
            "Expect contention."
        )

    # ── Build species list ────────────────────────────────────────────────────
    if args.species:
        all_species = [{"species_name": args.species, "actual_obs": "?"}]
    else:
        all_species = get_species_from_db(args.min_obs) or get_species_from_csv(args.min_obs)
        if not all_species:
            log.error("No species found. Check DB connection or metadata CSV.")
            sys.exit(1)

    # ── Skip already-complete species (unless --force) ────────────────────────
    pending, skipped = [], []
    for sp in all_species:
        if not args.force and output_exists(sp["species_name"], args.output_dir):
            skipped.append(sp)
        else:
            pending.append(sp)

    log.info(
        f"Plan: {len(pending)} to run | {len(skipped)} already complete | "
        f"{len(all_species)} total"
    )
    if skipped:
        log.info("  (use --force to re-run completed species)")

    if not pending:
        log.info("Nothing to run.")
        return

    # ── Build pass-through args for sdm.py ───────────────────────────────────
    extra: list[str] = ["--output-dir", str(args.output_dir)]
    if args.validate:        extra.append("--validate")
    if args.preview:         extra.append("--preview")
    if args.target_group:    extra.append("--target-group")
    if args.resolution:      extra += ["--resolution",      str(args.resolution)]
    if args.n_absences:      extra += ["--n-absences",      str(args.n_absences)]
    if args.block_size:      extra += ["--block-size",      str(args.block_size)]
    if args.range_coverage:  extra += ["--range-coverage",  str(args.range_coverage)]
    if args.db_url:          extra += ["--db-url",          args.db_url]

    # ── Dry-run: print plan and exit ──────────────────────────────────────────
    if args.dry_run:
        print(f"\nDRY RUN — would run {len(pending)} species:")
        for sp in pending:
            print(f"  {sp['species_name']}  ({sp['actual_obs']} obs)")
        if skipped:
            print(f"\nWould skip {len(skipped)} already-complete species.")
        return

    # ── Initialise results log ────────────────────────────────────────────────
    init_results_log(RESULTS_LOG)
    log.info(f"Logging results to {RESULTS_LOG}")

    names = [sp["species_name"] for sp in pending]
    success = failed = 0

    # ── Sequential run (default) ──────────────────────────────────────────────
    if args.workers == 1:
        for i, name in enumerate(names, 1):
            log.info(f"[{i}/{len(names)}] {name}")
            result = run_species(name, extra, args.timeout)
            append_result(RESULTS_LOG, result)

            if result["status"] == "success":
                success += 1
                log.info(f"  ✓ done in {result['duration']}s")
            else:
                failed += 1
                log.warning(f"  ✗ {result['status']} ({result['duration']}s) — {result['error']}")

    # ── Parallel run ──────────────────────────────────────────────────────────
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_species, name, extra, args.timeout): name
                for name in names
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                result = future.result()
                append_result(RESULTS_LOG, result)
                sym = "✓" if result["status"] == "success" else "✗"
                log.info(
                    f"[{done}/{len(names)}] {sym} {result['species']} "
                    f"— {result['status']} ({result['duration']}s)"
                )
                if result["status"] != "success" and result["error"]:
                    log.warning(f"  {result['error']}")
                if result["status"] == "success":
                    success += 1
                else:
                    failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(
        f"\nFinished: {success}/{len(names)} succeeded | {failed} failed "
        f"| results → {RESULTS_LOG}"
    )


if __name__ == "__main__":
    main()
