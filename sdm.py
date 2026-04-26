#!/usr/bin/env python3
"""
sdm.py - Species Distribution Model

Produces a habitat-suitability GeoTIFF (0–1) for a target species using:
  - Occurrence points from PostGIS (with GeoJSON fallback)
  - BioClim variables averaged 2015-2024 (bio01/04/05/06/12/15/18/19)

Usage:
    python sdm.py "danaus plexippus"
    python sdm.py "vanessa cardui" --resolution 0.08 --n-absences 10000

DB connection (checked in order):
  1. --db-url flag  e.g. --db-url "$NEON_CONN"
  2. NEON_CONN env var  (full postgres:// URL — same var used by update.py)
  3. Individual env vars: DB_HOST, DB_USER, DB_APP_PASSWORD, DB_NAME
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import psycopg2
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────────

BIOCLIM_DIR    = Path("bioclim-data")
OCCURRENCE_DIR = Path("occurrence-data")
OUTPUT_DIR     = Path("sdm-output")

BIOCLIM_VARS = {
    "bio01": "Annual Mean Temperature",
    "bio04": "Temperature Seasonality",
    "bio05": "Max Temp of Warmest Month",
    "bio06": "Min Temp of Coldest Month",
    "bio12": "Annual Precipitation",
    "bio15": "Precipitation Seasonality",
    "bio18": "Precipitation of Warmest Quarter",
    "bio19": "Precipitation of Coldest Quarter",
}

DEFAULT_RESOLUTION = 0.1    # degrees ≈ 11 km (matches WorldClim global coverage)

TARGET_CRS = CRS.from_epsg(4326)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Occurrence data ────────────────────────────────────────────────────────────

_db_url: str | None = None   # set by main() from --db-url or env


def connect_db() -> psycopg2.extensions.connection:
    # 1. explicit URL (--db-url flag or NEON_CONN env var)
    url = _db_url or os.environ.get("NEON_CONN", "")
    if url:
        return psycopg2.connect(url)
    # 2. individual env vars (local Postgres)
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ.get("DB_NAME", "lepidoptera_data"),
        user=os.environ["DB_USER"],
        password=os.environ["DB_APP_PASSWORD"],
    )


def get_occurrences_db(species_name: str) -> pd.DataFrame:
    species_name = species_name.replace("-", " ")
    conn = connect_db()
    try:
        sql = """
            SELECT
                ST_X(lo.geom) AS lon,
                ST_Y(lo.geom) AS lat
            FROM lepidoptera_occurrences lo
            JOIN species s ON lo.species_id = s.id
            WHERE LOWER(s.scientific_name) = LOWER(%s)
              AND lo.geom IS NOT NULL
        """
        df = pd.read_sql(sql, conn, params=[species_name])
    finally:
        conn.close()
    return df.drop_duplicates(subset=["lon", "lat"])


def get_occurrences_geojson(species_name: str) -> pd.DataFrame:
    import geopandas as gpd

    slug = species_name.replace(" ", "-").lower()
    path = OCCURRENCE_DIR / f"{slug}-gbif.geojson"
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {path}")
    gdf = gpd.read_file(path)
    df = pd.DataFrame({"lon": gdf.geometry.x, "lat": gdf.geometry.y}).dropna()
    return df.drop_duplicates(subset=["lon", "lat"])


def get_occurrences(species_name: str) -> pd.DataFrame:
    """Load presence points; try PostGIS first, fall back to GeoJSON."""
    try:
        df = get_occurrences_db(species_name)
        if df.empty:
            raise ValueError("No rows returned from DB")
        log.info(f"Loaded {len(df):,} presence points from PostGIS")
        return df
    except Exception as db_err:
        log.warning(f"PostGIS unavailable ({db_err}); trying GeoJSON fallback")
        df = get_occurrences_geojson(species_name)
        log.info(f"Loaded {len(df):,} presence points from GeoJSON")
        return df


def get_target_group_absences(species_name: str, n: int) -> pd.DataFrame | None:
    """
    Sample background points from other lepidoptera in the DB.
    Returns None if the DB is unavailable or contains too few records.
    """
    species_name = species_name.replace("-", " ")
    try:
        conn = connect_db()
        try:
            sql = """
                SELECT ST_X(lo.geom) AS lon, ST_Y(lo.geom) AS lat
                FROM lepidoptera_occurrences lo
                JOIN species s ON lo.species_id = s.id
                WHERE LOWER(s.scientific_name) != LOWER(%s)
                  AND lo.geom IS NOT NULL
                ORDER BY RANDOM()
                LIMIT %s
            """
            df = pd.read_sql(sql, conn, params=[species_name, n * 3])
        finally:
            conn.close()
        if len(df) >= n // 2:
            return df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    except Exception:
        pass
    return None

# ── Target grid ────────────────────────────────────────────────────────────────

def build_target_grid(extent: tuple, resolution: float) -> tuple:
    """Return (transform, (height, width), lons, lats) for the output raster."""
    west, south, east, north = extent
    width  = int(round((east  - west)  / resolution))
    height = int(round((north - south) / resolution))
    transform = from_origin(west, north, resolution, resolution)
    # Cell-centre coordinates
    lons = west  + (np.arange(width)  + 0.5) * resolution
    lats = north - (np.arange(height) + 0.5) * resolution   # descending
    return transform, (height, width), lons, lats

# ── Climate ────────────────────────────────────────────────────────────────────

def bioclim_extent() -> tuple[float, float, float, float]:
    """Return (west, south, east, north) of the BioClim data files."""
    path = BIOCLIM_DIR / f"{next(iter(BIOCLIM_VARS))}_mean.tif"
    if not path.exists():
        raise FileNotFoundError(f"BioClim file not found: {path}. Run download_bioclim.py first.")
    with rasterio.open(path) as src:
        b = src.bounds
    return b.left, b.bottom, b.right, b.top


def load_bioclim_features(
    target_transform: rasterio.Affine,
    target_shape: tuple[int, int],
    target_crs: CRS,
) -> dict[str, np.ndarray]:
    """Reproject each BioClim mean TIF to the target grid and return as a dict."""
    features: dict[str, np.ndarray] = {}
    for var in BIOCLIM_VARS:
        path = BIOCLIM_DIR / f"{var}_mean.tif"
        if not path.exists():
            raise FileNotFoundError(
                f"BioClim file not found: {path}. Run download_bioclim.py first."
            )
        dest = np.full(target_shape, np.nan, dtype=np.float32)
        with rasterio.open(path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
        features[var] = dest
        log.info(f"  Loaded {var} ({BIOCLIM_VARS[var]})")
    return features


# ── Land cover ─────────────────────────────────────────────────────────────────

# ── Raster sampling ────────────────────────────────────────────────────────────

def sample_raster(
    predictor_stack: np.ndarray,   # (n_features, H, W)
    tgt_lons: np.ndarray,
    tgt_lats: np.ndarray,
    pts_lon: np.ndarray,
    pts_lat: np.ndarray,
) -> np.ndarray:
    """Extract predictor values at lon/lat points. Returns (n_pts, n_features)."""
    lat_idx = np.abs(tgt_lats[:, None] - pts_lat[None, :]).argmin(axis=0)
    lon_idx = np.abs(tgt_lons[:, None] - pts_lon[None, :]).argmin(axis=0)
    return predictor_stack[:, lat_idx, lon_idx].T

# ── Model ──────────────────────────────────────────────────────────────────────

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    train_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    log.info(f"Training AUC: {train_auc:.3f}")

    importance = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
    )
    log.info("Feature importances:\n" + importance.to_string())
    return model

# ── Output ─────────────────────────────────────────────────────────────────────

def save_suitability_raster(
    suitability: np.ndarray,
    transform: rasterio.Affine,
    crs: CRS,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=suitability.shape[0],
        width=suitability.shape[1],
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="lzw",
    ) as dst:
        dst.write(suitability.astype(np.float32), 1)
        dst.update_tags(
            band=1,
            description="Habitat suitability (0 = unsuitable, 1 = optimal)",
        )
    log.info(f"Saved → {path}")

def save_preview_png(
    suitability: np.ndarray,
    presence_df: pd.DataFrame,
    extent: tuple,
    species_name: str,
    path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, s, e, n = extent
    height, width = suitability.shape

    # Coordinate arrays matching the suitability grid
    lons = np.linspace(w, e, width)
    lats = np.linspace(n, s, height)   # descending (origin="upper")
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    fig, ax = plt.subplots(figsize=(14, 8))

    # ── Potential habitat heatmap ─────────────────────────────────────────────
    im = ax.imshow(
        suitability,
        origin="upper",
        extent=[w, e, s, n],
        cmap="YlOrRd",
        vmin=0, vmax=1,
        interpolation="bilinear",
    )

    # ── Predicted range boundary (suitability = 0.5 contour) ─────────────────
    suit_clean = np.where(np.isfinite(suitability), suitability, 0.0)
    ax.contour(
        lon_grid, lat_grid, suit_clean,
        levels=[0.5],
        colors=["#1a1a2e"],
        linewidths=1.2,
        linestyles="solid",
    )
    # Invisible proxy for legend
    ax.plot([], [], color="#1a1a2e", linewidth=1.2, linestyle="solid",
            label="Predicted range boundary (p = 0.5)")

    # ── Actual observed range (KDE contour) ───────────────────────────────────
    pts = presence_df[["lon", "lat"]].dropna()
    sample = pts if len(pts) <= 10_000 else pts.sample(10_000, random_state=42)
    kde = gaussian_kde(sample[["lon", "lat"]].values.T, bw_method=0.08)

    # Find density threshold enclosing 90% of occurrences
    occ_densities = kde(pts[["lon", "lat"]].values.T)
    threshold = np.percentile(occ_densities, 10)

    # Evaluate KDE on a coarse grid (1° steps) then let matplotlib interpolate the contour
    coarse_lons = np.arange(w, e, 1.0)
    coarse_lats = np.arange(n, s, -1.0)
    clon_grid, clat_grid = np.meshgrid(coarse_lons, coarse_lats)
    kde_vals = kde(np.vstack([clon_grid.ravel(), clat_grid.ravel()])).reshape(clat_grid.shape)
    ax.contour(
        clon_grid, clat_grid, kde_vals,
        levels=[threshold],
        colors=["#00b4d8"],
        linewidths=1.4,
        linestyles="dashed",
    )
    ax.plot([], [], color="#00b4d8", linewidth=1.4, linestyle="dashed",
            label="Observed range (90% occurrence density)")

    # ── Labels & legend ───────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Habitat suitability (potential)")
    ax.set_title(f"{species_name.replace('-', ' ').title()} — Potential vs. Realized Habitat",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower right", framealpha=0.85, fontsize=9)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Preview saved → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Species Distribution Model — produces a habitat suitability raster."
    )
    p.add_argument("species", help="Scientific name, e.g. 'danaus plexippus'")
    p.add_argument(
        "--extent", nargs=4, type=float, metavar=("W", "S", "E", "N"),
        default=None,
        help="Study extent in WGS84 decimal degrees (default: derived from BioClim data)",
    )
    p.add_argument(
        "--resolution", type=float, default=DEFAULT_RESOLUTION,
        help="Output cell size in degrees (default: 0.04 ≈ 4 km)",
    )
    p.add_argument(
        "--n-absences", type=int, default=5000,
        help="Number of pseudo-absence points (default: 5000)",
    )
    p.add_argument(
        "--target-group", action="store_true",
        help="Use other lepidoptera occurrences as background (requires DB)",
    )
    p.add_argument(
        "--db-url",
        help="PostgreSQL connection URL (overrides NEON_CONN / individual DB_* vars)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument(
        "--preview", action="store_true",
        help="Save a PNG preview with occurrence points overlaid instead of writing the TIFF",
    )
    return p.parse_args()

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    global _db_url
    args = parse_args()
    _db_url = args.db_url   # may be None; connect_db() falls back to NEON_CONN / env vars
    rng = np.random.default_rng(args.seed)
    extent: tuple = tuple(args.extent) if args.extent else bioclim_extent()
    west, south, east, north = extent
    log.info(f"Study extent: W={west} S={south} E={east} N={north}")

    # 1 ── Occurrence data ─────────────────────────────────────────────────────
    log.info(f"Species: {args.species}")
    presence_df = get_occurrences(args.species)

    # Clip to study extent
    presence_df = presence_df[
        presence_df["lon"].between(west, east) &
        presence_df["lat"].between(south, north)
    ].reset_index(drop=True)
    log.info(f"{len(presence_df):,} presence points within study extent")

    if len(presence_df) < 10:
        sys.exit("Too few presence points (<10) — aborting.")

    # 2 ── Target grid ─────────────────────────────────────────────────────────
    transform, shape, tgt_lons, tgt_lats = build_target_grid(extent, args.resolution)
    height, width = shape
    log.info(f"Grid: {height} rows × {width} cols at {args.resolution}° resolution")

    # 3 ── Climate features ────────────────────────────────────────────────────
    log.info("Loading BioClim features...")
    climate_grids = load_bioclim_features(transform, shape, TARGET_CRS)

    # 4 ── Predictor stack ─────────────────────────────────────────────────────
    feature_names = list(climate_grids.keys())
    predictor_stack = np.stack(list(climate_grids.values()), axis=0)  # (n_features, H, W)

    valid_mask = np.all(np.isfinite(predictor_stack), axis=0)
    log.info(f"Valid pixels: {valid_mask.sum():,} / {valid_mask.size:,}")

    # 5 ── Clip presences to valid pixels ─────────────────────────────────────
    pres_lat_idx = np.abs(tgt_lats[:, None] - presence_df["lat"].values[None, :]).argmin(axis=0)
    pres_lon_idx = np.abs(tgt_lons[:, None] - presence_df["lon"].values[None, :]).argmin(axis=0)
    in_valid = valid_mask[pres_lat_idx, pres_lon_idx]
    presence_df = presence_df[in_valid].reset_index(drop=True)
    log.info(f"{len(presence_df):,} presence points within valid BioClim coverage")

    if len(presence_df) < 10:
        sys.exit("Too few presence points within BioClim coverage (<10) — aborting.")

    # 6 ── Pseudo-absences sampled from valid pixels ───────────────────────────
    if args.target_group:
        absence_df = get_target_group_absences(args.species, args.n_absences)
        if absence_df is not None:
            log.info(f"Using {len(absence_df):,} target-group background points")
        else:
            log.warning("Target-group background unavailable; falling back to random")
            absence_df = None

    if not args.target_group or absence_df is None:
        valid_indices = np.argwhere(valid_mask)
        chosen = rng.choice(len(valid_indices), size=min(args.n_absences, len(valid_indices)), replace=False)
        abs_rows, abs_cols = valid_indices[chosen, 0], valid_indices[chosen, 1]
        absence_df = pd.DataFrame({"lon": tgt_lons[abs_cols], "lat": tgt_lats[abs_rows]})
        log.info(f"Generated {len(absence_df):,} pseudo-absence points from valid pixels")

    # 7 ── Training data ──────────────────────────────────────────────────────
    X_pres = sample_raster(predictor_stack, tgt_lons, tgt_lats,
                           presence_df["lon"].values, presence_df["lat"].values)
    X_abs  = sample_raster(predictor_stack, tgt_lons, tgt_lats,
                           absence_df["lon"].values, absence_df["lat"].values)

    X = np.vstack([X_pres, X_abs])
    y = np.concatenate([np.ones(len(X_pres)), np.zeros(len(X_abs))])

    # Drop points that landed on nodata pixels
    valid_rows = np.all(np.isfinite(X), axis=1)
    X, y = X[valid_rows], y[valid_rows]
    log.info(
        f"Training set: {int(y.sum()):,} presences, {int((y == 0).sum()):,} absences"
    )

    # 8 ── Train model ─────────────────────────────────────────────────────────
    log.info("Training Random Forest...")
    model = train_model(X, y, feature_names)

    # 9 ── Predict suitability ─────────────────────────────────────────────────
    log.info("Predicting suitability across full grid...")
    flat_X = predictor_stack[:, valid_mask].T     # (n_valid, n_features)
    prob   = model.predict_proba(flat_X)[:, 1]

    suitability = np.full(shape, np.nan, dtype=np.float32)
    suitability[valid_mask] = prob

    log.info(
        f"Suitability range: {np.nanmin(suitability):.3f} – {np.nanmax(suitability):.3f}"
    )

    # 10 ── Save output ────────────────────────────────────────────────────────
    slug = args.species.replace("-", " ").replace(" ", "_").lower()
    if args.preview:
        out_path = args.output_dir / f"{slug}_preview.png"
        save_preview_png(suitability, presence_df, extent, args.species, out_path)
    else:
        out_path = args.output_dir / f"{slug}_suitability.tif"
        save_suitability_raster(suitability, transform, TARGET_CRS, out_path)


if __name__ == "__main__":
    main()
