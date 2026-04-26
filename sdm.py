#!/usr/bin/env python3
"""
sdm.py - Species Distribution Model

Produces a habitat-suitability GeoTIFF (0–1) for a target species using:
  - Occurrence points from PostGIS (with GeoJSON fallback)
  - TerraClimate climate variables (tmax, tmin, precipitation)
  - NALCMS 2020 land cover raster

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

import numpy as np
import pandas as pd
import psycopg2
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.interpolate import RegularGridInterpolator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────────

CLIMATE_DIR = Path("climate-data")
LAND_COVER_PATH = Path("land-cover-data/data/NA_NALCMS_landcover_2020v2_30m.tif")
OCCURRENCE_DIR = Path("occurrence-data")
OUTPUT_DIR = Path("sdm-output")

# TerraClimate subdirectory names keyed by variable
CLIMATE_SUBDIRS = {
    "tmax": "Maximum Temperature",
    "tmin": "Minimum Temperature",
    "ppt":  "Precipitation",
}

# Study area: North America + Central America
DEFAULT_EXTENT = (-180.0, 7.0, -50.0, 84.0)   # west, south, east, north
DEFAULT_RESOLUTION = 0.04                        # degrees  ≈ 4 km

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
    try:
        conn = connect_db()
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

def lat_coord(da: xr.DataArray) -> str:
    return "lat" if "lat" in da.coords else "latitude"


def lon_coord(da: xr.DataArray) -> str:
    return "lon" if "lon" in da.coords else "longitude"


def compute_climate_features_streamed(
    lat_mask: np.ndarray,
    lon_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Derive 9 bioclimate-style predictor grids from monthly TerraClimate data,
    processing one year at a time to stay within memory limits.

    Peak memory: ~4 × (12 × H_sub × W_sub × float32) ≈ 1–2 GB.
    Returns (feature_dict, sub_lats, sub_lons).
    """
    tmax_dir = CLIMATE_DIR / CLIMATE_SUBDIRS["tmax"]
    tmin_dir = CLIMATE_DIR / CLIMATE_SUBDIRS["tmin"]
    ppt_dir  = CLIMATE_DIR / CLIMATE_SUBDIRS["ppt"]

    years = sorted({f.stem.split("_")[-1] for f in tmax_dir.glob("TerraClimate_tmax_*.nc")})
    if not years:
        raise FileNotFoundError(f"No TerraClimate tmax files found in {tmax_dir}")

    # acc[var] accumulates: sum, sum_sq, count, min, max over all months
    acc: dict[str, dict] = {}
    sub_lats: np.ndarray | None = None
    sub_lons: np.ndarray | None = None

    for year in years:
        var_files = {
            "tmax": tmax_dir / f"TerraClimate_tmax_{year}.nc",
            "tmin": tmin_dir / f"TerraClimate_tmin_{year}.nc",
            "ppt":  ppt_dir  / f"TerraClimate_ppt_{year}.nc",
        }

        subs: dict[str, np.ndarray] = {}
        for var, fpath in var_files.items():
            with xr.open_dataset(fpath, decode_times=True, mask_and_scale=True) as ds:
                da = ds[var]
                if sub_lats is None:
                    sub_lats = da.coords[lat_coord(da)].values[lat_mask]
                    sub_lons = da.coords[lon_coord(da)].values[lon_mask]
                subs[var] = da.isel({
                    lat_coord(da): lat_mask,
                    lon_coord(da): lon_mask,
                }).values.astype(np.float32)  # (12, H_sub, W_sub)

        subs["tmean"] = (subs["tmax"] + subs["tmin"]) / 2.0

        for var, data in subs.items():
            valid = np.isfinite(data)
            b = {
                "sum":    np.nansum(data,    axis=0),
                "sum_sq": np.nansum(data**2, axis=0),
                "count":  valid.sum(axis=0).astype(np.float32),
                "min":    np.where(valid.any(axis=0), np.nanmin(data, axis=0),  np.inf),
                "max":    np.where(valid.any(axis=0), np.nanmax(data, axis=0), -np.inf),
            }
            if var not in acc:
                acc[var] = b
            else:
                acc[var]["sum"]    += b["sum"]
                acc[var]["sum_sq"] += b["sum_sq"]
                acc[var]["count"]  += b["count"]
                acc[var]["min"]     = np.fmin(acc[var]["min"], b["min"])
                acc[var]["max"]     = np.fmax(acc[var]["max"], b["max"])

        log.info(f"  Processed year {year}")

    def finalize(a: dict) -> dict[str, np.ndarray]:
        with np.errstate(invalid="ignore", divide="ignore"):
            cnt  = np.where(a["count"] > 0, a["count"], np.nan)
            mean = (a["sum"] / cnt).astype(np.float32)
            std  = np.sqrt(np.maximum(a["sum_sq"] / cnt - mean**2, 0)).astype(np.float32)
            mn   = np.where(np.isfinite(a["min"]), a["min"], np.nan).astype(np.float32)
            mx   = np.where(np.isfinite(a["max"]), a["max"], np.nan).astype(np.float32)
        return {"mean": mean, "std": std, "min": mn, "max": mx}

    s = {v: finalize(acc[v]) for v in acc}

    with np.errstate(invalid="ignore", divide="ignore"):
        ppt_cv = np.where(
            s["ppt"]["mean"] > 0, s["ppt"]["std"] / s["ppt"]["mean"], 0.0
        ).astype(np.float32)

    features: dict[str, np.ndarray] = {
        "mean_tmax":          s["tmax"]["mean"],
        "mean_tmin":          s["tmin"]["mean"],
        "mean_temp":          s["tmean"]["mean"],
        "temp_range":         s["tmax"]["mean"] - s["tmin"]["mean"],
        "temp_seasonality":   s["tmean"]["std"],
        "annual_precip":      s["ppt"]["mean"] * 12,
        "precip_driest":      s["ppt"]["min"],
        "precip_wettest":     s["ppt"]["max"],
        "precip_seasonality": ppt_cv,
    }

    return features, sub_lats, sub_lons


def align_to_grid(
    arr: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    tgt_lats: np.ndarray,
    tgt_lons: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour regrid from TerraClimate (~0.042°) to target grid."""
    # Ensure lats ascending for interpolator
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        arr = arr[::-1, :]

    interp = RegularGridInterpolator(
        (src_lats, src_lons), arr,
        method="nearest",
        bounds_error=False,
        fill_value=np.nan,
    )
    lon_grid, lat_grid = np.meshgrid(tgt_lons, tgt_lats)
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    return interp(pts).reshape(len(tgt_lats), len(tgt_lons)).astype(np.float32)

# ── Land cover ─────────────────────────────────────────────────────────────────

def resample_land_cover(
    target_transform: rasterio.Affine,
    target_shape: tuple[int, int],
    target_crs: CRS,
) -> np.ndarray:
    """
    Warp NALCMS 30m land cover to the target grid using mode resampling.
    Streams the 3.2 GB source in blocks — memory usage stays manageable.
    """
    log.info("Warping land cover (this may take a moment)...")
    dest = np.zeros(target_shape, dtype=np.uint8)
    with rasterio.open(LAND_COVER_PATH) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.mode,
        )
    dest = dest.astype(np.float32)
    dest[dest == 0] = np.nan    # NALCMS nodata = 0
    return dest

# ── Pseudo-absences ────────────────────────────────────────────────────────────

def generate_random_absences(
    extent: tuple,
    n: int,
    presence_df: pd.DataFrame,
    rng: np.random.Generator,
    min_dist_deg: float = 1.0,
) -> pd.DataFrame:
    """Random background points at least min_dist_deg away from any presence."""
    west, south, east, north = extent
    pres_lon = presence_df["lon"].values
    pres_lat = presence_df["lat"].values
    pts: list[dict] = []
    max_attempts = n * 30
    attempts = 0
    while len(pts) < n and attempts < max_attempts:
        lon = rng.uniform(west, east)
        lat = rng.uniform(south, north)
        if not (
            (np.abs(pres_lon - lon) < min_dist_deg) &
            (np.abs(pres_lat - lat) < min_dist_deg)
        ).any():
            pts.append({"lon": lon, "lat": lat})
        attempts += 1
    if len(pts) < n // 2:
        log.warning(f"Only generated {len(pts)} pseudo-absences (requested {n})")
    return pd.DataFrame(pts)

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

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Species Distribution Model — produces a habitat suitability raster."
    )
    p.add_argument("species", help="Scientific name, e.g. 'danaus plexippus'")
    p.add_argument(
        "--extent", nargs=4, type=float, metavar=("W", "S", "E", "N"),
        default=list(DEFAULT_EXTENT),
        help="Study extent in WGS84 decimal degrees (default: North America)",
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
    return p.parse_args()

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    global _db_url
    args = parse_args()
    _db_url = args.db_url   # may be None; connect_db() falls back to NEON_CONN / env vars
    rng = np.random.default_rng(args.seed)
    extent: tuple = tuple(args.extent)
    west, south, east, north = extent

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
    # Read the coordinate grid from the first tmax file (metadata only, no data loaded)
    log.info("Loading climate data...")
    _tmax_files = sorted((CLIMATE_DIR / CLIMATE_SUBDIRS["tmax"]).glob("TerraClimate_tmax_*.nc"))
    if not _tmax_files:
        sys.exit(f"No TerraClimate tmax files found in {CLIMATE_DIR / CLIMATE_SUBDIRS['tmax']}")
    with xr.open_dataset(_tmax_files[0], mask_and_scale=False) as _ds:
        _da = _ds["tmax"]
        _src_lats = _da.coords[lat_coord(_da)].values
        _src_lons = _da.coords[lon_coord(_da)].values

    lat_mask = (_src_lats >= south - 2) & (_src_lats <= north + 2)
    lon_mask = (_src_lons >= west  - 2) & (_src_lons <= east  + 2)

    raw_features, sub_lats, sub_lons = compute_climate_features_streamed(lat_mask, lon_mask)

    log.info("Aligning climate features to target grid...")
    climate_grids: dict[str, np.ndarray] = {}
    for name, arr in raw_features.items():
        climate_grids[name] = align_to_grid(arr, sub_lats, sub_lons, tgt_lats, tgt_lons)

    # 4 ── Land cover ──────────────────────────────────────────────────────────
    land_cover = resample_land_cover(transform, shape, TARGET_CRS)

    # 5 ── Predictor stack ─────────────────────────────────────────────────────
    feature_names = list(climate_grids.keys()) + ["land_cover"]
    predictor_stack = np.stack(
        [*climate_grids.values(), land_cover], axis=0
    )  # (n_features, H, W)

    valid_mask = np.all(np.isfinite(predictor_stack), axis=0)
    log.info(f"Valid pixels: {valid_mask.sum():,} / {valid_mask.size:,}")

    # 6 ── Pseudo-absences ─────────────────────────────────────────────────────
    if args.target_group:
        absence_df = get_target_group_absences(args.species, args.n_absences)
        if absence_df is not None:
            log.info(f"Using {len(absence_df):,} target-group background points")
        else:
            log.warning("Target-group background unavailable; falling back to random")
            absence_df = None

    if not args.target_group or absence_df is None:
        absence_df = generate_random_absences(extent, args.n_absences, presence_df, rng)
        log.info(f"Generated {len(absence_df):,} random pseudo-absence points")

    # 7 ── Training data ───────────────────────────────────────────────────────
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
    slug = args.species.replace(" ", "_").lower()
    out_path = args.output_dir / f"{slug}_suitability.tif"
    save_suitability_raster(suitability, transform, TARGET_CRS, out_path)


if __name__ == "__main__":
    main()
