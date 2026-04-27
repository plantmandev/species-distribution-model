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
    # Direct index calculation — O(N) vs the previous O(H*N + W*N) argmin approach
    lon_step = tgt_lons[1] - tgt_lons[0]
    lat_step = tgt_lats[1] - tgt_lats[0]   # negative (grid is north→south)
    lon_idx = np.clip(np.round((pts_lon - tgt_lons[0]) / lon_step).astype(int), 0, len(tgt_lons) - 1)
    lat_idx = np.clip(np.round((pts_lat - tgt_lats[0]) / lat_step).astype(int), 0, len(tgt_lats) - 1)
    return predictor_stack[:, lat_idx, lon_idx].T

# ── Predictor selection ────────────────────────────────────────────────────────

def filter_correlated_predictors(
    predictor_stack: np.ndarray,
    feature_names: list[str],
    valid_mask: np.ndarray,
    threshold: float = 0.85,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Remove one predictor from each pair with Pearson |r| > threshold."""
    valid_idx = np.argwhere(valid_mask)
    sel = (rng or np.random.default_rng(42)).choice(
        len(valid_idx), size=min(10_000, len(valid_idx)), replace=False
    )
    rows, cols = valid_idx[sel, 0], valid_idx[sel, 1]
    corr = np.corrcoef(predictor_stack[:, rows, cols])

    n = len(feature_names)
    drop: set[int] = set()
    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j not in drop and abs(corr[i, j]) > threshold:
                drop.add(j)
                log.info(
                    f"  Collinearity: dropping {feature_names[j]} "
                    f"(|r|={abs(corr[i, j]):.2f} with {feature_names[i]})"
                )

    keep = [i for i in range(n) if i not in drop]
    log.info(
        f"Predictors after collinearity filter: "
        f"{[feature_names[i] for i in keep]} ({len(keep)}/{n} kept)"
    )
    return predictor_stack[keep], [feature_names[i] for i in keep]


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


class _Ensemble:
    """Averages predict_proba from multiple fitted classifiers."""
    def __init__(self, models: list) -> None:
        self.models = models

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        avg = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1 - avg, avg])

# ── Validation ─────────────────────────────────────────────────────────────────

def assign_spatial_blocks(df: pd.DataFrame, block_size: float) -> pd.DataFrame:
    """Label each row with a lat/lon block ID for spatial CV."""
    df = df.copy()
    df["block_id"] = (
        (df["lon"] // block_size).astype(int).astype(str)
        + "_"
        + (df["lat"] // block_size).astype(int).astype(str)
    )
    return df


def spatial_block_cv(
    presence_df: pd.DataFrame,
    absence_df: pd.DataFrame,
    predictor_stack: np.ndarray,
    tgt_lons: np.ndarray,
    tgt_lats: np.ndarray,
    feature_names: list[str],
    block_size: float = 10.0,
) -> dict:
    """
    Leave-one-block-out spatial cross-validation.
    Returns mean AUC, TSS, sensitivity, and specificity across folds.
    """
    pres = assign_spatial_blocks(presence_df, block_size)
    abss = assign_spatial_blocks(absence_df, block_size)
    blocks = sorted(pres["block_id"].unique())
    log.info(f"Spatial CV: {len(blocks)} blocks at {block_size}° resolution")

    # Pre-sample feature matrices once — avoids repeating the lookup N_folds times
    X_pres_all = sample_raster(predictor_stack, tgt_lons, tgt_lats,
                               pres["lon"].values, pres["lat"].values)
    X_abs_all  = sample_raster(predictor_stack, tgt_lons, tgt_lats,
                               abss["lon"].values, abss["lat"].values)
    pres_block = pres["block_id"].values
    abs_block  = abss["block_id"].values

    fold_aucs, fold_tss, fold_sens, fold_spec = [], [], [], []

    for block in blocks:
        pres_te_m = pres_block == block
        abs_te_m  = abs_block  == block

        if pres_te_m.sum() < 5 or (~pres_te_m).sum() < 10:
            continue

        X_pres_tr = X_pres_all[~pres_te_m]; X_pres_te = X_pres_all[pres_te_m]
        X_abs_tr  = X_abs_all[~abs_te_m];   X_abs_te  = X_abs_all[abs_te_m]

        X_tr = np.vstack([X_pres_tr, X_abs_tr])
        y_tr = np.concatenate([np.ones(len(X_pres_tr)), np.zeros(len(X_abs_tr))])
        X_te = np.vstack([X_pres_te, X_abs_te])
        y_te = np.concatenate([np.ones(len(X_pres_te)), np.zeros(len(X_abs_te))])

        valid_tr = np.all(np.isfinite(X_tr), axis=1)
        valid_te = np.all(np.isfinite(X_te), axis=1)
        X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]
        X_te, y_te = X_te[valid_te], y_te[valid_te]

        if len(np.unique(y_te)) < 2:
            continue

        fold_model = RandomForestClassifier(
            n_estimators=100,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        fold_model.fit(X_tr, y_tr)
        y_prob = fold_model.predict_proba(X_te)[:, 1]

        fold_aucs.append(roc_auc_score(y_te, y_prob))

        # Vectorised max-TSS search across all thresholds at once
        thresholds = np.linspace(0, 1, 200)
        y_preds = y_prob[None, :] >= thresholds[:, None]   # (200, n_test)
        pres_m  = y_te == 1
        n_pres  = pres_m.sum()
        n_abs   = (~pres_m).sum()
        tp = y_preds[:, pres_m].sum(axis=1).astype(float)
        fp = y_preds[:, ~pres_m].sum(axis=1).astype(float)
        sens_v = np.where(n_pres > 0, tp / n_pres, 0.0)
        spec_v = np.where(n_abs  > 0, (n_abs - fp) / n_abs, 0.0)
        tss_v  = sens_v + spec_v - 1.0
        best   = int(np.argmax(tss_v))
        fold_tss.append(float(tss_v[best]))
        fold_sens.append(float(sens_v[best]))
        fold_spec.append(float(spec_v[best]))

    if not fold_aucs:
        log.warning("No valid CV folds produced — try a smaller --block-size")
        return {}

    return {
        "n_folds":          len(fold_aucs),
        "auc_mean":         float(np.mean(fold_aucs)),
        "auc_std":          float(np.std(fold_aucs)),
        "tss_mean":         float(np.mean(fold_tss)),
        "tss_std":          float(np.std(fold_tss)),
        "sensitivity_mean": float(np.mean(fold_sens)),
        "specificity_mean": float(np.mean(fold_spec)),
    }


def boyce_index(
    suitability: np.ndarray,
    presence_df: pd.DataFrame,
    tgt_lons: np.ndarray,
    tgt_lats: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Continuous Boyce index — Spearman correlation between bin midpoints and
    the predicted-to-expected (P/E) ratio of presences vs. background area.
    Ranges -1 to +1; values near +1 mean presences cluster in high-suitability cells.
    """
    lat_idx = np.abs(tgt_lats[:, None] - presence_df["lat"].values[None, :]).argmin(axis=0)
    lon_idx = np.abs(tgt_lons[:, None] - presence_df["lon"].values[None, :]).argmin(axis=0)
    pres_suit = suitability[lat_idx, lon_idx]
    pres_suit = pres_suit[np.isfinite(pres_suit)]

    bg_suit = suitability[np.isfinite(suitability)].ravel()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids, pe = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        p = np.mean((pres_suit >= lo) & (pres_suit < hi))
        e = np.mean((bg_suit   >= lo) & (bg_suit   < hi))
        if e > 0:
            mids.append((lo + hi) / 2)
            pe.append(p / e)

    if len(mids) < 3:
        return float("nan")

    # Spearman rank correlation
    mids_arr = np.array(mids, dtype=float)
    pe_arr   = np.array(pe,   dtype=float)
    rank_m = np.argsort(np.argsort(mids_arr)).astype(float)
    rank_p = np.argsort(np.argsort(pe_arr)).astype(float)
    n = len(mids_arr)
    return float(1.0 - 6.0 * np.sum((rank_m - rank_p) ** 2) / (n * (n ** 2 - 1)))


# ── Output ─────────────────────────────────────────────────────────────────────

def build_range_polygon(
    presence_df: pd.DataFrame,
    transform: rasterio.Affine,
    shape: tuple[int, int],
    tgt_lons: np.ndarray,
    tgt_lats: np.ndarray,
    valid_mask: np.ndarray | None = None,
    sigma_deg: float = 0.5,
    coverage: float = 0.90,
) -> tuple:
    """
    Observed-range polygon derived from a smoothed presence-density raster.

    The threshold is derived from the presence records themselves: it is set to
    the density value below which (1 - coverage) of presences fall, so the
    polygon is guaranteed to contain `coverage` fraction of all records.

    Returns (polygon, actual_coverage) where actual_coverage may be slightly
    above the target due to cells shared by multiple presences.
    """
    from scipy.ndimage import gaussian_filter
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as sh_shape, MultiPoint
    from shapely.ops import unary_union

    height, width = shape
    resolution = float(tgt_lons[1] - tgt_lons[0])

    pts = presence_df[["lon", "lat"]].dropna()
    lon_step = tgt_lons[1] - tgt_lons[0]
    lat_step = tgt_lats[1] - tgt_lats[0]   # negative (north→south)

    c_idx = np.clip(np.round((pts["lon"].values - tgt_lons[0]) / lon_step).astype(int), 0, width  - 1)
    r_idx = np.clip(np.round((pts["lat"].values - tgt_lats[0]) / lat_step).astype(int), 0, height - 1)

    density = np.zeros(shape, dtype=np.float32)
    np.add.at(density, (r_idx, c_idx), 1)

    sigma_cells = sigma_deg / abs(resolution)
    smoothed = gaussian_filter(density, sigma=sigma_cells)

    if valid_mask is not None:
        smoothed[~valid_mask] = 0.0

    # Threshold = the smoothed-density value at the (1-coverage) percentile of
    # presence locations — guarantees `coverage` fraction of records are inside
    pres_vals = smoothed[r_idx, c_idx]
    nonzero = pres_vals[pres_vals > 0]
    threshold = (
        float(np.percentile(nonzero, (1 - coverage) * 100))
        if len(nonzero)
        else smoothed.max() * 0.02
    )
    actual_coverage = float((pres_vals >= threshold).mean())

    binary = (smoothed >= threshold).astype(np.uint8)
    polys = [sh_shape(geom) for geom, val in rio_shapes(binary, transform=transform) if val == 1]

    if not polys:
        mp = MultiPoint(list(zip(pts["lon"].values, pts["lat"].values)))
        return mp.convex_hull.buffer(abs(resolution) * 5), actual_coverage

    return unary_union(polys).simplify(abs(resolution)), actual_coverage


def save_range_vector(range_poly, crs: CRS, path: Path) -> None:
    """Save the observed-range polygon as a GeoPackage (.gpkg)."""
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"layer": ["observed_range"]},
        geometry=[range_poly],
        crs=crs.to_epsg() or 4326,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")
    log.info(f"Range vector saved → {path}")


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
    range_poly=None,
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

    # ── Actual observed range (buffered union of occurrence points) ───────────
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    if range_poly is None:
        from shapely.geometry import MultiPoint
        pts = presence_df[["lon", "lat"]].dropna()
        range_poly = MultiPoint(list(zip(pts["lon"].values, pts["lat"].values))).convex_hull.buffer(1.5)


    def plot_polygon(poly, ax, **kwargs):
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MplPath
        import numpy as np
        def ring_to_path(ring):
            coords = np.array(ring.coords)
            codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(coords) - 2) + [MplPath.CLOSEPOLY]
            return coords, codes
        all_verts, all_codes = [], []
        geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        for geom in geoms:
            v, c = ring_to_path(geom.exterior)
            all_verts.append(v); all_codes.extend(c)
            for interior in geom.interiors:
                v, c = ring_to_path(interior)
                all_verts.append(v); all_codes.extend(c)
        path = MplPath(np.vstack(all_verts), all_codes)
        ax.add_patch(PathPatch(path, **kwargs))

    plot_polygon(
        range_poly, ax,
        facecolor="none", edgecolor="#00b4d8",
        linewidth=1.4, linestyle="solid", zorder=3,
    )
    ax.plot([], [], color="#00b4d8", linewidth=1.4, linestyle="solid",
            label="Observed range")

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
    p.add_argument(
        "--range-coverage", type=float, default=1.0,
        help="Fraction of presence records the range polygon must contain (default: 1.0)",
    )
    p.add_argument(
        "--validate", action="store_true",
        help="Run spatial block CV + Boyce index after training",
    )
    p.add_argument(
        "--block-size", type=float, default=10.0,
        help="Spatial block size in degrees for CV (default: 10.0)",
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

    # 4 ── Predictor stack + collinearity filter ──────────────────────────────
    feature_names = list(climate_grids.keys())
    predictor_stack = np.stack(list(climate_grids.values()), axis=0)  # (n_features, H, W)

    valid_mask = np.all(np.isfinite(predictor_stack), axis=0)
    log.info(f"Valid pixels: {valid_mask.sum():,} / {valid_mask.size:,}")

    log.info("Checking predictor collinearity...")
    predictor_stack, feature_names = filter_correlated_predictors(
        predictor_stack, feature_names, valid_mask, rng=rng
    )

    # 5 ── Clip presences to valid pixels + spatial thinning ──────────────────
    lon_step = tgt_lons[1] - tgt_lons[0]
    lat_step = tgt_lats[1] - tgt_lats[0]
    pres_col_idx = np.clip(np.round((presence_df["lon"].values - tgt_lons[0]) / lon_step).astype(int), 0, width - 1)
    pres_row_idx = np.clip(np.round((presence_df["lat"].values - tgt_lats[0]) / lat_step).astype(int), 0, height - 1)
    in_valid = valid_mask[pres_row_idx, pres_col_idx]
    presence_df = presence_df[in_valid].reset_index(drop=True)
    log.info(f"{len(presence_df):,} presence points within valid BioClim coverage")

    if len(presence_df) < 10:
        sys.exit("Too few presence points within BioClim coverage (<10) — aborting.")

    # Thin to one record per grid cell — reduces sampling bias from over-surveyed sites
    n_before = len(presence_df)
    cell_ids = pres_row_idx[in_valid] * width + pres_col_idx[in_valid]
    _, first = np.unique(cell_ids, return_index=True)
    presence_df = presence_df.iloc[first].reset_index(drop=True)
    log.info(f"Spatial thinning: {n_before:,} → {len(presence_df):,} records (1 per {args.resolution}° cell)")

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

    # 8 ── Train ensemble (Random Forest + Extra Trees) ───────────────────────
    from sklearn.ensemble import ExtraTreesClassifier
    log.info("Training Random Forest...")
    rf = train_model(X, y, feature_names)
    log.info("Training Extra Trees...")
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=0,
    )
    et.fit(X, y)
    log.info(f"Extra Trees training AUC: {roc_auc_score(y, et.predict_proba(X)[:, 1]):.3f}")
    model = _Ensemble([rf, et])

    # 8b ── Spatial block cross-validation ────────────────────────────────────
    if args.validate:
        log.info("Running spatial block cross-validation (this may take a few minutes)...")
        cv = spatial_block_cv(
            presence_df, absence_df, predictor_stack,
            tgt_lons, tgt_lats, feature_names,
            block_size=args.block_size,
        )
        if cv:
            log.info(
                f"Spatial CV results ({cv['n_folds']} folds, {args.block_size}° blocks):\n"
                f"  AUC         = {cv['auc_mean']:.3f} ± {cv['auc_std']:.3f}\n"
                f"  TSS         = {cv['tss_mean']:.3f} ± {cv['tss_std']:.3f}\n"
                f"  Sensitivity = {cv['sensitivity_mean']:.3f}  "
                f"(omission rate = {1 - cv['sensitivity_mean']:.3f})\n"
                f"  Specificity = {cv['specificity_mean']:.3f}  "
                f"(commission rate = {1 - cv['specificity_mean']:.3f})"
            )

    # 9 ── Predict suitability ─────────────────────────────────────────────────
    log.info("Predicting suitability across full grid...")
    flat_X = predictor_stack[:, valid_mask].T     # (n_valid, n_features)
    prob   = model.predict_proba(flat_X)[:, 1]

    suitability = np.full(shape, np.nan, dtype=np.float32)
    suitability[valid_mask] = prob

    log.info(
        f"Suitability range: {np.nanmin(suitability):.3f} – {np.nanmax(suitability):.3f}"
    )

    # 9b ── Boyce index ────────────────────────────────────────────────────────
    if args.validate:
        bi = boyce_index(suitability, presence_df, tgt_lons, tgt_lats)
        log.info(
            f"Boyce index: {bi:.3f}  "
            f"({'excellent' if bi > 0.7 else 'acceptable' if bi > 0.5 else 'poor'})"
        )

    # 10 ── Save output ────────────────────────────────────────────────────────
    slug = args.species.replace("-", " ").replace(" ", "_").lower()

    log.info("Building observed-range polygon...")
    range_poly, range_coverage = build_range_polygon(
        presence_df, transform, shape, tgt_lons, tgt_lats,
        valid_mask=valid_mask, coverage=args.range_coverage,
    )
    log.info(f"Range polygon captures {range_coverage:.1%} of presence records")
    range_path = args.output_dir / f"{slug}_range.gpkg"
    save_range_vector(range_poly, TARGET_CRS, range_path)

    if args.preview:
        out_path = args.output_dir / f"{slug}_preview.png"
        save_preview_png(suitability, presence_df, extent, args.species, out_path, range_poly)
    else:
        out_path = args.output_dir / f"{slug}_suitability.tif"
        save_suitability_raster(suitability, transform, TARGET_CRS, out_path)


if __name__ == "__main__":
    main()
