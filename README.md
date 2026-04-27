# Lepidoptera Species Distribution Model

A full pipeline for procuring, storing, and modeling the habitat suitability and observed ranges of North American butterflies. Occurrence data from GBIF is stored in a PostGIS database and fed into an ensemble Random Forest / Extra Trees SDM trained on WorldClim 2.1 BioClim climate variables.

## Species

| Common name | Scientific name | IUCN Status |
|---|---|---|
| Painted lady | *Vanessa cardui* | Least Concern |
| Monarch butterfly | *Danaus plexippus* | Least Concern |
| Tiger swallowtail | *Papilio glaucus* | N/A |
| Cabbage white | *Pieris rapae* | N/A |
| Spring azure | *Celastrina ladon* | N/A |
| American copper | *Lycaena phlaeas* | N/A |
| American lady | *Vanessa virginiensis* | Least Concern |
| Black swallowtail | *Papilio polyxenes* | Least Concern |
| Cloudless sulfur | *Phoebis sennae* | Least Concern |
| Great spangled fritillary | *Speyeria cybele* | Least Concern |
| Variegated fritillary | *Euptoieta claudia* | N/A |
| Red admiral | *Vanessa atalanta* | Least Concern |
| Zebra swallowtail | *Eurytides marcellus* | N/A |
| Common buckeye | *Junonia coenia* | Least Concern |
| Colorado hairstreak | *Hypaurotis crysalus* | N/A |
| Regal fritillary | *Speyeria idalia* | Vulnerable |
| Dakota skipper | *Hesperia dacotae* | Endangered |
| Ottoe skipper | *Hesperia ottoe* | Endangered |

---

## Pipeline overview

```
procure.py          download_bioclim.py
    │                       │
    ▼                       ▼
occurrence-data/       bioclim-data/
(GeoJSON per species)  (8 × WorldClim TIFs)
    │                       │
    ▼                       │
ingest.py                   │
    │                       │
    ▼                       ▼
PostGIS (Neon)  ──────► sdm.py
                              │
                              ▼
                    sdm-output/
                    ├── {species}_suitability.tif
                    └── {species}_range.gpkg
```

---

## Scripts

### `download_bioclim.py` — fetch climate predictors

Downloads WorldClim 2.1 BioClim (2.5 arcmin, global) and extracts the 8 variables used by `sdm.py` into `bioclim-data/`.

```bash
python download_bioclim.py
```

Variables downloaded: `bio01` Annual Mean Temp · `bio04` Temp Seasonality · `bio05` Max Temp Warmest Month · `bio06` Min Temp Coldest Month · `bio12` Annual Precipitation · `bio15` Precip Seasonality · `bio18` Precip Warmest Quarter · `bio19` Precip Coldest Quarter

---

### `procure.py` — download GBIF occurrences

Downloads research-grade occurrence records from GBIF for all species listed in `occurrence-data/species-metadata.csv`. Resume-aware: only fetches years not already present in the GeoJSON. Automatically splits years with >80k records into monthly chunks to stay within the GBIF API offset limit.

```bash
python procure.py                          # download all pending species
python procure.py --year 2010              # start from 2010 (default: 2015)
python procure.py --countries US CA MX     # restrict to specific countries
python procure.py --workers 5              # parallel downloads (default: 3)
python procure.py --status                 # show metadata summary
python procure.py --force                  # re-download all species
```

Output: `occurrence-data/{species-name}-gbif.geojson` + updated `species-metadata.csv`

---

### `ingest.py` — load occurrences into PostGIS

Reads species-metadata.csv, loads each GeoJSON into the PostGIS database, links host plants, and marks the row as `ingested`. Uses GBIF occurrence IDs as stable deduplication keys so re-ingesting is safe.

```bash
python ingest.py                           # ingest all pending species
python ingest.py --dry-run                 # preview without writing
python ingest.py --species "Vanessa cardui"
python ingest.py --no-delete               # keep GeoJSON files after ingest
python ingest.py --populate-hosts resource.csv   # load NHM HOSTS data
```

Species flagged with `#drop` in the notes column are deleted from the database and removed from metadata before ingestion begins.

---

### `sdm.py` — species distribution model

Trains an ensemble SDM (Random Forest + Extra Trees) on BioClim climate variables and occurrence data from PostGIS (with GeoJSON fallback). Outputs a habitat suitability raster and an observed-range polygon.

```bash
python sdm.py "danaus plexippus"
python sdm.py "vanessa cardui" --resolution 0.05 --n-absences 10000
python sdm.py "danaus plexippus" --preview          # save PNG instead of TIFF
python sdm.py "vanessa cardui" --validate           # run spatial block CV + Boyce index
python sdm.py "danaus plexippus" --target-group     # use other Lepidoptera as background
```

**Key options**

| Flag | Default | Description |
|---|---|---|
| `--resolution` | `0.1` | Output cell size in degrees (~11 km) |
| `--n-absences` | `5000` | Number of pseudo-absence background points |
| `--range-coverage` | `1.0` | Fraction of presence records inside the range polygon |
| `--validate` | off | Run spatial block CV, TSS, sensitivity, specificity, and Boyce index |
| `--block-size` | `10.0` | Spatial block size in degrees for CV |
| `--extent W S E N` | BioClim extent | Clip study area |
| `--target-group` | off | Sample background from other Lepidoptera occurrences (requires DB) |

**Outputs** (written to `sdm-output/`)

| File | Contents |
|---|---|
| `{species}_suitability.tif` | Habitat suitability raster, 0–1 (Float32 GeoTIFF) |
| `{species}_range.gpkg` | Observed range polygon (GeoPackage, EPSG:4326) |

**Methodology**

1. Occurrence points loaded from PostGIS; falls back to `occurrence-data/{species}-gbif.geojson`
2. Spatially thinned to one record per grid cell to reduce sampling bias
3. BioClim predictors checked for collinearity (Pearson |r| > 0.85); correlated predictors dropped
4. Pseudo-absences sampled from valid land pixels (or target-group Lepidoptera if `--target-group`)
5. Ensemble trained: Random Forest (200 trees) + Extra Trees (200 trees), probabilities averaged
6. Suitability predicted across all valid land pixels
7. Range polygon derived from a smoothed presence-density raster: occurrences binned onto the model grid → Gaussian smoothed (σ ≈ 55 km) → ocean/nodata masked → thresholded to capture `--range-coverage` fraction of records → vectorised

**Validation** (`--validate`)

- **Spatial block cross-validation**: leave-one-block-out at `--block-size` degree blocks; reports AUC, TSS, sensitivity (omission rate), specificity (commission rate)
- **Boyce index**: Spearman correlation of predicted-to-expected ratios across suitability bins; +1 = presences cluster in high-suitability cells

---

### `update.py` — sync local database to Neon

Snapshots the current Neon `main` branch as `save-state-YYYYMMDD`, then restores a fresh dump from the local PostgreSQL instance to Neon `main`.

```bash
python update.py
```

Requires `NEON_CONN` (Neon connection string) and `LOCAL_CONN` (defaults to `postgresql://postgres@localhost:5432/lepidoptera_data`).

---

### `setup_database.sql` — initialise the PostGIS schema

```bash
psql -d lepidoptera_data \
  -v APP_PASSWORD="yourpassword" \
  -v DEMO_PASSWORD="yourpassword" \
  -f setup_database.sql
```

Creates the `lepidoptera_occurrences`, `species`, `host_plant_occurrences`, and `species_host_plants` tables with PostGIS geometry columns and appropriate roles.

---

## Setup

### 1. Database

```bash
createdb lepidoptera_data
psql -d lepidoptera_data -v APP_PASSWORD="..." -v DEMO_PASSWORD="..." -f setup_database.sql
```

### 2. Environment variables

```bash
# PostGIS (local)
export DB_HOST=localhost
export DB_USER=lepidoptera_app
export DB_APP_PASSWORD=yourpassword
export DB_NAME=lepidoptera_data

# Neon (remote sync — update.py only)
export NEON_CONN="postgresql://..."
```

Or use a `.env` file — `sdm.py` loads it automatically via `python-dotenv`.

### 3. Climate data

```bash
python download_bioclim.py   # ~658 MB, one-time download
```

### 4. Occurrence data

Edit `occurrence-data/species-metadata.csv` to list your target species, then:

```bash
python procure.py
python ingest.py
```

### 5. Run the SDM

```bash
python sdm.py "danaus plexippus"
python sdm.py "vanessa cardui" --validate
```

---

## Data sources

| Data | Source | Notes |
|---|---|---|
| Occurrence data | [GBIF](https://www.gbif.org/) via [pygbif](https://pypi.org/project/pygbif/) | Research-grade, presence-only |
| Climate predictors | [WorldClim 2.1 BioClim](https://worldclim.org/data/bioclim.html) | 2.5 arcmin (~5 km), global |
| Land cover | [NALCMS 2020](https://www.cec.org/north-american-environmental-atlas/land-cover-30m-2020/) (CEC) | 30 m, North America only — not yet wired into SDM |
| Host plant associations | [NHM HOSTS database](https://data.nhm.ac.uk/dataset/hosts) | Loaded via `ingest.py --populate-hosts` |

---

## Constraints

The primary data source is research-grade occurrence observations from GBIF — presence-only data points that suffer from **sample selection bias** more strongly than presence-absence data (Elith et al., 2011). Sampling effort is geographically uneven and observer-concentrated, which means raw occurrence density reflects where people survey as much as where the species lives.

**Spatial thinning** (one record per grid cell) and **target-group background sampling** are applied to mitigate this, but the fundamental limitation remains: **this model estimates relative habitat suitability, not absolute probability of presence.**

---

## References

- Elith, J. et al. (2011). [A statistical explanation of MaxEnt for ecologists](https://onlinelibrary.wiley.com/doi/10.1111/j.1472-4642.2010.00725.x)
- Araújo, M. & New, M. (2007). [Ensemble forecasting of species distributions](https://www.sciencedirect.com/science/article/abs/pii/S016953470600303X)
- Fourcade, Y. et al. (2014). [Mapping species distributions with MaxEnt using a geographically biased sample](https://pmc.ncbi.nlm.nih.gov/articles/PMC4018261/)
- Phillips, S. et al. (2009). [Sample selection bias and presence-only distribution models](https://esajournals.onlinelibrary.wiley.com/doi/10.1890/07-2153.1)
- CEC (2024). North American Environmental Atlas — Land Cover 2020 30m. CCRS / USGS / CONABIO / CONAFOR / INEGI. Ed. 2.0.

---

## Notes

**1/31/25** — Encountered gaps in GBIF data richness. *Apodemia mormo* not found in the database despite significant iNaturalist observations. Consider adding iNaturalist fallback procurement if GBIF data is absent.

[View SQL schema diagram](https://dbdiagram.io/d/698d781bbd82f5fce27b99a9)

Host plant data scraped from the [NHM HOSTS database](https://data.nhm.ac.uk/dataset/hosts).
