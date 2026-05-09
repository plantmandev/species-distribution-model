-- =============================================================================
-- Lepidoptera GIS Database
-- =============================================================================
-- Run with:
--   psql -d lepidoptera_data \
--     -v APP_PASSWORD="yourpassword" \
--     -v DEMO_PASSWORD="yourpassword" \
--     -f setup_database.sql
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------------
-- ROLES
-- ---------------------------------------------------------------------------
CREATE USER lepidoptera_app  WITH PASSWORD :'APP_PASSWORD';
CREATE USER lepidoptera_demo WITH PASSWORD :'DEMO_PASSWORD';

GRANT CONNECT ON DATABASE lepidoptera_data TO lepidoptera_app;
GRANT CONNECT ON DATABASE lepidoptera_data TO lepidoptera_demo;
GRANT USAGE   ON SCHEMA public TO lepidoptera_app;
GRANT USAGE   ON SCHEMA public TO lepidoptera_demo;

-- =============================================================================
-- SPECIES
-- Thin record. GBIF + Wikipedia APIs provide content at runtime.
-- occurrence_extent and host_plant_extent are precomputed bounding boxes —
-- refreshed after each ingest. Used to auto-fit the map viewport on species
-- selection without running a live aggregate over millions of points.
-- =============================================================================
CREATE TABLE species (
    id                   SERIAL PRIMARY KEY,
    scientific_name      TEXT    NOT NULL UNIQUE,
    common_name          TEXT,
    family               TEXT,
    gbif_taxon_key       INTEGER UNIQUE,
    inat_taxon_id        INTEGER UNIQUE,
    occurrence_extent    GEOMETRY(Polygon, 4326),   -- bbox of all lepidoptera occurrences
    host_plant_extent    GEOMETRY(Polygon, 4326),   -- bbox of all associated host plant occurrences
    created_at           TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- LEPIDOPTERA OCCURRENCES
-- =============================================================================
CREATE TABLE lepidoptera_occurrences (
    id                   SERIAL PRIMARY KEY,
    species_id           INTEGER  NOT NULL REFERENCES species(id) ON DELETE CASCADE,
    observed_date        DATE,
    source               TEXT     NOT NULL,   -- GBIF | iNaturalist
    source_id            TEXT     NOT NULL,   -- provider's own record ID
    inat_observation_id  TEXT,               -- set for all iNat-origin records
    data_tier            SMALLINT NOT NULL DEFAULT 1 CHECK (data_tier IN (1, 2)),
    geom                 GEOMETRY(Point, 4326) NOT NULL,
    created_at           TIMESTAMP DEFAULT NOW(),

    UNIQUE (source, source_id)
);

CREATE UNIQUE INDEX idx_lep_inat_dedup
    ON lepidoptera_occurrences (inat_observation_id)
    WHERE inat_observation_id IS NOT NULL;

-- =============================================================================
-- HOST PLANT OCCURRENCES
-- =============================================================================
CREATE TABLE host_plant_occurrences (
    id               SERIAL PRIMARY KEY,
    scientific_name  TEXT     NOT NULL,
    observed_date    DATE,
    source           TEXT     NOT NULL,
    source_id        TEXT     NOT NULL,
    geom             GEOMETRY(Point, 4326) NOT NULL,
    created_at       TIMESTAMP DEFAULT NOW(),

    UNIQUE (source, source_id)
);

-- =============================================================================
-- SPECIES ↔ HOST PLANT LOOKUP
-- Cross-reference query:
--   SELECT h.*
--   FROM   host_plant_occurrences h
--   JOIN   species_host_plants shp ON shp.plant_scientific_name = h.scientific_name
--   WHERE  shp.species_id = $1;
-- =============================================================================
CREATE TABLE species_host_plants (
    id                    SERIAL PRIMARY KEY,
    species_id            INTEGER NOT NULL REFERENCES species(id) ON DELETE CASCADE,
    plant_scientific_name TEXT    NOT NULL,
    UNIQUE (species_id, plant_scientific_name)
);



-- =============================================================================
-- INDEXES
-- =============================================================================
CREATE INDEX idx_species_name_trgm  ON species                 USING GIN (scientific_name gin_trgm_ops);
CREATE INDEX idx_species_occ_ext    ON species                 USING GIST (occurrence_extent);
CREATE INDEX idx_species_hp_ext     ON species                 USING GIST (host_plant_extent);

CREATE INDEX idx_lep_geom           ON lepidoptera_occurrences USING GIST (geom);
CREATE INDEX idx_lep_species        ON lepidoptera_occurrences (species_id);
CREATE INDEX idx_lep_tier           ON lepidoptera_occurrences (data_tier);
CREATE INDEX idx_lep_date           ON lepidoptera_occurrences (observed_date);

CREATE INDEX idx_host_geom          ON host_plant_occurrences  USING GIST (geom);
CREATE INDEX idx_host_name          ON host_plant_occurrences  (scientific_name);
CREATE INDEX idx_host_date          ON host_plant_occurrences  (observed_date);

CREATE INDEX idx_sxhp_species       ON species_host_plants     (species_id);
CREATE INDEX idx_sxhp_plant         ON species_host_plants     (plant_scientific_name);

-- =============================================================================
-- FUNCTION: refresh_species_extents(species_id)
-- Call this after every ingest to keep the bbox columns current.
-- =============================================================================
CREATE OR REPLACE FUNCTION refresh_species_extents(p_species_id INTEGER)
RETURNS VOID AS $$
BEGIN
    UPDATE species
    SET
        -- Bounding box of all butterfly occurrence points
        occurrence_extent = (
            SELECT ST_Envelope(ST_Collect(geom))
            FROM   lepidoptera_occurrences
            WHERE  species_id = p_species_id
        ),
        -- Bounding box of all host plant occurrence points for this butterfly
        host_plant_extent = (
            SELECT ST_Envelope(ST_Collect(h.geom))
            FROM   host_plant_occurrences h
            JOIN   species_host_plants shp
                   ON shp.plant_scientific_name = h.scientific_name
            WHERE  shp.species_id = p_species_id
        )
    WHERE id = p_species_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_species_extents IS
    'Recomputes occurrence and host plant bounding boxes for a species after ingest. '
    'Call as: SELECT refresh_species_extents($species_id);';

-- =============================================================================
-- SDM OUTPUTS
-- Tracks which species have completed SDM outputs (suitability raster +
-- observed-range vector) uploaded to the plantmandev public directory.
-- Populated by update.py --upload-sdm; queried by the web front-end.
-- =============================================================================
CREATE TABLE sdm_outputs (
    id               SERIAL PRIMARY KEY,
    species_id       INTEGER NOT NULL REFERENCES species(id) ON DELETE CASCADE,
    suitability_data BYTEA,         -- raw GeoTIFF bytes
    range_data       BYTEA,         -- GeoJSON bytes (converted from .gpkg at upload time)
    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'completed')),
    created_at       TIMESTAMP DEFAULT NOW(),
    completed_at     TIMESTAMP,
    UNIQUE (species_id)
);

CREATE INDEX idx_sdm_species ON sdm_outputs (species_id);
CREATE INDEX idx_sdm_status  ON sdm_outputs (status);

-- =============================================================================
-- GRANTS
-- =============================================================================
GRANT SELECT                     ON ALL TABLES    IN SCHEMA public TO lepidoptera_demo;
GRANT INSERT, UPDATE, DELETE     ON ALL TABLES    IN SCHEMA public TO lepidoptera_app;
GRANT USAGE, SELECT              ON ALL SEQUENCES IN SCHEMA public TO lepidoptera_app;
GRANT EXECUTE ON FUNCTION refresh_species_extents(INTEGER) TO lepidoptera_app;
GRANT INSERT, UPDATE             ON sdm_outputs TO lepidoptera_app;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO lepidoptera_demo;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT INSERT, UPDATE, DELETE ON TABLES TO lepidoptera_app;