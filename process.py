import os
from qgis.core import (
    QgsVectorLayer,
    QgsProject,
    QgsVectorLayerTemporalProperties,
    QgsDataSourceUri,
    QgsWkbTypes
)

DB_HOST    = "10.11.11.58"
DB_PORT    = "5432"
DB_NAME    = "lepidoptera_data"
DB_USER    = "lepidoptera_demo"
DB_PASS    = "demo"
DATE_FIELD = "observed_date"


def make_uri(layer_name, sql):
    uri = QgsDataSourceUri()
    uri.setConnection(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS)
    uri.setDataSource("", f"({sql})", "geom", "", "id")
    uri.setSrid("4326")
    uri.setWkbType(QgsWkbTypes.Point)
    return uri


def get_loaded_names():
    return {layer.name() for layer in QgsProject.instance().mapLayers().values()}


def add_temporal_layer(uri, name):
    """Load a PostGIS layer, set temporal properties, add to project."""
    loaded = get_loaded_names()

    if name in loaded:
        print(f"  → {name:40s} already loaded, skipping")
        return None

    layer = QgsVectorLayer(uri.uri(False), name, "postgres")

    if not layer.isValid():
        print(f"  ✗ {name:40s} failed to load")
        return None

    QgsProject.instance().addMapLayer(layer)

    tp = layer.temporalProperties()
    tp.setIsActive(True)
    tp.setMode(QgsVectorLayerTemporalProperties.ModeFeatureDateTimeInstantFromField)
    tp.setStartField(DATE_FIELD)

    return layer


# ---------------------------------------------------------------------------
# Lepidoptera occurrences
# ---------------------------------------------------------------------------
print("=" * 60)
print("SCANNING DATABASE FOR NEW SPECIES")
print("=" * 60)

uri = QgsDataSourceUri()
uri.setConnection(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS)
uri.setDataSource("public", "species", None)
species_layer = QgsVectorLayer(uri.uri(), "species_list", "postgres")

if not species_layer.isValid():
    print("✗ Could not connect to database")
else:
    added   = 0
    skipped = 0
    failed  = 0

    for feature in species_layer.getFeatures():
        species_id   = feature["id"]
        species_name = feature["scientific_name"]
        common_name  = feature["common_name"] or species_name

        sql = f"""
            SELECT id, observed_date, source, data_tier, geom
            FROM   lepidoptera_occurrences
            WHERE  species_id = {species_id}
              AND  data_tier  = 1
        """

        layer = add_temporal_layer(make_uri(common_name, sql), common_name)

        if layer is None:
            if common_name in get_loaded_names():
                skipped += 1
            else:
                failed += 1
        else:
            print(f"  ✓ {common_name:40s} {layer.featureCount():>7,} observations")
            added += 1

    print(f"\n  Butterflies — added: {added} | skipped: {skipped} | failed: {failed}")

# ---------------------------------------------------------------------------
# Host plant occurrences
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SCANNING DATABASE FOR NEW HOST PLANTS")
print("=" * 60)

host_sql = """
    SELECT DISTINCT scientific_name
    FROM   host_plant_occurrences
    ORDER  BY scientific_name
"""

uri = QgsDataSourceUri()
uri.setConnection(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS)
uri.setDataSource("", f"({host_sql})", None, "", "scientific_name")
host_list = QgsVectorLayer(uri.uri(False), "host_list", "postgres")

added   = 0
skipped = 0
failed  = 0

for feature in host_list.getFeatures():
    plant_name = feature["scientific_name"]

    sql = f"""
        SELECT id, scientific_name, observed_date, geom
        FROM   host_plant_occurrences
        WHERE  scientific_name = '{plant_name}'
    """

    layer = add_temporal_layer(make_uri(plant_name, sql), plant_name)

    if layer is None:
        if plant_name in get_loaded_names():
            skipped += 1
        else:
            failed += 1
    else:
        print(f"  ✓ {plant_name:40s} {layer.featureCount():>7,} observations")
        added += 1

print(f"\n  Host plants — added: {added} | skipped: {skipped} | failed: {failed}")
print("\n" + "=" * 60)
print(f"Total layers in project: {len(QgsProject.instance().mapLayers())}")
print("=" * 60)


# Quick Run
# exec(open('C:/Users/sirpl/Projects/species-distribution-model/process.py').read())