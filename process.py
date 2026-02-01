import processing
import os
from pathlib import Path
from qgis.core import (
    QgsVectorLayer, 
    QgsProject,
    QgsVectorLayerTemporalProperties
)

base_path = Path("C:/Users/sirpl/Projects/species-distribution-model/occurrence-data")
date_field = "eventDate"

print("=" * 60)
print("PROCESSING ALL SPECIES")
print("=" * 60)

species_processed = 0
species_failed = 0

# Look for GeoJSON files directly in occurrence-data directory
geojson_files = list(base_path.glob("*.geojson"))

if not geojson_files:
    print(f"\n⚠ No .geojson files found in {base_path}")
    print("=" * 60)
else:
    for geojson_file in geojson_files:
        # Extract species name from filename (e.g., "ailurus-fulgens-gbif.geojson" -> "ailurus-fulgens")
        species_name = geojson_file.stem.replace('-gbif', '').replace('-', ' ').title()
        
        print(f"\nProcessing: {species_name}")
        print(f"  Found: {geojson_file.name}")
        
        layer = QgsVectorLayer(str(geojson_file), species_name, "ogr")
        
        if not layer.isValid():
            print(f"  ✗ Failed to load: {layer.error().message()}")
            species_failed += 1
            continue
        
        QgsProject.instance().addMapLayer(layer)
        
        temporal_props = layer.temporalProperties()
        temporal_props.setIsActive(True)
        temporal_props.setMode(QgsVectorLayerTemporalProperties.ModeFeatureDateTimeInstantFromField)
        temporal_props.setStartField(date_field)
        
        print(f"  ✓ Loaded {layer.featureCount()} observations")
        print(f"  ✓ Temporal properties configured")
        species_processed += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successfully processed: {species_processed} species")
    print(f"Failed: {species_failed} species")
    print(f"Total layers in project: {len(QgsProject.instance().mapLayers())}")

    # exec(open('C:/Users/sirpl/Projects/species-distribution-model/process.py').read())