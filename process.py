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

for species_dir in base_path.iterdir():
    if not species_dir.is_dir():
        continue
    
    species_name = species_dir.name
    print(f"\nProcessing: {species_name}")
    
    gpkg_files = list(species_dir.glob("*.gpkg"))
    
    if not gpkg_files:
        print(f"  ⚠ No .gpkg files found, skipping")
        species_failed += 1
        continue
    
    gpkg_path = str(gpkg_files[0])
    print(f"  Found: {gpkg_files[0].name}")
    
    layer = QgsVectorLayer(gpkg_path, species_name, "ogr")
    
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