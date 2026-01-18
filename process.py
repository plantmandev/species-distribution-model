import processing
import os
from qgis.core import (
    QgsVectorLayer, 
    QgsProject,
    QgsVectorLayerTemporalProperties
)

csv_path = ("C:/Users/sirpl/Projects/species-distribution-model/vanessa-cardui-gbif.gpkg")
date_field = "eventDate"
output_path = "/"
species_name = "vanessa_cardui"

layer = QgsVectorLayer(csv_path, species_name, "ogr")

if not layer.isValid():
    print("Layer failed to load!")
    print(f"Error: {layer.error().message()}")
else:
    QgsProject.instance().addMapLayer(layer)
    print(f"Loaded {layer.featureCount()} observations")
    
    temporal_props = layer.temporalProperties()
    temporal_props.setIsActive(True)
    temporal_props.setMode(QgsVectorLayerTemporalProperties.ModeFeatureDateTimeInstantFromField)
    temporal_props.setStartField(date_field)
    
    print(f"Temporal properties configured using field: {date_field}")