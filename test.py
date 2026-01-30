import geopandas as gpd

species = 'vanessa-atalanta'

gdf = gpd.read_file(f'occurrence-data/{species}/{species}-gbif.gpkg')
gdf.to_file(f'{species}.geojson', driver='GeoJSON')