import pandas as pd
import geopandas as gpd

df = pd.read_csv("vanessa-cardui.csv") # Create dataframe for data manipulation
cleaned_df = df.drop(['id', 'uuid', 'time_observed_at', 'time_zone', 'user_id', 'user_login', 'user_name', 'created_at', 'quality_grade', 'url', 'tag_list', 'num_identification_agreements', 'captive_cultivated', 'oauth_application_id', 'place_guess', 'positional_accuracy', 'private_place_guess', 'private_latitude', 'private_longitude', 'geoprivacy', 'taxon_geoprivacy', 'coordinates_obscured', 'positioning_device', 'scientific_name', 'common_name'], axis = 1 ) # Only leave out longitude, latitude and observation date (observed_on)

cleaned_df['observed_on'] = pd.to_datetime(df['observed_on'])

gdf = gpd.GeoDataFrame(cleaned_df, geometry = gpd.points_from_xy(cleaned_df. longitude, cleaned_df.latitude), crs="EPSG:4326")

gdf.to_file('vanessa-cardui-cleaned.geojson', driver='GeoJSON')