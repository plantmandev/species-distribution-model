import pandas as pd
from pathlib import Path

df = pd.read_csv('occurrence-data/species-metadata.csv', dtype={'gbif_key': str})
df['category'] = df['category'].replace('butterfly', 'lepidoptera')
df.to_csv('occurrence-data/species-metadata.csv', index=False)
print(df[['species_name', 'category', 'common_name']].to_string())