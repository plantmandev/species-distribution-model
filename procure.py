from pygbif import occurrences as occ
import pandas as pd

all_records = []
offset = 0

while True:
    batch = occ.search(
        scientificName='Vanessa cardui',
        country= ['US', 'CA', 'MX'], # Refine search to North America
        hasCoordinate=True,
        limit=300,
        offset=offset
    )
    
    if not batch['results']:
        break
    
    all_records.extend(batch['results'])
    offset += 300 # GBIF API request limit
    print(f"Downloaded {len(all_records)} records...")

df = pd.DataFrame(all_records)
df.to_csv('vanessa-cardui-gbif.csv', index=False)
print(f"Total records: {len(df)}")