from pygbif import occurrences as occ
import pandas as pd
import os

checkpoint_file = 'checkpoint.csv'
all_records = []
offset = 0

# Load existing checkpoint if exists
if os.path.exists(checkpoint_file):
    existing = pd.read_csv(checkpoint_file)
    all_records = existing.to_dict('records')
    offset = len(all_records)
    print(f"Resuming from {offset} records...")

while True:
    try:
        batch = occ.search(
            scientificName='Vanessa cardui',
            country=['US', 'CA', 'MX'],
            hasCoordinate=True,
            limit=300,
            offset=offset
        )
        
        if not batch['results']:
            break
        
        all_records.extend(batch['results'])
        offset += 300
        
        # Save checkpoint every 1000 records
        if offset % 1000 == 0:
            pd.DataFrame(all_records).to_csv(checkpoint_file, index=False)
            print(f"Checkpoint saved: {len(all_records)} records")
        else:
            print(f"Downloaded {len(all_records)} records...")
            
    except Exception as e:
        print(f"Error at offset {offset}: {e}")
        pd.DataFrame(all_records).to_csv(checkpoint_file, index=False)
        break

# Final save
df = pd.DataFrame(all_records)
df.to_csv('vanessa_cardui_gbif.csv', index=False)
print(f"Complete: {len(df)} records")