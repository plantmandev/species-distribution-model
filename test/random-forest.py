import pandas as pd

og = pd.read_csv("danaus-plexippus-gbif.csv")
new = pd.read_csv("danaus-plexippus_gbif.csv")

print(f'OG length: {len(og)}')
print(f'new length: {len(new)}')