import pandas as pd

og = pd.read_csv("/mnt/c/Users/sirpl/Projects/species-distribution-model/occurrence-data/danaus-plexippus/danaus-plexippus-gbif.csv")
new = pd.read_csv("/mnt/c/Users/sirpl/Projects/species-distribution-model/occurrence-data/danaus-plexippus/danaus-plexippus_gbif.csv")

print(f'OG length: {len(og)}')
print(f'new length: {len(new)}')