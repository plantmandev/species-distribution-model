import pandas as pd

og = pd.read_csv("../occurrence-data/vanessa-atalanta/vanessa-atalanta-gbif.csv")
# new = pd.read_csv("../occurrence-data/vanessa-atalanta/vanessa-atalanta_gbif.csv")

print(f'OG length: {len(og)}')
# print(f'new length: {len(new)}')