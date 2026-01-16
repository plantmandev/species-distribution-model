from pygbif import species as species
from pygbif import occurrences as occ

species_list = ['Vanessa cardui']

for idx in species_list:
    keys = [species.name_backbone(x)['usage']]