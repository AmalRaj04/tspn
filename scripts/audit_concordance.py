import json


import random
with open(
    "data/processed/concordance/hs6_to_wiod56_weights.json"
) as f:
    mapping = json.load(f)

sector_count = {}

for hs6, weights in mapping.items():
    for sector in weights:
        sector_count[sector] = sector_count.get(sector, 0) + 1

for sector, count in sorted(
    sector_count.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(sector, count)


with open(

    "data/processed/concordance/hs6_to_wiod56_weights.json"

) as b:
    mapping = json.load(b)

for hs6 in random.sample(

    list(mapping.keys()),

    20

):

    print(hs6, mapping[hs6])