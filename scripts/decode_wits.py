import json
import pandas as pd

with open("usa_2018.json") as f:
    data = json.load(f)

products = data["structure"]["dimensions"]["series"][3]["values"]

rows = []

for key, series in data["dataSets"][0]["series"].items():

    product_idx = int(key.split(":")[2])

    hs6 = products[product_idx]["id"]
    product_name = products[product_idx]["name"]

    obs = list(series["observations"].values())[0]

    rows.append({
        "hs6": hs6,
        "product": product_name,
        "raw_obs": obs
    })

df = pd.DataFrame(rows)

print(df.head())
print(df.shape)

df.to_csv("decoded_wits_sample.csv", index=False)