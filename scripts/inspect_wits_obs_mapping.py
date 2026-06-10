import json
import pprint

with open("usa_2018.json") as f:
    data = json.load(f)

obs_attrs = data["structure"]["attributes"]["observation"]

print("NUMBER OF OBS ATTRIBUTES:")
print(len(obs_attrs))
print()

for i, attr in enumerate(obs_attrs):
    print("=" * 60)
    print("INDEX:", i)
    print("ID:", attr["id"])
    print("NAME:", attr["name"])

    if "values" in attr:
        print("N_VALUES:", len(attr["values"]))

        if len(attr["values"]) > 0:
            print("FIRST 10 VALUES:")
            pprint.pprint(attr["values"][:10])

    print()
