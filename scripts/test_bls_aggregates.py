import requests

ids = [
    "PCU31--31--",
    "PCU311---311---",
    "PCU321---321---",
    "PCU327---327---",
    "PCU331---331---",
]

for sid in ids:
    payload = {
        "seriesid": [sid],
        "startyear": "2023",
        "endyear": "2023",
    }

    r = requests.post(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        json=payload,
        timeout=60,
    )

    print("\n", sid)
    print(r.json())