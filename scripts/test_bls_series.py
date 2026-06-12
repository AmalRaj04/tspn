import requests
import json

ids = [
    "PCU311111311111",
    "PCU311119311119",
    "PCU327310327310",
    "PCU331110331110",
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

    body = r.json()

    print()
    print("=" * 80)
    print(sid)
    print("=" * 80)

    print(json.dumps(body, indent=2)[:2000])