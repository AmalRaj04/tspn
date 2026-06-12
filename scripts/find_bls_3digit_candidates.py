import requests
import json

tests = [
    "PCU311111311111",
    "PCU311119311119",
    "PCU311211311211",
    "PCU311221311221",
    "PCU312111312111",
    "PCU313110313110",
    "PCU321113321113",
    "PCU327310327310",
    "PCU331110331110",
]

for sid in tests:
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

    rows = len(
        body.get("Results", {})
            .get("series", [{}])[0]
            .get("data", [])
    )

    print(sid, rows)