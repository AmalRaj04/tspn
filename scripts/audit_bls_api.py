import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

API_KEY = os.getenv("BLS_API_KEY")

if not API_KEY:
    raise RuntimeError("BLS_API_KEY not found")

TEST_SERIES = [
    "PCU311311",
    "PCU212212",
    "PCU211211",
    "PCU327327",
    "PCU331331",
    "PCU484484",
]

payload = {
    "seriesid": TEST_SERIES,
    "startyear": "2014",
    "endyear": "2023",
    "registrationkey": API_KEY,
}

print("=" * 80)
print("REQUEST")
print("=" * 80)
print(json.dumps(payload, indent=2))

r = requests.post(API_URL, json=payload, timeout=120)

print("\n")
print("=" * 80)
print("HTTP STATUS")
print("=" * 80)
print(r.status_code)

body = r.json()

print("\n")
print("=" * 80)
print("FULL RESPONSE")
print("=" * 80)
print(json.dumps(body, indent=2))

print("\n")
print("=" * 80)
print("SERIES SUMMARY")
print("=" * 80)

for s in body.get("Results", {}).get("series", []):
    print("\nSERIES:", s.get("seriesID"))
    print("ROWS:", len(s.get("data", [])))

    if s.get("data"):
        print("FIRST ROW:")
        print(json.dumps(s["data"][0], indent=2))