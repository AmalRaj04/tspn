import requests

urls = [
    "https://download.bls.gov/pub/time.series/pc/",
    "https://download.bls.gov/pub/time.series/pc/pc.series",
]

for url in urls:
    print("=" * 80)
    print(url)
    print("=" * 80)

    try:
        r = requests.get(url, timeout=60)

        print("STATUS:", r.status_code)
        print("CONTENT-TYPE:", r.headers.get("Content-Type"))

        print("\nFIRST 50 LINES:\n")

        for line in r.text.splitlines()[:50]:
            print(line)

    except Exception as e:
        print("ERROR:", e)