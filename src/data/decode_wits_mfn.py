import json
import pandas as pd
from pathlib import Path


def decode_wits(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    products = data["structure"]["dimensions"]["series"][3]["values"]

    rows = []

    for key, series in data["dataSets"][0]["series"].items():

        product_idx = int(key.split(":")[2])

        hs6 = products[product_idx]["id"]

        obs = list(series["observations"].values())[0]

        mfn_rate = float(obs[0])

        rows.append({
            "hs6": hs6,
            "mfn_rate": mfn_rate
        })

    return pd.DataFrame(rows)


def main():

    json_dir = Path("data/raw/wits/json")
    output_dir = Path("data/raw/wits")

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:

        try:

            df = decode_wits(json_file)

            parts = json_file.stem.split("_")

            reporter = parts[0].upper()
            year = parts[1]

            output_file = (
                output_dir /
                f"tariff_{reporter}_{year}.csv"
            )

            df.to_csv(output_file, index=False)

            print(
                f"✓ {json_file.name}"
                f" -> {output_file.name}"
                f" ({len(df):,} rows)"
            )

        except Exception as e:

            print(
                f"✗ Failed: {json_file.name}"
            )
            print(e)

    print("\nDone.")


if __name__ == "__main__":
    main()