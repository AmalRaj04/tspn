import re
import pdfplumber
import pandas as pd

PDF_PATH = "data/raw/tariff_events/2018-17709.pdf"
OUTPUT_CSV = "data/raw/tariff_events/us_301_list2_2018.csv"

HTS_PATTERN = r"\b\d{4}\.\d{2}\.\d{2}\b"

codes = set()

with pdfplumber.open(PDF_PATH) as pdf:

    # Annex A tariff codes are on pages 4-5
    for page_num in [3, 4]:   # 0-based indexing

        text = pdf.pages[page_num].extract_text()

        if not text:
            continue

        matches = re.findall(HTS_PATTERN, text)

        for code in matches:
            codes.add(code.replace(".", ""))

codes = sorted(codes)

df = pd.DataFrame({
    "hts_code": codes,
    "delta_tariff_pct": 25.0
})

df.to_csv(OUTPUT_CSV, index=False)

print(f"Rows: {len(df)}")
print(f"Saved: {OUTPUT_CSV}")