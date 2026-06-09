import re
import pdfplumber
import pandas as pd

PDF_PATH = "data/raw/tariff_events/aluminumHTSlist final.pdf"
OUTPUT = "data/raw/tariff_events/us_232_aluminum_2018.csv"

HTS_PATTERN = r"\b\d{4}\.\d{2}\.\d{2}\b"

codes = set()

with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
        text = page.extract_text()

        if not text:
            continue

        matches = re.findall(HTS_PATTERN, text)

        for code in matches:
            codes.add(code.replace(".", ""))

codes = sorted(codes)

df = pd.DataFrame({
    "hts_code": codes,
    "product_description": "",
    "delta_tariff_pct": 10.0
})

df.to_csv(OUTPUT, index=False)

print("Rows:", len(df))
print("Saved:", OUTPUT)