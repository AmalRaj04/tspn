import re
import pdfplumber
import pandas as pd

PDF_PATH = "data/raw/tariff_events/2018-13248.pdf"   # adjust path if needed
OUTPUT_CSV = "data/raw/tariff_events/us_301_list1_2018.csv"

# HTS pattern used in Annex A
HTS_PATTERN = r"\b\d{4}\.\d{2}\.\d{2}\b"

codes = set()

with pdfplumber.open(PDF_PATH) as pdf:

    # Annex A HTS codes are on pages 5-9 of the PDF
    # (pdfplumber uses 0-based indexing)
    for page_num in range(4, 9):

        page = pdf.pages[page_num]
        text = page.extract_text()

        if not text:
            continue

        matches = re.findall(HTS_PATTERN, text)

        for code in matches:
            codes.add(code.replace(".", ""))

codes = sorted(codes)

print(f"Found {len(codes)} HTS codes")

df = pd.DataFrame({
    "hts_code": codes,
    "delta_tariff_pct": 25.0
})

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {OUTPUT_CSV}")