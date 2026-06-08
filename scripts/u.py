from pyxlsb import open_workbook

with open_workbook("data/raw/wiod/WIOT2014_Nov16_ROW.xlsb") as wb:
    with wb.get_sheet("2014") as sheet:

        found = []

        for i, row in enumerate(sheet.rows()):
            vals = [c.v for c in row[:4]]

            if "U" in [str(v) for v in vals]:
                found.append((i, vals))

        print("Rows containing U:")
        for r in found[:20]:
            print(r)

        print("\nTotal:", len(found))