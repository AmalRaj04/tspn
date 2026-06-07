from pyxlsb import open_workbook

FILE = "data/raw/wiod/WIOT2014_Nov16_ROW.xlsb"

with open_workbook(FILE) as wb:
    with wb.get_sheet("2014") as sheet:

        rows = list(sheet.rows())

        sector_codes = []

        row2 = rows[2]

        for cell in row2[4:]:
            value = str(cell.v)

            if value == "None":
                continue

            if value not in sector_codes:
                sector_codes.append(value)

            if len(sector_codes) > 100:
                break

        print("\nSECTORS:")
        for i, s in enumerate(sector_codes, start=1):
            print(i, s)

        print("\nTOTAL =", len(sector_codes))