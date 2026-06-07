from pyxlsb import open_workbook

file_path = "data/raw/wiod/WIOT2014_Nov16_ROW.xlsb"

with open_workbook(file_path) as wb:
    with wb.get_sheet("2014") as sheet:

        rows = list(sheet.rows())

        sector_headers = []

        col = 4

        while True:
            value = rows[2][col].v

            if value is None:
                break

            sector_headers.append(value)
            col += 1

        print("\nSECTORS")
        print("Count:", len(sector_headers))
        print(sector_headers)

        countries = []

        col = 4

        while True:
            value = rows[4][col].v

            if value is None:
                break

            countries.append(value)
            col += 1

        print("\nCOUNTRIES")
        print("Unique count:", len(set(countries)))
        print(sorted(set(countries)))