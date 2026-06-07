from pyxlsb import open_workbook

file_path = "data/raw/wiod/WIOT2014_Nov16_ROW.xlsb"

with open_workbook(file_path) as wb:
    with wb.get_sheet("2014") as sheet:

        for row_idx, row in enumerate(sheet.rows()):
            values = [cell.v for cell in row[:15]]

            print(f"ROW {row_idx}:")
            print(values)
            print("-" * 80)

            if row_idx >= 20:
                break