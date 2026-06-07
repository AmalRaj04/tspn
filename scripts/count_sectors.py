import config

for i, sector in enumerate(config.GRAPH["SECTOR_LIST"], start=1):
    print(i, sector)

print()
print("TOTAL:", len(config.GRAPH["SECTOR_LIST"]))