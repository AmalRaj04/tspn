import numpy as np

for year in range(2000, 2015):
    bl = np.load(
        f"data/processed/leontief/backward_linkage_{year}.npy"
    )

    print(
        year,
        round(float(bl.max()), 2),
        round(float(bl.mean()), 2)
    )