import comtradeapicall as ct

API_KEY = ""

print(
    ct.getFinalDataAvailability(
        subscription_key=API_KEY,
        typeCode="C",
        freqCode="A",
        clCode="HS"
    )
)