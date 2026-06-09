# scripts/inspect_raw_comtrade.py

import os
import comtradeapicall

api_key = os.environ["COMTRADE_API_KEY"]

raw = comtradeapicall.getFinalData(
    subscription_key=api_key,
    typeCode="C",
    freqCode="A",
    clCode="HS",
    period="2019",
    reporterCode="036",
    cmdCode="AG2",
    flowCode="M",
    partnerCode=None,
    partner2Code=None,
    customsCode=None,
    motCode=None,
)

print(raw.columns.tolist())

for col in raw.columns:
    print(col)