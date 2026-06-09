import os
import comtradeapicall

api_key = os.environ["COMTRADE_API_KEY"]

raw = comtradeapicall.getFinalData(
    subscription_key=api_key,
    typeCode="C",
    freqCode="A",
    clCode="HS",
    period="2019",
    reporterCode="036",  # Australia
    cmdCode="AG2",
    flowCode="M",
    partnerCode=None,
)

print("=" * 80)
print("TYPE")
print("=" * 80)
print(type(raw))

print("\n" + "=" * 80)
print("COLUMNS")
print("=" * 80)

if hasattr(raw, "columns"):
    print(raw.columns.tolist())

print("\n" + "=" * 80)
print("HEAD")
print("=" * 80)

try:
    print(raw.head())
except Exception as e:
    print(e)

print("\n" + "=" * 80)
print("SHAPE")
print("=" * 80)

try:
    print(raw.shape)
except Exception as e:
    print(e)