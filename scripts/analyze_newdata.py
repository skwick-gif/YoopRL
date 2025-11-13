import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

NEWDATA_DIR = PROJECT_ROOT / "NEWDATA"

symbols = ["TNA", "IWM", "SPY", "UPRO", "QQQ", "TQQQ"]

results = {}
for symbol in symbols:
    csv_path = NEWDATA_DIR / f"{symbol}_15MIN_DATA.csv"
    if not csv_path.exists():
        results[symbol] = "missing"
        continue

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime")
    results[symbol] = {
        "rows": len(df),
        "first": df["datetime"].iloc[0],
        "last": df["datetime"].iloc[-1],
        "sample_cols": list(df.columns),
    }

for symbol, info in results.items():
    print(symbol)
    print(info)
