import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.data_download.intraday_loader import build_intraday_dataset

def main():
    df = build_intraday_dataset(("TNA", "IWM"), start="2021-01-04", end="2025-11-10")
    print(len(df))
    if not df.empty:
        print(df.index[0], df.index[-1])


if __name__ == "__main__":
    main()
