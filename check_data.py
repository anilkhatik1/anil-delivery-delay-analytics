import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

datasets = {
    "Orders": "orders.csv",
    "Shipments": "shipments.csv",
    "Warehouses": "warehouses.csv",
    "Carriers": "carriers.csv",
    "Delivery Events": "delivery_events.csv"
}

print("\nDELIVERY DATA QUALITY REPORT")
print("=" * 45)

for name, filename in datasets.items():
    try:
        csv_path = BASE_DIR / filename
        df = pd.read_csv(csv_path)

        print(f"\n{name}")
        print("-" * 30)
        print("Total records:", len(df))
        print("Total columns:", len(df.columns))
        print("Missing values:", int(df.isnull().sum().sum()))
        print("Duplicate rows:", int(df.duplicated().sum()))

        if df.empty:
            print("STATUS: FAILED - Dataset is empty")
        else:
            print("STATUS: PASSED")

    except Exception as error:
        print(f"\n{name}: FAILED - {error}")

print("\nData quality check completed.")