import pandas as pd
import numpy as np
from pathlib import Path

# Reproducible sample data
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
TOTAL_ORDERS = 2000

print("Generating delivery operations data...")

# -----------------------------
# 1. Warehouse data
# -----------------------------
warehouses = pd.DataFrame({
    "warehouse_id": ["WH01", "WH02", "WH03", "WH04", "WH05"],
    "warehouse_city": [
        "Jaipur", "Delhi", "Mumbai", "Bangalore", "Pune"
    ],
    "warehouse_capacity": [5000, 6500, 4500, 5500, 4000]
})

# -----------------------------
# 2. Carrier data
# -----------------------------
carriers = pd.DataFrame({
    "carrier": [
        "BlueDart", "Delhivery", "DTDC", "Ecom Express"
    ],
    "service_level": [
        "Premium", "Standard", "Standard", "Economy"
    ]
})

# -----------------------------
# 3. Orders data
# -----------------------------
order_ids = [f"ORD{i:05d}" for i in range(1, TOTAL_ORDERS + 1)]

order_dates = pd.to_datetime(
    np.random.choice(
        pd.date_range("2025-01-01", "2026-10-01"),
        TOTAL_ORDERS
    )
)

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    return "Autumn"

orders = pd.DataFrame({
    "order_id": order_ids,
    "order_date": order_dates,
    "promised_days": np.random.choice(
        [2, 3, 4, 5, 7],
        TOTAL_ORDERS,
        p=[0.15, 0.30, 0.25, 0.20, 0.10]
    ),
    "warehouse_id": np.random.choice(
        warehouses["warehouse_id"],
        TOTAL_ORDERS
    ),
    "product_type": np.random.choice(
        ["Electronics", "Clothing", "Grocery", "Furniture", "Books"],
        TOTAL_ORDERS
    ),
    "customer_region": np.random.choice(
        ["North", "South", "East", "West", "Central"],
        TOTAL_ORDERS
    )
})

orders["day_of_week"] = orders["order_date"].dt.day_name()
orders["season"] = orders["order_date"].dt.month.map(get_season)

# -----------------------------
# 4. Shipment data
# -----------------------------
shipments = pd.DataFrame({
    "order_id": order_ids,
    "carrier": np.random.choice(
        carriers["carrier"],
        TOTAL_ORDERS,
        p=[0.30, 0.35, 0.20, 0.15]
    ),
    "route_type": np.random.choice(
        ["Local", "Regional", "National"],
        TOTAL_ORDERS,
        p=[0.35, 0.40, 0.25]
    )
})

# -----------------------------
# 5. Delivery events
# -----------------------------
events = []

for i in range(TOTAL_ORDERS):

    order = orders.iloc[i]
    shipment = shipments.iloc[i]

    order_date = order["order_date"]
    promised_days = int(order["promised_days"])

    route = shipment["route_type"]
    carrier = shipment["carrier"]

    # Simulate delivery delay
    delay_probability = 0.18

    if route == "National":
        delay_probability += 0.15

    if route == "Regional":
        delay_probability += 0.07

    if carrier == "Ecom Express":
        delay_probability += 0.10

    if carrier == "BlueDart":
        delay_probability -= 0.06

    is_late = np.random.random() < delay_probability

    if is_late:
        extra_days = np.random.randint(1, 5)
    else:
        extra_days = -np.random.randint(0, 2)

    actual_days = max(1, promised_days + extra_days)

    # Keep a small proportion of orders undelivered
    is_delivered = np.random.random() < 0.95

    events.append({
        "order_id": order["order_id"],
        "event_type": "Order Confirmed",
        "event_timestamp": order_date
    })

    if is_delivered:
        delivery_date = order_date + pd.Timedelta(days=actual_days)

        events.append({
            "order_id": order["order_id"],
            "event_type": "Delivered",
            "event_timestamp": delivery_date
        })

delivery_events = pd.DataFrame(events)

# -----------------------------
# 6. Save CSV files
# -----------------------------
orders.to_csv(BASE_DIR / "orders.csv", index=False)
shipments.to_csv(BASE_DIR / "shipments.csv", index=False)
warehouses.to_csv(BASE_DIR / "warehouses.csv", index=False)
carriers.to_csv(BASE_DIR / "carriers.csv", index=False)
delivery_events.to_csv(
    BASE_DIR / "delivery_events.csv",
    index=False
)

print("\nAll datasets generated successfully!")
print(f"Total orders: {len(orders)}")
print(f"Total shipments: {len(shipments)}")
print(f"Total warehouses: {len(warehouses)}")
print(f"Total carriers: {len(carriers)}")
print(f"Total delivery events: {len(delivery_events)}")

print("\nCSV files saved in:", BASE_DIR)