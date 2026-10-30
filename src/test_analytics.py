
from pathlib import Path

from test_analytics import (
    load_data,
    prepare_data,
    calculate_kpis,
    analyze_segment,
)

BASE_DIR = Path(__file__).resolve().parent

tables = load_data(BASE_DIR / "data")
df = prepare_data(tables)

kpis = calculate_kpis(df)

print("\nDELIVERY ANALYTICS TEST")
print("-" * 35)
print("Total orders:", kpis["total_orders"])
print("Delivered:", kpis["delivered_orders"])
print("Late orders:", kpis["late_orders"])
print("On-time percentage:", kpis["on_time_percentage"], "%")
print("Average delay:", kpis["average_delay_days"], "days")

print("\nLate rate by carrier:")
print(analyze_segment(df, "carrier").to_string(index=False))

assert len(df) > 0
assert kpis["total_orders"] == len(df)
assert 0 <= kpis["on_time_percentage"] <= 100

print("\nAll basic tests passed!")