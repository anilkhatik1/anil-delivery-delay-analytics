
from pathlib import Path

import numpy as np
import pandas as pd

def load_data(data_dir="data"):
    """Load all operational CSV tables."""
    folder = Path(data_dir)

    return {
        "orders": pd.read_csv(folder / "orders.csv"),
        "shipments": pd.read_csv(folder / "shipments.csv"),
        "warehouses": pd.read_csv(folder / "warehouses.csv"),
        "carriers": pd.read_csv(folder / "carriers.csv"),
        "events": pd.read_csv(folder / "delivery_events.csv"),
    }

def prepare_data(tables):
    """Join tables and calculate delivery performance."""

    orders = tables["orders"].copy()
    shipments = tables["shipments"].copy()
    events = tables["events"].copy()

    required_orders = {
        "order_id", "order_date", "promised_days"
    }
    required_shipments = {
        "order_id", "carrier", "route_type"
    }

    if not required_orders.issubset(orders.columns):
        raise ValueError("Required columns are missing from orders.csv")

    if not required_shipments.issubset(shipments.columns):
        raise ValueError("Required columns are missing from shipments.csv")

    orders["order_date"] = pd.to_datetime(
        orders["order_date"], errors="coerce"
    )
    orders["promised_days"] = pd.to_numeric(
        orders["promised_days"], errors="coerce"
    )

    orders = orders.dropna(
        subset=["order_id", "order_date", "promised_days"]
    )

    # One shipment record per order for this project.
    shipments = shipments.drop_duplicates(
        subset="order_id", keep="last"
    )

    df = orders.merge(
        shipments,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    # Add warehouse details.
    if "warehouses" in tables:
        warehouse_df = tables["warehouses"]

        df = df.merge(
            warehouse_df,
            on="warehouse_id",
            how="left",
            validate="many_to_one"
        )

    # Add carrier details.
    if "carriers" in tables:
        carrier_df = tables["carriers"]

        df = df.merge(
            carrier_df,
            on="carrier",
            how="left",
            validate="many_to_one"
        )

    # Obtain actual delivery from the Delivered event.
    events["event_timestamp"] = pd.to_datetime(
        events["event_timestamp"], errors="coerce"
    )

    delivered_events = events[
        events["event_type"].str.casefold() == "delivered"
    ].dropna(subset=["event_timestamp"])

    actual_dates = (
        delivered_events
        .sort_values("event_timestamp")
        .groupby("order_id")["event_timestamp"]
        .last()
    )

    df["actual_delivery_date"] = df["order_id"].map(
        actual_dates
    )

    # Calculate promised delivery date.
    df["promised_delivery_date"] = (
        df["order_date"]
        + pd.to_timedelta(df["promised_days"], unit="D")
    )

    # Calculate delivery metrics.
    df["delay_days"] = (
        df["actual_delivery_date"].dt.normalize()
        - df["promised_delivery_date"].dt.normalize()
    ).dt.days

    df["transit_days"] = (
        df["actual_delivery_date"] - df["order_date"]
    ).dt.total_seconds() / 86400

    df["is_delivered"] = df["actual_delivery_date"].notna()

    df["is_late"] = (
        df["is_delivered"] & (df["delay_days"] > 0)
    )

    return df

def calculate_kpis(df):
    """Calculate high-level delivery KPIs."""

    delivered = df[df["is_delivered"]]
    late = delivered[delivered["delay_days"] > 0]

    if delivered.empty:
        return {
            "total_orders": len(df),
            "delivered_orders": 0,
            "late_orders": 0,
            "on_time_percentage": 0.0,
            "average_delay_days": 0.0,
            "median_transit_days": 0.0,
        }

    return {
        "total_orders": int(len(df)),
        "delivered_orders": int(len(delivered)),
        "late_orders": int(len(late)),
        "on_time_percentage": round(
            100 * (delivered["delay_days"] <= 0).mean(), 2
        ),
        "average_delay_days": round(
            late["delay_days"].mean(), 2
        ) if not late.empty else 0.0,
        "median_transit_days": round(
            delivered["transit_days"].median(), 2
        ),
    }

def analyze_segment(df, column):
    """Compare delivery delays across an operational category."""

    delivered = df[df["is_delivered"]].copy()

    if column not in delivered.columns:
        raise ValueError(f"Unknown analysis column: {column}")

    if delivered.empty:
        return pd.DataFrame(columns=[
            column, "orders", "late_orders",
            "late_rate_percentage", "average_delay_days"
        ])

    delivered["late_flag"] = delivered["delay_days"] > 0

    result = (
        delivered.groupby(column, dropna=False)
        .agg(
            orders=("order_id", "count"),
            late_orders=("late_flag", "sum"),
            average_delay_days=("delay_days", "mean"),
        )
        .reset_index()
    )

    result["late_rate_percentage"] = (
        100 * result["late_orders"] / result["orders"]
    )

    result["average_delay_days"] = (
        delivered.assign(
            positive_delay=delivered["delay_days"].clip(lower=0)
        )
        .groupby(column, dropna=False)["positive_delay"]
        .mean()
        .reindex(result[column])
        .to_numpy()
    )

    return result.sort_values(
        "late_rate_percentage", ascending=False
    ).round(2)
def train_delay_model(df):
    """
    Train a decision tree to estimate delivery-delay risk.

    Only completed deliveries are used because their actual
    delivery outcomes are known.
    """

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    # Use completed deliveries only.
    data = df[df["is_delivered"]].copy()

    if len(data) < 40:
        return {
            "error": "At least 40 completed deliveries are recommended."
        }

    # Target variable:
    # 1 = late delivery
    # 0 = on-time delivery
    y = (data["delay_days"] > 0).astype(int)

    if y.nunique() < 2 or y.value_counts().min() < 2:
        return {
            "error": (
                "The dataset must contain enough on-time "
                "and late deliveries."
            )
        }

    # Operational factors available before delivery.
    feature_columns = [
        "warehouse_id",
        "carrier",
        "route_type",
        "product_type",
        "day_of_week",
        "season",
        "customer_region",
    ]

    missing = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing:
        return {
            "error": f"Missing feature columns: {missing}"
        }

    X = data[feature_columns].copy()

    # Handle missing categorical values.
    X = X.fillna("Unknown").astype(str)

    # Separate training and testing records.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Convert categorical variables into numeric features.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                feature_columns,
            )
        ]
    )

    # Build an interpretable decision tree.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=15,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # Train the model.
    model.fit(X_train, y_train)

    # Evaluate on previously unseen test records.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(
            y_test, probabilities
        )
    except ValueError:
        metrics["roc_auc"] = None

    # Extract feature importance.
    encoder = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = encoder.get_feature_names_out()
    importance_values = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    importance_df["feature"] = (
        importance_df["feature"]
        .str.replace("categories__", "", regex=False)
    )

    importance_df = (
        importance_df[importance_df["importance"] > 0]
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    return {
        "model": model,
        "metrics": metrics,
        "importance": importance_df,
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }
def validate_tables(tables):
    """
    Validate the five operational CSV tables.

    Returns a list of validation errors and warnings.
    An empty error list means the required structure is valid.
    """

    errors = []
    warnings = []

    required_columns = {
        "orders": [
            "order_id",
            "order_date",
            "promised_days",
            "warehouse_id",
            "product_type",
            "customer_region",
        ],
        "shipments": [
            "order_id",
            "carrier",
            "route_type",
        ],
        "warehouses": [
            "warehouse_id",
        ],
        "carriers": [
            "carrier",
        ],
        "events": [
            "order_id",
            "event_type",
            "event_timestamp",
        ],
    }

    # 1. Check required tables and columns.
    for table_name, columns in required_columns.items():

        if table_name not in tables:
            errors.append(
                f"Missing table: {table_name}"
            )
            continue

        table = tables[table_name]

        if table.empty:
            errors.append(
                f"The {table_name} table is empty."
            )
            continue

        missing_columns = [
            column
            for column in columns
            if column not in table.columns
        ]

        if missing_columns:
            errors.append(
                f"{table_name}: missing columns "
                f"{missing_columns}"
            )

    # Stop deeper checks if the basic structure is invalid.
    if errors:
        return errors, warnings

    orders = tables["orders"]
    shipments = tables["shipments"]
    events = tables["events"]

    # 2. Check missing order IDs.
    if orders["order_id"].isna().any():
        errors.append(
            "Orders contain missing order_id values."
        )

    if shipments["order_id"].isna().any():
        errors.append(
            "Shipments contain missing order_id values."
        )

    # 3. Check duplicate identifiers.
    duplicate_orders = int(
        orders["order_id"].duplicated().sum()
    )

    if duplicate_orders:
        errors.append(
            f"Orders contain {duplicate_orders} duplicate order IDs."
        )

    duplicate_shipments = int(
        shipments["order_id"].duplicated().sum()
    )

    if duplicate_shipments:
        warnings.append(
            f"Shipments contain {duplicate_shipments} repeated "
            "order IDs. The current analysis uses one shipment "
            "record per order."
        )

    # 4. Check date and numeric fields.
    order_dates = pd.to_datetime(
        orders["order_date"],
        errors="coerce",
    )

    if order_dates.isna().any():
        errors.append(
            "Orders contain missing or invalid order dates."
        )

    promised_days = pd.to_numeric(
        orders["promised_days"],
        errors="coerce",
    )

    if promised_days.isna().any():
        errors.append(
            "Orders contain missing or non-numeric promised_days."
        )

    elif (promised_days < 0).any():
        errors.append(
            "promised_days cannot be negative."
        )

    # 5. Check references.
    missing_shipments = (
        ~orders["order_id"].isin(shipments["order_id"])
    ).sum()

    if missing_shipments:
        warnings.append(
            f"{int(missing_shipments)} orders have no matching shipment."
        )

    missing_warehouses = (
        ~orders["warehouse_id"].isin(
            tables["warehouses"]["warehouse_id"]
        )
    ).sum()

    if missing_warehouses:
        warnings.append(
            f"{int(missing_warehouses)} orders reference "
            "unknown warehouses."
        )

    missing_carriers = (
        ~shipments["carrier"].isin(
            tables["carriers"]["carrier"]
        )
    ).sum()

    if missing_carriers:
        warnings.append(
            f"{int(missing_carriers)} shipments reference "
            "unknown carriers."
        )

    # 6. Check delivery-event dates.
    event_dates = pd.to_datetime(
        events["event_timestamp"],
        errors="coerce",
    )

    invalid_event_dates = int(event_dates.isna().sum())

    if invalid_event_dates:
        warnings.append(
            f"{invalid_event_dates} delivery events have "
            "missing or invalid timestamps."
        )

    # 7. Check unexpected event types.
    valid_event_types = {
        "order confirmed",
        "delivered",
    }

    event_types = (
        events["event_type"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    unexpected_types = sorted(
        set(event_types) - valid_event_types
    )

    if unexpected_types:
        warnings.append(
            f"Unrecognized event types: {unexpected_types}"
        )

    # 8. Check missing important values.
    for column in [
        "warehouse_id",
        "product_type",
        "customer_region",
    ]:

        missing_count = int(
            orders[column].isna().sum()
        )

        if missing_count:
            warnings.append(
                f"Orders contain {missing_count} missing values "
                f"in {column}."
            )

    return errors, warnings 
def analyze_delay_factors(df, min_orders=10):
    """
    Analyze operational factors associated with delivery delays.
    """

    feature_columns = [
        "warehouse_id",
        "carrier",
        "route_type",
        "product_type",
        "day_of_week",
        "season",
        "customer_region",
    ]

    # Select completed deliveries
    delivered = df[df["is_delivered"]].copy()

    if delivered.empty:
        return pd.DataFrame()

    # Calculate delay indicators
    delivered["late_flag"] = (
        delivered["delay_days"] > 0
    )

    delivered["positive_delay"] = (
        delivered["delay_days"].clip(lower=0)
    )

    results = []

    # Analyze each operational factor
    for column in feature_columns:

        if column not in delivered.columns:
            continue

        grouped = delivered.groupby(
            column,
            dropna=False
        ).agg(
            orders=("order_id", "count"),
            late_orders=("late_flag", "sum"),
            average_delay_days=("positive_delay", "mean")
        ).reset_index()

        # Ignore categories with insufficient records
        grouped = grouped[
            grouped["orders"] >= min_orders
        ].copy()

        if grouped.empty:
            continue

        # Calculate late delivery percentage
        grouped["late_rate_percentage"] = (
            grouped["late_orders"]
            / grouped["orders"]
            * 100
        )

        grouped["factor"] = column

        grouped.rename(
            columns={column: "category"},
            inplace=True
        )

        results.append(
            grouped[
                [
                    "factor",
                    "category",
                    "orders",
                    "late_orders",
                    "late_rate_percentage",
                    "average_delay_days",
                ]
            ]
        )

    # Return empty DataFrame if no results exist
    if not results:
        return pd.DataFrame()

    # Combine all factor analysis results
    final_result = pd.concat(
        results,
        ignore_index=True
    )

    final_result = final_result.sort_values(
        by="late_rate_percentage",
        ascending=False
    )

def rank_carriers(df):
    """
    Rank delivery carriers based on delivery performance.
    """

    delivered = df[df["is_delivered"]].copy()

    if delivered.empty or "carrier" not in delivered.columns:
        return pd.DataFrame()

    ranking = (
        delivered.groupby("carrier")
        .agg(
            total_deliveries=("order_id", "count"),
            late_deliveries=("is_late", "sum"),
            average_delay_days=("delay_days", "mean")
        )
        .reset_index()
    )

    ranking["on_time_deliveries"] = (
        ranking["total_deliveries"]
        - ranking["late_deliveries"]
    )

    ranking["on_time_percentage"] = (
        ranking["on_time_deliveries"]
        / ranking["total_deliveries"]
        * 100
    )

    ranking["late_percentage"] = (
        ranking["late_deliveries"]
        / ranking["total_deliveries"]
        * 100
    )

    ranking = ranking.sort_values(
        by="on_time_percentage",
        ascending=False
    )

    return ranking.round(2).reset_index(drop=True)
def rank_warehouses(df):
    """
    Rank warehouses based on delivery performance.
    """

    delivered = df[df["is_delivered"]].copy()

    if delivered.empty or "warehouse_id" not in delivered.columns:
        return pd.DataFrame()

    ranking = (
        delivered.groupby("warehouse_id")
        .agg(
            total_deliveries=("order_id", "count"),
            late_deliveries=("is_late", "sum"),
            average_delay_days=("delay_days", "mean")
        )
        .reset_index()
    )

    ranking["on_time_deliveries"] = (
        ranking["total_deliveries"]
        - ranking["late_deliveries"]
    )

    ranking["on_time_percentage"] = (
        ranking["on_time_deliveries"]
        / ranking["total_deliveries"]
        * 100
    )

    ranking["late_percentage"] = (
        ranking["late_deliveries"]
        / ranking["total_deliveries"]
        * 100
    )

    ranking = ranking.sort_values(
        by="on_time_percentage",
        ascending=False
    )

    return ranking.round(2).reset_index(drop=True)
def generate_operational_alerts(df, min_orders=10):
    """Generate alerts for operational areas with poor delivery performance."""

    delivered = df[df["is_delivered"]].copy()

    if delivered.empty:
        return ["No completed deliveries available for analysis."]

    alerts = []

    # Carrier alerts
    if "carrier" in delivered.columns:
        carrier_stats = delivered.groupby("carrier").agg(
            orders=("order_id", "count"),
            late_rate=("is_late", "mean"),
        )

        for carrier, row in carrier_stats.iterrows():
            if row["orders"] >= min_orders and row["late_rate"] >= 0.30:
                alerts.append(
                    f"Carrier {carrier} has a high late-delivery rate "
                    f"of {row['late_rate'] * 100:.1f}%."
                )

    # Warehouse alerts
    if "warehouse_id" in delivered.columns:
        warehouse_stats = delivered.groupby("warehouse_id").agg(
            orders=("order_id", "count"),
            late_rate=("is_late", "mean"),
        )

        for warehouse, row in warehouse_stats.iterrows():
            if row["orders"] >= min_orders and row["late_rate"] >= 0.30:
                alerts.append(
                    f"Warehouse {warehouse} has a high late-delivery rate "
                    f"of {row['late_rate'] * 100:.1f}%."
                )

    if not alerts:
        alerts.append(
            "No major operational risks detected using the current alert thresholds."
        )

    return alerts
def generate_executive_summary(df):
    """Generate a business-friendly summary of delivery performance."""

    delivered = df[df["is_delivered"]].copy()

    if delivered.empty:
        return "There are no completed deliveries available for analysis."

    total_orders = len(df)
    delivered_orders = len(delivered)
    late_orders = int(delivered["is_late"].sum())

    on_time_rate = (
        (delivered_orders - late_orders)
        / delivered_orders
        * 100
    )

    average_delay = delivered["delay_days"].clip(lower=0).mean()

    summary = (
        f"Out of {total_orders:,} orders, {delivered_orders:,} have been delivered. "
        f"{late_orders:,} delivered orders were late. "
        f"The on-time delivery rate is {on_time_rate:.1f}%, "
        f"with an average positive delay of {average_delay:.1f} days."
    )

    if on_time_rate >= 90:
        performance = "Delivery performance is strong."
    elif on_time_rate >= 75:
        performance = "Delivery performance needs improvement."
    else:
        performance = "Delivery performance requires immediate attention."

    return f"{summary} {performance}"