from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import (
    analyze_segment,
    generate_executive_summary,
    analyze_delay_factors,
    calculate_kpis,
    generate_operational_alerts,
    load_data,
    prepare_data,
    rank_carriers,
    rank_warehouses,
    train_delay_model,
    validate_tables,
)


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Delivery Operations Analytics",
    page_icon="📦",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

st.title("📦 Delivery Delay & Operations Analytics")
st.caption(
    "Monitor delivery performance, identify operational bottlenecks, "
    "and predict potential delivery delays."
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 14px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2. DATA UPLOAD
# =========================================================

st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.subheader("📂 Data Source")

st.sidebar.write(
    "Upload all five operational CSV files to analyze another "
    "dataset. Leave all upload fields empty to use sample data."
)

upload_config = {
    "orders": "Upload Orders CSV",
    "shipments": "Upload Shipments CSV",
    "warehouses": "Upload Warehouses CSV",
    "carriers": "Upload Carriers CSV",
    "events": "Upload Delivery Events CSV",
}

uploaded_files = {}

for table_name, label in upload_config.items():
    uploaded_files[table_name] = st.sidebar.file_uploader(
        label,
        type=["csv"],
        key=f"upload_{table_name}",
    )

uploaded_count = sum(
    uploaded_file is not None
    for uploaded_file in uploaded_files.values()
)


# =========================================================
# 3. LOAD AND VALIDATE DATA
# =========================================================

try:
    if uploaded_count == 0:

        tables = load_data(BASE_DIR)

        st.info(
            "Using the generated sample dataset. Upload all five "
            "CSV files in the sidebar to analyze another dataset."
        )

    elif uploaded_count < len(upload_config):

        st.warning(
            f"You uploaded {uploaded_count} of 5 required files. "
            "Please upload all five CSV files."
        )

        st.stop()

    else:
        tables = {}

        for table_name, uploaded_file in uploaded_files.items():
            uploaded_file.seek(0)
            tables[table_name] = pd.read_csv(
                BytesIO(uploaded_file.getvalue())
            )

        st.success("All five CSV files uploaded successfully.")

    errors, warnings = validate_tables(tables)

    with st.expander("🧪 Data Quality Report", expanded=True):

        if errors:
            st.error(
                f"Data validation found {len(errors)} error(s). "
                "Please correct the dataset before continuing."
            )

            for error in errors:
                st.write(f"❌ {error}")

            for warning in warnings:
                st.warning(f"⚠️ {warning}")

            st.stop()

        st.success(
            "Required columns and basic data checks passed."
        )

        if warnings:
            for warning in warnings:
                st.warning(f"⚠️ {warning}")
        else:
            st.caption(
                "No additional data-quality warnings were detected."
            )
    df = prepare_data(tables)

    if df.empty:
        st.error(
            "No valid orders remain after data preparation."
        )
        st.stop()
except Exception as error:

    st.error(
        f"Unable to load or process the dataset: {error}"
    )

    st.info(
        "Check CSV file names, column names, and data types. "
        "If using sample data, run python generate_data.py first."
    )

    st.stop()
# =========================================================
# 4. SIDEBAR FILTERS
# =========================================================

st.sidebar.divider()
st.sidebar.subheader("🔎 Filter Data")

warehouse_options = sorted(
    df["warehouse_id"].dropna().unique().tolist()
)

carrier_options = sorted(
    df["carrier"].dropna().unique().tolist()
)

route_options = sorted(
    df["route_type"].dropna().unique().tolist()
)

product_options = sorted(
    df["product_type"].dropna().unique().tolist()
)

selected_warehouses = st.sidebar.multiselect(
    "Warehouse",
    options=warehouse_options,
    default=warehouse_options,
)

selected_carriers = st.sidebar.multiselect(
    "Carrier",
    options=carrier_options,
    default=carrier_options,
)

selected_routes = st.sidebar.multiselect(
    "Route Type",
    options=route_options,
    default=route_options,
)

# Clear the trained model when dashboard filters change

current_filters = (
    tuple(selected_warehouses),
    tuple(selected_carriers),
    tuple(selected_routes),
    tuple(selected_products),
)

if st.session_state.get("previous_filters") != current_filters:

    st.session_state.pop(
        "delivery_model_result",
        None,
    )

    st.session_state["previous_filters"] = current_filters
filtered_df = df[
    df["warehouse_id"].isin(selected_warehouses)
    & df["carrier"].isin(selected_carriers)
    & df["route_type"].isin(selected_routes)
    & df["product_type"].isin(selected_products)].copy()
# Executive Summary

st.header("📊 Executive Summary")

executive_summary = generate_executive_summary(filtered_df)

st.info(executive_summary)

if filtered_df.empty:
    st.warning(
        "No records match the selected filters. "
        "Change one or more filters in the sidebar."
    )
    st.stop()


# =========================================================
# 5. KEY PERFORMANCE INDICATORS
# =========================================================

st.header("📊 Business Performance Overview")

kpis = calculate_kpis(filtered_df)

metric_columns = st.columns(5)

metric_columns[0].metric(
    "Total Orders",
    f"{kpis['total_orders']:,}",
)

metric_columns[1].metric(
    "Delivered Orders",
    f"{kpis['delivered_orders']:,}",
)

metric_columns[2].metric(
    "Late Orders",
    f"{kpis['late_orders']:,}",
)

metric_columns[3].metric(
    "On-Time Delivery",
    f"{kpis['on_time_percentage']:.1f}%",
)

metric_columns[4].metric(
    "Average Late Delay",
    f"{kpis['average_delay_days']:.1f} days",
)

st.divider()


# =========================================================
# 6. DELIVERY PERFORMANCE TRENDS
# =========================================================

st.header("📈 Delivery Performance Trends")

delivered_df = filtered_df[
    filtered_df["is_delivered"]
].copy()

if not delivered_df.empty:

    delivered_df["delivery_month"] = (
        delivered_df["actual_delivery_date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_data = (
        delivered_df.groupby("delivery_month")
        .agg(
            total_deliveries=("order_id", "count"),
            late_orders=("is_late", "sum"),
        )
        .reset_index()
    )

    monthly_data["on_time_percentage"] = (
        100
        * (
            monthly_data["total_deliveries"]
            - monthly_data["late_orders"]
        )
        / monthly_data["total_deliveries"]
    )

    trend_chart = px.line(
        monthly_data,
        x="delivery_month",
        y="on_time_percentage",
        markers=True,
        title="Monthly On-Time Delivery Percentage",
        labels={
            "delivery_month": "Delivery Month",
            "on_time_percentage": "On-Time Delivery (%)",
        },
    )

    trend_chart.update_yaxes(range=[0, 100])

    trend_chart.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(
        trend_chart,
        use_container_width=True,
    )

else:
    st.info(
        "No completed deliveries are available for trend analysis."
    )

st.divider()

# =========================================================
# WAREHOUSE EFFICIENCY LEADERBOARD
# =========================================================

st.header("🏭 Warehouse Efficiency Leaderboard")

warehouse_ranking = rank_warehouses(filtered_df)

if not warehouse_ranking.empty:

    st.write(
        "Warehouses ranked by on-time delivery performance."
    )

    st.dataframe(
        warehouse_ranking,
        width="stretch",
        hide_index=True
    )

    best_warehouse = warehouse_ranking.iloc[0]

    st.success(
        f"Top-performing warehouse: "
        f"{best_warehouse['warehouse_id']} "
        f"with {best_warehouse['on_time_percentage']:.1f}% "
        "on-time deliveries."
    )

    warehouse_ranking_chart = px.bar(
        warehouse_ranking,
        x="warehouse_id",
        y="on_time_percentage",
        text="on_time_percentage",
        title="Warehouse On-Time Delivery Ranking",
        labels={
            "warehouse_id": "Warehouse",
            "on_time_percentage": "On-Time Delivery (%)"
        }
    )

    warehouse_ranking_chart.update_traces(
        texttemplate="%{text:.1f}%"
    )

    warehouse_ranking_chart.update_yaxes(
        range=[0, 100]
    )

    st.plotly_chart(
        warehouse_ranking_chart,
        width="stretch"
    )

else:

    st.info(
        "No completed deliveries available for warehouse ranking."
    )

st.divider()
# =========================================================
# 7. CARRIER PERFORMANCE
# =========================================================

st.header("🚚 Carrier Performance Analysis")

carrier_data = analyze_segment(
    filtered_df,
    "carrier",
)

if not carrier_data.empty:

    left_column, right_column = st.columns(2)

    with left_column:

        carrier_late_chart = px.bar(
            carrier_data,
            x="carrier",
            y="late_rate_percentage",
            text="late_rate_percentage",
            title="Late Delivery Rate by Carrier",
            labels={
                "carrier": "Carrier",
                "late_rate_percentage": "Late Rate (%)",
            },
        )

        carrier_late_chart.update_traces(
            texttemplate="%{text:.1f}%"
        )

        st.plotly_chart(
            carrier_late_chart,
            use_container_width=True,
        )

    with right_column:

        carrier_delay_chart = px.bar(
            carrier_data,
            x="carrier",
            y="average_delay_days",
            text="average_delay_days",
            title="Average Delay by Carrier",
            labels={
                "carrier": "Carrier",
                "average_delay_days": "Average Delay (Days)",
            },
        )

        carrier_delay_chart.update_traces(
            texttemplate="%{text:.1f}"
        )

        st.plotly_chart(
            carrier_delay_chart,
            use_container_width=True,
        )

    with st.expander("View carrier performance details"):
        st.dataframe(
            carrier_data,
            use_container_width=True,
            hide_index=True,
        )

else:
    st.info(
        "There is not enough completed-delivery data for carrier analysis."
    )

st.divider()


# =========================================================
# 8. WAREHOUSE BOTTLENECK ANALYSIS
# =========================================================

st.header("🏭 Warehouse Bottleneck Analysis")

warehouse_data = analyze_segment(
    filtered_df,
    "warehouse_id",
)

if not warehouse_data.empty:

    warehouse_chart = px.bar(
        warehouse_data.sort_values(
            "late_rate_percentage",
            ascending=False,
        ),
        x="warehouse_id",
        y="late_rate_percentage",
        text="late_rate_percentage",
        title="Late Delivery Rate by Warehouse",
        labels={
            "warehouse_id": "Warehouse",
            "late_rate_percentage": "Late Rate (%)",
        },
    )

    warehouse_chart.update_traces(
        texttemplate="%{text:.1f}%"
    )

    st.plotly_chart(
        warehouse_chart,
        use_container_width=True,
    )

    with st.expander("View warehouse performance details"):
        st.dataframe(
            warehouse_data,
            use_container_width=True,
            hide_index=True,
        )

else:
    st.info(
        "There is not enough completed-delivery data for warehouse analysis."
    )

st.divider()


# =========================================================
# 9. ROUTE PERFORMANCE
# =========================================================

st.header("🛣️ Route Performance")

route_data = analyze_segment(
    filtered_df,
    "route_type",
)

if not route_data.empty:

    route_chart = px.bar(
        route_data.sort_values(
            "late_rate_percentage",
            ascending=False,
        ),
        x="route_type",
        y="late_rate_percentage",
        text="late_rate_percentage",
        title="Late Delivery Rate by Route Type",
        labels={
            "route_type": "Route Type",
            "late_rate_percentage": "Late Rate (%)",
        },
    )

    route_chart.update_traces(
        texttemplate="%{text:.1f}%"
    )

    st.plotly_chart(
        route_chart,
        use_container_width=True,
    )

    with st.expander("View route performance details"):
        st.dataframe(
            route_data,
            use_container_width=True,
            hide_index=True,
        )

else:
    st.info(
        "There is not enough completed-delivery data for route analysis."
    )

st.divider()


# =========================================================
# 10. PRODUCT CATEGORY ANALYSIS
# =========================================================

st.header("📦 Product Category Analysis")

product_data = analyze_segment(
    filtered_df,
    "product_type",
)

if not product_data.empty:

    product_chart = px.bar(
        product_data.sort_values(
            "late_rate_percentage",
            ascending=False,
        ),
        x="product_type",
        y="late_rate_percentage",
        text="late_rate_percentage",
        title="Late Delivery Rate by Product Category",
        labels={
            "product_type": "Product Category",
            "late_rate_percentage": "Late Rate (%)",
        },
    )

    product_chart.update_traces(
        texttemplate="%{text:.1f}%"
    )

    st.plotly_chart(
        product_chart,
        use_container_width=True,
    )

    with st.expander("View product category details"):
        st.dataframe(
            product_data,
            use_container_width=True,
            hide_index=True,
        )

else:
    st.info(
        "There is not enough completed-delivery data for product analysis."
    )

st.divider()


# =========================================================
# 11. OPERATIONAL RECOMMENDATIONS
# =========================================================

st.header("💡 Operational Recommendations")

if not delivered_df.empty:

    overall_late_rate = (
        100 * delivered_df["is_late"].mean()
    )

    st.write(
        f"Current late-delivery rate: "
        f"**{overall_late_rate:.2f}%**"
    )

    minimum_orders = 10

    carrier_issues = carrier_data[
        carrier_data["orders"] >= minimum_orders
    ].sort_values(
        "late_rate_percentage",
        ascending=False,
    )

    warehouse_issues = warehouse_data[
        warehouse_data["orders"] >= minimum_orders
    ].sort_values(
        "late_rate_percentage",
        ascending=False,
    )

    if not carrier_issues.empty:

        worst_carrier = carrier_issues.iloc[0]

        if (
            worst_carrier["late_rate_percentage"]
            > overall_late_rate
        ):

            st.warning(
                f"**Carrier review:** {worst_carrier['carrier']} "
                f"has a late-delivery rate of "
                f"{worst_carrier['late_rate_percentage']:.1f}%. "
                "Review dispatch timing, transit scans, and carrier "
                "handoff performance."
            )

    if not warehouse_issues.empty:

        worst_warehouse = warehouse_issues.iloc[0]

        if (
            worst_warehouse["late_rate_percentage"]
            > overall_late_rate
        ):

            st.warning(
                f"**Warehouse review:** {worst_warehouse['warehouse_id']} "
                f"has a late-delivery rate of "
                f"{worst_warehouse['late_rate_percentage']:.1f}%. "
                "Investigate packing queues, staffing, and dispatch "
                "cut-off times."
            )

    if overall_late_rate > 0:

        st.info(
            "Track the late-delivery rate weekly and investigate "
            "high-volume problem areas first."
        )

    else:

        st.success(
            "No late deliveries were identified in the selected "
            "completed-delivery records."
        )

else:

    st.info(
        "Recommendations require completed delivery records."
    )

st.divider()
# =========================================================
# 12. CARRIER PERFORMANCE LEADERBOARD
# =========================================================

st.header("🏆 Carrier Performance Leaderboard")

carrier_ranking = rank_carriers(filtered_df)

if not carrier_ranking.empty:

    st.write(
        "Carriers ranked by on-time delivery percentage."
    )

    st.dataframe(
        carrier_ranking,
        width="stretch",
        hide_index=True
    )

    best_carrier = carrier_ranking.iloc[0]

    st.success(
        f"Top-performing carrier: {best_carrier['carrier']} "
        f"with {best_carrier['on_time_percentage']:.1f}% "
        "on-time deliveries."
    )

    ranking_chart = px.bar(
        carrier_ranking,
        x="carrier",
        y="on_time_percentage",
        text="on_time_percentage",
        title="Carrier On-Time Performance Ranking",
        labels={
            "carrier": "Delivery Carrier",
            "on_time_percentage": "On-Time Delivery (%)"
        }
    )

    ranking_chart.update_traces(
        texttemplate="%{text:.1f}%"
    )

    ranking_chart.update_yaxes(range=[0, 100])

    st.plotly_chart(
        ranking_chart,
        width="stretch"
    )

else:

    st.info(
        "No completed deliveries available for carrier ranking."
    )

st.divider()
# =========================================================
# 12. ADVANCED DELAY FACTOR ANALYSIS
# =========================================================

st.header("🔬 Advanced Delay Factor Analysis")

st.write(
    "Identify operational categories associated with "
    "higher delivery delay rates."
)

delay_factors = analyze_delay_factors(
    filtered_df,
    min_orders=10
)

if not delay_factors.empty:

    st.subheader("Highest Delay Risk Factors")

    st.dataframe(
        delay_factors,
        width="stretch",
        hide_index=True
    )

    top_factors = delay_factors.head(10)

    factor_chart = px.bar(
        top_factors,
        x="late_rate_percentage",
        y="factor",
        color="category",
        orientation="h",
        title="Operational Categories with Highest Late Rates",
        labels={
            "late_rate_percentage": "Late Rate (%)",
            "factor": "Operational Factor",
            "category": "Category"
        }
    )

    st.plotly_chart(
        factor_chart,
        width="stretch"
    )

else:

    st.info(
        "Not enough completed delivery records for "
        "advanced delay factor analysis."
    )

st.divider()

# Operational Risk Alerts

st.header("🚨 Operational Risk Alerts")

alerts = generate_operational_alerts(filtered_df)

for alert in alerts:
    if "No major operational risks" in alert:
        st.success(alert)
    elif "No completed deliveries" in alert:
        st.info(alert)
    else:
        st.warning(alert)
# =========================================================
# 12. MACHINE-LEARNING DELAY RISK PREDICTION
# =========================================================

st.header("🤖 Delivery Delay Risk Prediction")

st.write(
    "Train a machine-learning model using historical delivery "
    "records and estimate the delay risk for a new delivery."
)

st.caption(
    "Predictions are estimates based on the available data. "
    "The included sample dataset contains simulated operational records."
)

with st.expander("How does this model work?"):

    st.markdown(
        """
        **Algorithm:** Decision Tree Classifier

        **Input features:**
        - Warehouse
        - Carrier
        - Route type
        - Product category
        - Day of the week
        - Season
        - Customer region

        **Target:** Late delivery or on-time delivery.

        **Evaluation:** Accuracy, precision, recall, F1-score,
        and ROC-AUC where available.

        The model uses completed deliveries and evaluates performance
        on a held-out test dataset.
        """
    )

if st.button(
    "🚀 Train / Retrain Delay Prediction Model",
    key="train_delay_model_button",
):

    with st.spinner("Training the machine-learning model..."):

        model_result = train_delay_model(filtered_df)

    if "error" in model_result:

        st.session_state.pop(
            "delivery_model_result",
            None,
        )

        st.error(model_result["error"])

    else:

        st.session_state["delivery_model_result"] = model_result

        st.success("Model trained successfully.")


model_result = st.session_state.get(
    "delivery_model_result"
)

if model_result is None:

    st.info(
        "Click 'Train / Retrain Delay Prediction Model' to train "
        "the model and enable the prediction form."
    )

else:

    metrics = model_result["metrics"]

    st.subheader("📊 Model Evaluation")

    st.caption(
        f"Training records: "
        f"{model_result['training_records']:,} | "
        f"Testing records: "
        f"{model_result['testing_records']:,}"
    )

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.1f}%",
    )

    metric_columns[1].metric(
        "Precision",
        f"{metrics['precision'] * 100:.1f}%",
    )

    metric_columns[2].metric(
        "Recall",
        f"{metrics['recall'] * 100:.1f}%",
    )

    metric_columns[3].metric(
        "F1 Score",
        f"{metrics['f1']:.3f}",
    )

    if metrics["roc_auc"] is not None:

        st.metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.3f}",
        )

    st.caption(
        "These metrics are calculated on a held-out test set. "
        "Performance on future deliveries may differ."
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    st.subheader("🔍 Important Operational Factors")

    importance_df = model_result["importance"]

    if not importance_df.empty:

        importance_chart = px.bar(
            importance_df.sort_values(
                "importance",
                ascending=True,
            ),
            x="importance",
            y="feature",
            orientation="h",
            title="Decision Tree Feature Importance",
            labels={
                "importance": "Feature Importance",
                "feature": "Operational Feature",
            },
        )

        st.plotly_chart(
            importance_chart,
            use_container_width=True,
        )

    else:

        st.info(
            "The model did not identify positive feature importance."
        )

    # -----------------------------------------------------
    # NEW DELIVERY PREDICTION FORM
    # -----------------------------------------------------

    st.subheader("🎯 Predict a New Delivery's Risk")

    feature_columns = [
        "warehouse_id",
        "carrier",
        "route_type",
        "product_type",
        "day_of_week",
        "season",
        "customer_region",
    ]

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:

        st.error(
            f"The dataset is missing required model columns: "
            f"{', '.join(missing_columns)}"
        )

    else:

        prediction_data = df.copy()

        def get_options(column):

            values = (
                prediction_data[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            return sorted(values) if values else ["Unknown"]

        with st.form("delivery_risk_prediction_form"):

            form_left, form_right = st.columns(2)

            with form_left:

                selected_warehouse = st.selectbox(
                    "Warehouse",
                    get_options("warehouse_id"),
                )

                selected_carrier = st.selectbox(
                    "Delivery Carrier",
                    get_options("carrier"),
                )

                selected_route = st.selectbox(
                    "Route Type",
                    get_options("route_type"),
                )

                selected_product = st.selectbox(
                    "Product Category",
                    get_options("product_type"),
                )

            with form_right:

                selected_day = st.selectbox(
                    "Day of Week",
                    get_options("day_of_week"),
                )

                selected_season = st.selectbox(
                    "Season",
                    get_options("season"),
                )

                selected_region = st.selectbox(
                    "Customer Region",
                    get_options("customer_region"),
                )

            predict_clicked = st.form_submit_button(
                "🔮 Predict Delivery Risk",
                use_container_width=True,
            )

        if predict_clicked:

            input_data = pd.DataFrame(
                [
                    {
                        "warehouse_id": selected_warehouse,
                        "carrier": selected_carrier,
                        "route_type": selected_route,
                        "product_type": selected_product,
                        "day_of_week": selected_day,
                        "season": selected_season,
                        "customer_region": selected_region,
                    }
                ],
                columns=feature_columns,
            )

            try:

                trained_model = model_result["model"]

                probabilities = trained_model.predict_proba(
                    input_data
                )[0]

                classifier = trained_model.named_steps[
                    "classifier"
                ]

                model_classes = list(classifier.classes_)

                if 1 in model_classes:

                    late_probability = float(
                        probabilities[model_classes.index(1)]
                    )

                else:

                    late_probability = 0.0

                st.subheader("📋 Prediction Result")

                result_left, result_right = st.columns(2)

                result_left.metric(
                    "Estimated Late-Delivery Probability",
                    f"{late_probability * 100:.1f}%",
                )

                if late_probability < 0.30:

                    result_right.success("LOW ESTIMATED RISK")

                    suggested_action = (
                        "Continue standard monitoring and confirm "
                        "delivery milestones."
                    )

                elif late_probability < 0.60:

                    result_right.warning("MEDIUM ESTIMATED RISK")

                    suggested_action = (
                        "Monitor shipment progress and verify that "
                        "the carrier collects the shipment on schedule."
                    )

                else:

                    result_right.error("HIGH ESTIMATED RISK")

                    suggested_action = (
                        "Review dispatch timing, carrier service levels, "
                        "and route planning before confirming the schedule."
                    )

                st.progress(late_probability)

                st.info(
                    f"Suggested action: {suggested_action}"
                )

                st.caption(
                    "Risk bands are illustrative thresholds, not "
                    "calibrated business categories. Validate the model "
                    "with real historical data before operational use."
                )

            except Exception as error:

                st.error(
                    f"Unable to generate the prediction: {error}"
                )

st.divider()

# Business Performance Report

st.header("📥 Business Performance Report")

report_columns = [
    "order_id",
    "carrier",
    "warehouse_id",
    "is_delivered",
    "delay_days",
    "is_late",
]

available_columns = [
    column for column in report_columns
    if column in filtered_df.columns
]

business_report = filtered_df[available_columns].copy()

st.download_button(
    label="Download Business Performance Report",
    data=business_report.to_csv(index=False).encode("utf-8"),
    file_name="delivery_performance_report.csv",
    mime="text/csv",
)
# =========================================================
# 13. ORDER DATA EXPLORER
# =========================================================

st.header("🔍 Order Data Explorer")

display_columns = [
    "order_id",
    "order_date",
    "warehouse_id",
    "carrier",
    "route_type",
    "product_type",
    "promised_delivery_date",
    "actual_delivery_date",
    "transit_days",
    "delay_days",
    "is_delivered",
    "is_late",
]

available_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[available_columns].sort_values(
        "order_date",
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# 14. DOWNLOAD REPORTS
# =========================================================

st.subheader("📥 Download Reports")

download_left, download_right = st.columns(2)

with download_left:

    filtered_csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Dataset",
        data=filtered_csv,
        file_name="delivery_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_right:

    late_df = filtered_df[
        filtered_df["is_late"]
    ].copy()

    late_csv = late_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Late Orders",
        data=late_csv,
        file_name="late_orders.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# 15. FOOTER
# =========================================================

st.divider()

st.caption(
    "Delivery Delay & Operations Analytics | "
    "Python • Pandas • Plotly • Streamlit • Scikit-learn"
)

st.caption(
    "Note: Results depend on the quality of the supplied data. "
    "The included sample data is simulated and does not represent "
    "an actual logistics company."
)