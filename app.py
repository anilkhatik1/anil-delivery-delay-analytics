from pathlib import Path
from io import BytesIO
from src.analytics import (
    load_data,
    prepare_data,
    calculate_kpis,
    analyze_segment,
    train_delay_model,
    validate_tables,
)
import pandas as pd
import plotly.express as px
import streamlit as st
# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Delivery Operations Analytics",
    page_icon="📦",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# --------------------------------------------------
# 2. CSV UPLOAD AND DATA VALIDATION
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader("📂 Data Source")

st.sidebar.write(
    "Upload all five operational CSV files to analyze "
    "another dataset. Leave them empty to use the sample data."
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
# LOAD DATA
# --------------------------------------------------
uploaded_count = sum(
    file is not None
    for file in uploaded_files.values()
)
try:

    if uploaded_count == 0:

        # Use the existing sample CSV files.
        tables = load_data(BASE_DIR)

        st.info(
            "Using the generated sample dataset. "
            "Upload all five CSV files in the sidebar "
            "to analyze another dataset."
        )

    elif uploaded_count < len(upload_config):

        st.warning(
            f"You uploaded {uploaded_count} of 5 required files. "
            "Upload all five files to begin analysis."
        )

        st.stop()

    else:

        # Read uploaded CSVs without saving them to disk.
        tables = {}

        for table_name, uploaded_file in uploaded_files.items():

            uploaded_file.seek(0)

            tables[table_name] = pd.read_csv(
                BytesIO(uploaded_file.getvalue())
            )

        st.success(
            "All five CSV files have been uploaded."
        )

    # --------------------------------------------------
    # VALIDATE DATA
    # --------------------------------------------------

    errors, warnings = validate_tables(tables)

    st.subheader("🧪 Data Quality Report")

    if errors:

        st.error(
            f"Validation failed with {len(errors)} error(s). "
            "Correct the CSV files before continuing."
        )

        for error in errors:
            st.write(f"❌ {error}")

        for warning in warnings:
            st.warning(warning)

        st.stop()

    st.success(
        "Required columns and basic data checks passed."
    )

    if warnings:

        with st.expander(
            f"Review {len(warnings)} data-quality warning(s)"
        ):

            for warning in warnings:
                st.warning(warning)

    else:

        st.caption(
            "No additional data-quality warnings were detected."
        )

    # --------------------------------------------------
    # PREPARE ANALYTICAL DATASET
    # --------------------------------------------------

    df = prepare_data(tables)

    if df.empty:

        st.error(
            "No valid orders remain after data preparation."
        )

        st.stop()

    st.metric(
        "Orders Available for Analysis",
        f"{len(df):,}",
    )

except Exception as error:

    st.error(
        f"Unable to load or process the dataset: {error}"
    )

    st.info(
        "Check the CSV structure, column names, and data types. "
        "If using sample data, run generate_data.py first."
    )

    st.stop()

# --------------------------------------------------
# 3. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.write("Filter the data for detailed analysis.")

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
    "Select Warehouse",
    warehouse_options,
    default=warehouse_options,
)

selected_carriers = st.sidebar.multiselect(
    "Select Carrier",
    carrier_options,
    default=carrier_options,
)

selected_routes = st.sidebar.multiselect(
    "Select Route",
    route_options,
    default=route_options,
)

selected_products = st.sidebar.multiselect(
    "Select Product Category",
    product_options,
    default=product_options,
)


# --------------------------------------------------
# 4. FILTER DATA
# --------------------------------------------------

filtered_df = df[
    df["warehouse_id"].isin(selected_warehouses)
    & df["carrier"].isin(selected_carriers)
    & df["route_type"].isin(selected_routes)
    & df["product_type"].isin(selected_products)
].copy()


if filtered_df.empty:
    st.warning(
        "No records match the selected filters. "
        "Please change your filter selections."
    )
    st.stop()
# --------------------------------------------------
# MACHINE LEARNING: DELIVERY DELAY RISK PREDICTION
# --------------------------------------------------

st.divider()

st.subheader("🤖 Delivery Delay Risk Prediction")

st.write(
    "Train a machine-learning model using historical delivery "
    "records and estimate the delay risk for a new delivery."
)

st.caption(
    "Predictions are estimates based on the available sample data. "
    "The current dataset contains simulated operational records."
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

        The model uses completed deliveries for training and testing.
        """
    )


# --------------------------------------------------
# 1. TRAIN MODEL
# --------------------------------------------------

if st.button(
    "🚀 Train / Retrain Delay Prediction Model",
    key="train_delay_model_button",
):

    with st.spinner("Training the machine-learning model..."):

        result = train_delay_model(filtered_df)

    if "error" in result:

        st.session_state.pop("delivery_model_result", None)

        st.error(result["error"])

    else:

        # Store the trained model across Streamlit reruns.
        st.session_state["delivery_model_result"] = result

        st.success(
            "Model trained successfully on the selected dataset."
        )


# --------------------------------------------------
# 2. DISPLAY MODEL PERFORMANCE
# --------------------------------------------------

model_result = st.session_state.get(
    "delivery_model_result"
)

if model_result is None:

    st.info(
        "First click 'Train / Retrain Delay Prediction Model'. "
        "The prediction form will become available after training."
    )

else:

    metrics = model_result["metrics"]

    st.markdown("### 📊 Model Evaluation")

    st.caption(
        f"Training records: "
        f"{model_result['training_records']:,} | "
        f"Testing records: "
        f"{model_result['testing_records']:,}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.1f}%"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.1f}%"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.1f}%"
    )

    col4.metric(
        "F1 Score",
        f"{metrics['f1']:.3f}"
    )

    if metrics["roc_auc"] is not None:

        st.metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.3f}"
        )

    st.caption(
        "These metrics evaluate the model on a held-out test set. "
        "Actual performance on future deliveries may differ."
    )

    # Feature importance chart
    st.markdown("### 🔍 Important Operational Factors")

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
            "The trained model did not identify any positive "
            "feature importance."
        )


    # --------------------------------------------------
    # 3. DELIVERY RISK PREDICTION FORM
    # --------------------------------------------------

    st.divider()

    st.markdown("### 🎯 Predict a New Delivery's Risk")

    st.write(
        "Select the operational details below and click "
        "'Predict Delivery Risk'."
    )

    # The model uses these seven features.
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
            f"Missing required dataset columns: {missing_columns}"
        )

    else:

        # Use the full prepared dataset for available input options.
        # This avoids losing options when dashboard filters change.
        prediction_form_data = df.copy()

        def get_options(column):

            values = (
                prediction_form_data[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            return sorted(values) if values else ["Unknown"]

        with st.form("delivery_risk_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:

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

            with col2:

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

            submitted = st.form_submit_button(
                "🔮 Predict Delivery Risk",
                use_container_width=True,
            )

        if submitted:

            # Construct a single input record.
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

                # Probability of the late-delivery class.
                probabilities = trained_model.predict_proba(
                    input_data
                )[0]

                classifier = trained_model.named_steps[
                    "classifier"
                ]

                classes = list(classifier.classes_)

                if 1 not in classes:

                    late_probability = 0.0

                else:

                    late_probability = float(
                        probabilities[classes.index(1)]
                    )

                # Display the estimated risk.
                st.divider()

                st.markdown("### 📋 Prediction Result")

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.metric(
                        "Estimated Late-Delivery Probability",
                        f"{late_probability * 100:.1f}%",
                    )

                with result_col2:

                    if late_probability < 0.30:

                        st.success("LOW ESTIMATED RISK")

                    elif late_probability < 0.60:

                        st.warning("MEDIUM ESTIMATED RISK")

                    else:

                        st.error("HIGH ESTIMATED RISK")

                st.progress(late_probability)

                st.write(
                    "The estimated probability represents the "
                    "model's assessment of a late delivery for "
                    "the selected operational characteristics."
                )

                if late_probability >= 0.60:

                    st.warning(
                        "Suggested action: review dispatch timing, "
                        "carrier service levels, and route planning "
                        "before committing to the delivery schedule."
                    )

                elif late_probability >= 0.30:

                    st.info(
                        "Suggested action: monitor shipment progress "
                        "and verify that the carrier collects the "
                        "shipment on schedule."
                    )

                else:

                    st.info(
                        "Suggested action: continue standard "
                        "monitoring and confirm delivery milestones."
                    )

                st.caption(
                    "The LOW, MEDIUM, and HIGH categories are "
                    "illustrative thresholds, not calibrated business "
                    "risk categories. Validate the model using real "
                    "historical data before operational use."
                )

            except Exception as error:

                st.error(
                    f"Unable to generate the prediction: {error}"
                )

# --------------------------------------------------
# 5. CALCULATE KPIs
# --------------------------------------------------

kpis = calculate_kpis(filtered_df)

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Orders",
    f"{kpis['total_orders']:,}",
)

col2.metric(
    "Delivered Orders",
    f"{kpis['delivered_orders']:,}",
)

col3.metric(
    "Late Orders",
    f"{kpis['late_orders']:,}",
)

col4.metric(
    "On-Time Delivery",
    f"{kpis['on_time_percentage']:.1f}%",
)

col5.metric(
    "Average Late Delay",
    f"{kpis['average_delay_days']:.1f} days",
)

st.divider()


# --------------------------------------------------
# 6. MONTHLY DELIVERY TREND
# --------------------------------------------------

st.subheader("📈 Delivery Performance Trends")

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

    fig = px.line(
        monthly_data,
        x="delivery_month",
        y="on_time_percentage",
        markers=True,
        title="Monthly On-Time Delivery Percentage",
        labels={
            "delivery_month": "Month",
            "on_time_percentage": "On-Time Delivery (%)",
        },
    )

    fig.update_yaxes(range=[0, 100])

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:
    st.info("No completed deliveries are available for trend analysis.")


st.divider()


# --------------------------------------------------
# 7. CARRIER PERFORMANCE
# --------------------------------------------------

st.subheader("🚚 Carrier Performance Analysis")

carrier_data = analyze_segment(
    filtered_df,
    "carrier",
)

left, right = st.columns(2)

with left:

    fig = px.bar(
        carrier_data,
        x="carrier",
        y="late_rate_percentage",
        title="Late Delivery Rate by Carrier",
        labels={
            "carrier": "Delivery Carrier",
            "late_rate_percentage": "Late Rate (%)",
        },
        text="late_rate_percentage",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with right:

    fig = px.bar(
        carrier_data,
        x="carrier",
        y="average_delay_days",
        title="Average Delay by Carrier",
        labels={
            "carrier": "Delivery Carrier",
            "average_delay_days": "Average Delay (Days)",
        },
        text="average_delay_days",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

st.markdown("#### Carrier Performance Details")

st.dataframe(
    carrier_data,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# --------------------------------------------------
# 8. WAREHOUSE BOTTLENECK ANALYSIS
# --------------------------------------------------

st.subheader("🏭 Warehouse Bottleneck Analysis")

warehouse_data = analyze_segment(
    filtered_df,
    "warehouse_id",
)

fig = px.bar(
    warehouse_data,
    x="warehouse_id",
    y="late_rate_percentage",
    title="Late Delivery Rate by Warehouse",
    labels={
        "warehouse_id": "Warehouse",
        "late_rate_percentage": "Late Rate (%)",
    },
    text="late_rate_percentage",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.dataframe(
    warehouse_data,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# --------------------------------------------------
# 9. ROUTE ANALYSIS
# --------------------------------------------------

st.subheader("🛣️ Route Performance")

route_data = analyze_segment(
    filtered_df,
    "route_type",
)

fig = px.bar(
    route_data,
    x="route_type",
    y="late_rate_percentage",
    title="Late Deliveries by Route Type",
    labels={
        "route_type": "Route Type",
        "late_rate_percentage": "Late Rate (%)",
    },
    text="late_rate_percentage",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.dataframe(
    route_data,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# --------------------------------------------------
# 10. PRODUCT CATEGORY ANALYSIS
# --------------------------------------------------

st.subheader("📦 Product Category Analysis")

product_data = analyze_segment(
    filtered_df,
    "product_type",
)

fig = px.bar(
    product_data,
    x="product_type",
    y="late_rate_percentage",
    title="Delivery Delays by Product Category",
    labels={
        "product_type": "Product Category",
        "late_rate_percentage": "Late Rate (%)",
    },
    text="late_rate_percentage",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.dataframe(
    product_data,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# --------------------------------------------------
# 11. OPERATIONAL RECOMMENDATIONS
# --------------------------------------------------

st.subheader("💡 Operational Recommendations")

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
    ]

    warehouse_issues = warehouse_data[
        warehouse_data["orders"] >= minimum_orders
    ]

    if not carrier_issues.empty:

        worst_carrier = carrier_issues.iloc[0]

        if (
            worst_carrier["late_rate_percentage"]
            > overall_late_rate
        ):

            st.warning(
                f"**Carrier review:** "
                f"{worst_carrier['carrier']} has a late rate of "
                f"{worst_carrier['late_rate_percentage']:.1f}%. "
                f"Review dispatch timing, transit scans, and "
                f"carrier hand-off performance."
            )

    if not warehouse_issues.empty:

        worst_warehouse = warehouse_issues.iloc[0]

        if (
            worst_warehouse["late_rate_percentage"]
            > overall_late_rate
        ):

            st.warning(
                f"**Warehouse review:** "
                f"{worst_warehouse['warehouse_id']} has a late rate "
                f"of {worst_warehouse['late_rate_percentage']:.1f}%. "
                f"Investigate packing queues, staffing, and "
                f"dispatch cut-off times."
            )

    if overall_late_rate > 0:

        st.info(
            "Track late-delivery rate weekly and investigate "
            "the highest-volume problem areas first."
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


# --------------------------------------------------
# 12. ORDER DATA EXPLORER
# --------------------------------------------------

st.subheader("🔍 Order Data Explorer")

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

# Download the filtered dataset.
csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Dataset",
    data=csv_data,
    file_name="delivery_analysis.csv",
    mime="text/csv",
)

# Download only late deliveries.
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
)


# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Delivery Delay & Operations Analytics | "
    "Python • Pandas • Plotly • Streamlit"
)

st.caption(
    "Note: Results depend on the quality of the supplied data. "
    "Sample data is simulated and does not represent any actual "
    "logistics company."
)