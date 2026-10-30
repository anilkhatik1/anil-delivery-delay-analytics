# Delivery Delay & Operations Analytics

A Python-based analytics dashboard designed to monitor delivery performance, identify operational bottlenecks, analyze delivery delays, and support data-driven decisions.

## Project Overview

This project analyzes delivery operations using shipment, order, warehouse, carrier, and delivery event data. It provides interactive visualizations, operational alerts, performance rankings, and machine learning-based delay prediction.

## Key Features

* **Executive Summary:** Automatically summarizes delivery performance.
* **Delivery KPIs:** Tracks total orders, completed deliveries, late deliveries, and on-time performance.
* **Carrier Performance Leaderboard:** Compares carriers using delivery metrics.
* **Warehouse Efficiency Leaderboard:** Identifies warehouse performance differences.
* **Delay Factor Analysis:** Examines factors associated with delivery delays.
* **Operational Risk Alerts:** Highlights carriers and warehouses with high late-delivery rates.
* **Machine Learning:** Uses a Decision Tree classifier to predict delivery delays.
* **Interactive Filters:** Filters results by warehouse, carrier, route, and product category.
* **Business Reports:** Exports filtered delivery data as CSV files.

## Technology Stack

* Python
* Streamlit
* Pandas
* Plotly
* Scikit-learn
* MySQL (only if implemented; otherwise remove this item)
* CSV datasets

## Project Structure

```text
delivery-delay-operations-analytics/
│
├── app.py
├── generate_data.py
├── requirement.txt
├── orders.csv
├── shipments.csv
├── warehouses.csv
├── carriers.csv
├── delivery_events.csv
│
└── src/
    └── analytics.py
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/anilkhatik1/anil-delivery-delay-analytics.git
cd anil-delivery-delay-analytics
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirement.txt
```

### 5. Generate sample data

```bash
python generate_data.py
```

### 6. Run the dashboard

```bash
streamlit run app.py
```

## Machine Learning

The project uses a Decision Tree Classifier to estimate whether a completed delivery will be late based on operational characteristics such as carrier, warehouse, route, product category, season, and customer region.

Model performance is evaluated using classification metrics including accuracy, precision, recall, F1-score, and ROC-AUC where applicable.

## Dataset Information

The project uses simulated delivery data generated through `generate_data.py`. It includes:

* Orders
* Shipments
* Warehouses
* Carriers
* Delivery Events

The results demonstrate analytical functionality and should not be interpreted as actual business performance.

## Future Enhancements

* Live database integration
* Real-time shipment tracking
* Advanced predictive models
* Automated email alerts
* Interactive maps and route analysis
* Cloud deployment

## Author

**Anil Khatik**

B.Tech – Computer Science and Engineering (AI & Data Science)

[LinkedIn](https://www.linkedin.com/in/anil-khatik-8449a0320)

[GitHub](https://github.com/anilkhatik1)

---

*Built with Python, Streamlit, Pandas, Plotly, and Scikit-learn.*
