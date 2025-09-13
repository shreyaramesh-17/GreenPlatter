import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import json

# App Config
st.set_page_config(
    page_title="GreenPlatter — Sustainable Meal Management",
    page_icon="🥗",
    layout="wide",
)

# Paths
DATA_PATH = Path("Expanded_GreenPlatter_12000.csv")
MODEL_PATH = Path("greenplatter_pipeline.joblib")
CATEGORIES_PATH = Path("greenplatter_categories.json")


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure types
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    return df


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_categories(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_kpi(label: str, value):
    st.metric(label, value)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    min_date = pd.to_datetime(df["Date"]).min()
    max_date = pd.to_datetime(df["Date"]).max()
    default_start = min_date.date() if pd.notna(min_date) else None
    default_end = max_date.date() if pd.notna(max_date) else None
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, default_end) if default_start and default_end else None,
    )

    def options_for(column_name: str):
        series = pd.Series(df[column_name])
        values = series.dropna().astype(str).unique().tolist()
        return sorted(values)

    day = st.sidebar.multiselect("Day of Week", options=options_for("DayOfWeek"))
    event = st.sidebar.multiselect("Event", options=options_for("Event"))
    weather = st.sidebar.multiselect("Weather", options=options_for("Weather"))
    dish = st.sidebar.multiselect("Dish", options=options_for("Dish"))

    df_f = df.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and date_range[0] and date_range[1]:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_f = df_f[(df_f["Date"] >= start) & (df_f["Date"] <= end)]
    if day:
        df_f = df_f[df_f["DayOfWeek"].isin(day)]
    if event:
        df_f = df_f[df_f["Event"].isin(event)]
    if weather:
        df_f = df_f[df_f["Weather"].isin(weather)]
    if dish:
        df_f = df_f[df_f["Dish"].isin(dish)]
    return df_f


def render_overview(df: pd.DataFrame):
    st.subheader("Overview")
    total_prepared = int(df["Prepared_Qty"].sum())
    total_sold = int(df["Sold_Qty"].sum())
    total_waste = int(df["Waste_Qty"].sum())
    waste_rate = (total_waste / total_prepared * 100) if total_prepared else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        format_kpi("Total Prepared", total_prepared)
    with c2:
        format_kpi("Total Sold", total_sold)
    with c3:
        format_kpi("Total Waste", total_waste)
    with c4:
        format_kpi("Waste Rate", f"{waste_rate:.1f}%")

    st.divider()
    # Time series by date (aggregated)
    daily = (
        df.groupby("Date", as_index=False)[["Prepared_Qty", "Sold_Qty", "Waste_Qty"]].sum()
    )
    st.line_chart(daily.set_index("Date"))

    st.caption("Waste by Dish")
    waste_by_dish = df.groupby("Dish", as_index=False)["Waste_Qty"].sum().sort_values("Waste_Qty", ascending=False)
    st.bar_chart(waste_by_dish.set_index("Dish"))


def render_predict(model):
    st.subheader("Predict Demand")
    if model is None:
        st.warning("Model file not found. Place 'greenplatter_pipeline.joblib' in the app folder.")
        return

    categories = load_categories(CATEGORIES_PATH)
    if categories is None:
        st.warning("Category file not found. Place 'greenplatter_categories.json' in the app folder.")
        return

    c1, c2 = st.columns(2)
    with c1:
        day = st.selectbox("Day of Week", categories.get("DayOfWeek", []))
        guests = st.number_input("Expected Guests", min_value=50, max_value=10000, value=150, step=10)
        event = st.selectbox("Event Type", categories.get("Event", []))
    with c2:
        weather = st.selectbox("Weather", categories.get("Weather", []))

    if st.button("Predict Demand", use_container_width=True):
        rows = []
        dishes = categories.get("Dish", [])
        # Build a dataframe for all dishes and shared inputs; pipeline handles encoding
        input_df = pd.DataFrame([
            {"DayOfWeek": day, "Guests": guests, "Event": event, "Weather": weather, "Dish": dsh}
            for dsh in dishes
        ])
        preds = model.predict(input_df)
        for dsh, pred in zip(dishes, preds):
            pred = float(pred)
            optimal = int(pred * 1.05)

            # Per-dish recommendation based on predicted demand
            demand = pred
            if demand <= 5:
                rec = "Very low: cook-to-order only"
            elif demand <= 30:
                first = max(5, int(round(demand * 0.5)))
                hold = max(0, int(round(demand - first)))
                rec = f"start {first}, hold {hold}"
            elif demand <= 80:
                first = int(round(demand * 0.6))
                hold = max(0, int(round(demand - first)))
                rec = f"start {first}, hold {hold}"
            elif demand <= 150:
                first = int(round(demand * 0.7))
                hold = max(0, int(round(demand - first)))
                rec = f"start {first}, hold {hold}"
            else:
                first = int(round(demand * 0.7))
                hold = max(0, int(round(demand - first)))
                rec = f"start {first}, hold {hold}"

            rows.append({
                "Dish": dsh,
                "Predicted": round(pred, 1),
                "SuggestPrep": optimal,
                "Recommendation": rec,
            })

        st.success(f"Generated predictions for {len(rows)} dishes")
        pred_df = pd.DataFrame(rows).sort_values(["Predicted"], ascending=[False])
        st.dataframe(pred_df, use_container_width=True)
        csv = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download daily dish predictions (CSV)", csv, file_name="daily_dish_predictions.csv")


def render_recommendations(df: pd.DataFrame):
    st.subheader("Recommendations")
    # Simple heuristics for demo
    high_waste = df.groupby("Dish", as_index=False)["Waste_Qty"].mean().sort_values("Waste_Qty", ascending=False)
    top_waste_dishes = high_waste.head(3)["Dish"].tolist()
    if top_waste_dishes:
        st.write(
            "- Reduce prep for high-waste dishes: " + ", ".join(top_waste_dishes)
        )
    rainy_boost = df[df["Weather"] == "Rainy"].groupby("Dish", as_index=False)["Sold_Qty"].mean().sort_values("Sold_Qty", ascending=False).head(3)["Dish"].tolist()
    if rainy_boost:
        st.write("- On rainy days, prioritize: " + ", ".join(rainy_boost))

    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data (CSV)", csv_export, file_name="greenplatter_filtered.csv")


def main():
    st.markdown("""
        <style>
            .glow-card {background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;box-shadow:0 1px 2px rgba(16,24,40,0.04),0 1px 3px rgba(16,24,40,0.08)}
            .gp-header {display:flex;align-items:center;gap:12px;margin-bottom:4px}
            .gp-subtle {color:#6b7280;margin:0}
        </style>
        <div class="gp-header">
            <span style="font-size:28px;">🥗</span>
            <h1 style="margin:0;">GreenPlatter</h1>
        </div>
        <p class="gp-subtle">Sustainable hotel meal management with AI</p>
    """, unsafe_allow_html=True)

    df = load_data(DATA_PATH)
    df_filtered = sidebar_filters(df)
    model = load_model(MODEL_PATH)

    with st.container():
        render_overview(df_filtered)
    st.divider()
    with st.container():
        render_predict(model)
    st.divider()
    with st.container():
        render_recommendations(df_filtered)


if __name__ == "__main__":
    main()