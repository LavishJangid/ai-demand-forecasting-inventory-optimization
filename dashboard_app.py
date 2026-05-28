from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


APP_TITLE = "Food Demand Forecasting Dashboard"
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "model_ready_data.csv"
MODEL_PATH = ROOT / "models" / "xgboost_model.pkl"


st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(50, 90, 160, 0.22), transparent 30%),
                    radial-gradient(circle at top right, rgba(30, 130, 90, 0.14), transparent 24%),
                    #0b0d12;
                color: #f5f7fb;
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #f7f9fe !important;
                letter-spacing: -0.03em;
            }
            [data-testid="stSidebar"] {
                background: #0e1117;
            }
            [data-testid="stMetric"] {
                background: linear-gradient(180deg, rgba(24, 28, 38, 0.92), rgba(18, 22, 31, 0.96));
                border: 1px solid rgba(255, 255, 255, 0.07);
                border-radius: 18px;
                padding: 18px 18px 12px 18px;
                box-shadow: 0 18px 50px rgba(0, 0, 0, 0.26);
            }
            [data-testid="metric-container"] {
                gap: 0.25rem;
            }
            [data-testid="metric-container"] label {
                color: rgba(240, 243, 255, 0.7) !important;
                font-size: 0.98rem !important;
            }
            [data-testid="stButton"] button {
                background: linear-gradient(180deg, #1c2431, #121823);
                color: #f7f9fe;
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 14px;
                padding: 0.75rem 1.2rem;
                font-weight: 700;
            }
            [data-testid="stButton"] button:hover {
                border-color: rgba(120, 168, 255, 0.45);
                box-shadow: 0 0 0 1px rgba(120, 168, 255, 0.10), 0 10px 24px rgba(0, 0, 0, 0.22);
            }
            [data-testid="stSlider"] {
                padding-top: 0.4rem;
            }
            .hero {
                background: linear-gradient(135deg, rgba(24, 29, 40, 0.96), rgba(13, 16, 22, 0.96));
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 28px;
                padding: 1.5rem 1.6rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 24px 70px rgba(0, 0, 0, 0.30);
            }
            .hero h1 {
                font-size: clamp(2rem, 4vw, 3.5rem);
                margin: 0;
                line-height: 1.05;
            }
            .hero p {
                margin: 0.6rem 0 0 0;
                color: rgba(235, 239, 255, 0.72);
                font-size: 1.02rem;
            }
            .section-label {
                display: inline-flex;
                align-items: center;
                gap: 0.65rem;
                margin: 1.2rem 0 0.85rem 0;
                font-size: 1.75rem;
                font-weight: 800;
                color: #f7f9fe;
            }
            .soft-card {
                background: linear-gradient(180deg, rgba(30, 39, 56, 0.98), rgba(20, 29, 44, 0.98));
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 18px;
                padding: 1.25rem 1.35rem;
                box-shadow: 0 16px 42px rgba(0, 0, 0, 0.22);
            }
            .status-good {
                background: linear-gradient(90deg, rgba(26, 78, 54, 0.92), rgba(21, 64, 45, 0.92));
                color: #66f0a1;
                border: 1px solid rgba(102, 240, 161, 0.15);
            }
            .status-warn {
                background: linear-gradient(90deg, rgba(86, 48, 22, 0.92), rgba(62, 36, 18, 0.92));
                color: #ffce8a;
                border: 1px solid rgba(255, 206, 138, 0.12);
            }
            .status-banner {
                border-radius: 16px;
                padding: 1rem 1.1rem;
                margin: 1rem 0 1.25rem 0;
                font-weight: 700;
                font-size: 1.05rem;
            }
            .mini-note {
                color: rgba(233, 238, 255, 0.68);
                font-size: 0.95rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(icon: str, title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<div class='mini-note'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-label">{icon}<span>{title}</span></div>
        {subtitle_html}
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    if MODEL_PATH.exists():
        saved = joblib.load(MODEL_PATH)
        if isinstance(saved, dict):
            return saved.get("model"), saved.get("features")
        return saved, getattr(saved, "feature_names_in_", None)
    return None, None


@st.cache_resource(show_spinner=False)
def train_fallback_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.01,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    return model


def select_feature_columns(df: pd.DataFrame, model_features) -> list[str]:
    if model_features is not None and len(model_features):
        return [c for c in model_features if c in df.columns]
    return [c for c in df.columns if c not in {"date", "quantity"}]


def std_for_window(values: list[float], window: int) -> float:
    if len(values) < 2:
        return 0.0
    window_values = values[-window:] if len(values) >= window else values
    if len(window_values) < 2:
        return 0.0
    return float(np.std(window_values, ddof=1))


def build_feature_row(next_date: pd.Timestamp, history: pd.DataFrame, trend_value: int) -> dict[str, float]:
    values = history["quantity"].tolist()
    row = {
        "date": next_date,
        "quantity": np.nan,
        "day": int(next_date.day),
        "month": int(next_date.month),
        "year": int(next_date.year),
        "day_of_week": int(next_date.dayofweek),
        "week_of_year": int(next_date.isocalendar().week),
        "is_weekend": int(next_date.dayofweek >= 5),
        "day_of_week_sin": float(np.sin(2 * np.pi * next_date.dayofweek / 7)),
        "day_of_week_cos": float(np.cos(2 * np.pi * next_date.dayofweek / 7)),
        "month_sin": float(np.sin(2 * np.pi * next_date.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * next_date.month / 12)),
        "trend": int(trend_value),
        "lag_1": float(values[-1]),
        "lag_3": float(values[-3] if len(values) >= 3 else values[-1]),
        "lag_7": float(values[-7] if len(values) >= 7 else values[-1]),
        "lag_14": float(values[-14] if len(values) >= 14 else values[-1]),
        "lag_21": float(values[-21] if len(values) >= 21 else values[-1]),
        "rolling_mean_3": float(np.mean(values[-3:])),
        "rolling_std_3": std_for_window(values, 3),
        "rolling_mean_7": float(np.mean(values[-7:])),
        "rolling_std_7": std_for_window(values, 7),
        "rolling_mean_14": float(np.mean(values[-14:])),
        "rolling_std_14": std_for_window(values, 14),
    }
    return row


def evaluate_model(model, df: pd.DataFrame, feature_cols: list[str]) -> dict[str, object]:
    y = df["quantity"].copy()
    X = df[feature_cols].copy()

    split_idx = int(len(df) * 0.80)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if model is None:
        model = train_fallback_model(X_train, y_train)

    pred = model.predict(X_test)
    naive_pred = X_test["lag_1"].values if "lag_1" in X_test.columns else np.repeat(y_train.iloc[-1], len(y_test))

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    baseline_mae = float(mean_absolute_error(y_test, naive_pred))

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "pred": pred,
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "train_size": split_idx,
    }


def forecast_future(model, history: pd.DataFrame, feature_cols: list[str], days: int = 7):
    working = history.copy().reset_index(drop=True)
    trend_start = int(working["trend"].iloc[-1]) if "trend" in working.columns else len(working)
    forecast_rows = []

    for offset in range(1, days + 1):
        next_date = working["date"].iloc[-1] + pd.Timedelta(days=1)
        row = build_feature_row(next_date, working, trend_start + offset)
        row_df = pd.DataFrame([row])
        pred = float(model.predict(row_df[feature_cols])[0])
        row["quantity"] = pred
        forecast_rows.append({"Day": offset, "Date": next_date.date(), "Predicted Demand": pred})
        working = pd.concat([working, pd.DataFrame([row])], ignore_index=True)

    return pd.DataFrame(forecast_rows), working


def make_actual_vs_predicted_figure(y_test: pd.Series, pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(13.5, 6.6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(y_test.values, color="#1f77b4", linewidth=2.0, label="Actual")
    ax.plot(pred, color="#ff7f0e", linewidth=2.0, linestyle="--", label="Predicted")
    ax.set_title("Actual vs Predicted", fontsize=18, fontweight="bold", pad=16)
    ax.grid(True, alpha=0.18)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=12)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    return fig


def make_feature_importance_figure(model, feature_cols: list[str]):
    if not hasattr(model, "feature_importances_"):
        return None, None

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(13.5, 7.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    importance.plot(kind="barh", ax=ax, color="#1f77b4")
    ax.invert_yaxis()
    ax.set_title("Feature Importance", fontsize=18, fontweight="bold", pad=16)
    ax.grid(True, axis="x", alpha=0.16)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    return fig, importance


def make_forecast_figure(history: pd.DataFrame, forecast_df: pd.DataFrame):
    recent_actual = history["quantity"].tail(30).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(recent_actual.index, recent_actual.values, color="#1f77b4", linewidth=2.0, label="Recent Actual")
    forecast_x = range(len(recent_actual), len(recent_actual) + len(forecast_df))
    ax.plot(
        forecast_x,
        forecast_df["Predicted Demand"].values,
        color="#ff7f0e",
        linewidth=2.0,
        linestyle="--",
        label="Forecast",
    )
    ax.set_title("Forecast Visualization", fontsize=18, fontweight="bold", pad=16)
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=12)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    return fig


def render_status_banner(mae: float, baseline_mae: float) -> None:
    if mae < baseline_mae:
        st.markdown(
            "<div class='status-banner status-good'>Model outperforms baseline - Reliable</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='status-banner status-warn'>Model underperforms baseline - review feature engineering</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_styles()
    df = load_data()
    model, model_features = load_model()
    feature_cols = select_feature_columns(df, model_features)
    eval_result = evaluate_model(model, df, feature_cols)
    model = eval_result["model"]

    st.markdown(
        f"""
        <div class="hero">
            <div style="display:flex; align-items:flex-start; gap:1rem;">
                <div style="font-size:3rem; line-height:1;">📊</div>
                <div>
                    <h1>{APP_TITLE}</h1>
                    <p>Professional forecasting workspace for model validation, future demand planning, and feature-driven business decisions.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("📈", "Model Performance", "Holdout evaluation on the processed daily demand dataset.")
    metric_cols = st.columns(3)
    metric_cols[0].metric("MAE", f"{eval_result['mae']:.2f}")
    metric_cols[1].metric("RMSE", f"{eval_result['rmse']:.2f}")
    metric_cols[2].metric("Baseline MAE", f"{eval_result['baseline_mae']:.2f}")
    render_status_banner(eval_result["mae"], eval_result["baseline_mae"])

    section_header(
        "📉",
        "Actual vs Predicted",
        "Blue line shows actual demand, dashed orange shows the model forecast on the test split.",
    )
    fig_actual = make_actual_vs_predicted_figure(eval_result["y_test"], eval_result["pred"])
    st.pyplot(fig_actual, clear_figure=True, use_container_width=True)

    section_header(
        "🔮",
        "Future Demand Forecast",
        "Choose a forecast horizon and generate recursive predictions from the latest observed date.",
    )
    forecast_days = st.slider("Forecast Horizon (days)", 1, 14, 7)
    generate = st.button("Generate Forecast", use_container_width=False)

    if "forecast_df" not in st.session_state:
        st.session_state.forecast_df = None
        st.session_state.forecast_horizon = None
        st.session_state.forecast_history = None

    if generate:
        forecast_df, forecast_history = forecast_future(model, df, feature_cols, forecast_days)
        st.session_state.forecast_df = forecast_df
        st.session_state.forecast_history = forecast_history
        st.session_state.forecast_horizon = forecast_days

    if st.session_state.forecast_df is not None and st.session_state.forecast_horizon == forecast_days:
        forecast_df = st.session_state.forecast_df
        st.success(f"Forecast generated for next {forecast_days} days")
        st.dataframe(forecast_df.round(4), use_container_width=True, hide_index=True)
        st.pyplot(make_forecast_figure(df, forecast_df), clear_figure=True, use_container_width=True)

        section_header("💼", "Business Insight")
        avg_demand = float(forecast_df["Predicted Demand"].mean())
        peak_idx = int(forecast_df["Predicted Demand"].idxmax())
        peak_day = int(forecast_df.loc[peak_idx, "Day"])
        peak_value = float(forecast_df.loc[peak_idx, "Predicted Demand"])
        low_value = float(forecast_df["Predicted Demand"].min())

        st.markdown(
            f"""
            <div class="soft-card">
                <div style="font-size:1.05rem; color:#56e08e; font-weight:700; margin-bottom:1rem;">
                    Average predicted demand for next {forecast_days} days: {avg_demand:.2f}
                </div>
                <div style="font-size:1.02rem; margin-bottom:0.8rem; color:#f6f8ff;">👉 Use this to:</div>
                <ul style="margin:0; padding-left:1.2rem; line-height:1.9; color:#dbe4ff;">
                    <li>Plan inventory accordingly</li>
                    <li>Reduce food waste</li>
                    <li>Optimize staffing</li>
                </ul>
                <div class="mini-note" style="margin-top:0.95rem;">
                    Peak day: Day {peak_day} at {peak_value:.2f} | Lowest forecast: {low_value:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Click **Generate Forecast** to display the forecast table, chart, and business insight panel.")

    section_header("🔍", "Feature Importance", "The strongest drivers are usually the lag and rolling window features.")
    fig_importance, importance = make_feature_importance_figure(model, feature_cols)
    if fig_importance is not None and importance is not None:
        st.pyplot(fig_importance, clear_figure=True, use_container_width=True)
        st.markdown(
            f"""
            <div class="soft-card" style="margin-top: 0.85rem;">
                <div style="font-size:1.05rem; color:#55a8ff;">Top driver: {importance.index[0]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Feature importance is not available for the loaded model.")


if __name__ == "__main__":
    main()
