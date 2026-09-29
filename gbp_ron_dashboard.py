import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import time

st.set_page_config(page_title="GBP/RON Strategy Simulator", layout="wide")


@st.cache_data
def load_data():
    df = yf.download(
        "GBPRON=X",
        start="2016-01-01",
        end="2026-09-17",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # We only use daily closing prices for now.
    return df[["Close"]].dropna()


NR_DAYS_STANDARDIZE = 40  # number of observations used to standardize the N-day return
def z_signal(df, lookback):
    # Match the project:
    # 1) calculate the N-day return, where N is controlled by the slider;
    # 2) standardize that return against a FIXED 40-observation rolling
    #    mean and standard deviation.
    r = df["Close"].pct_change(lookback)
    mu = r.rolling(NR_DAYS_STANDARDIZE).mean()
    sd = r.rolling(NR_DAYS_STANDARDIZE).std().replace(0, np.nan)
    return (r - mu) / sd


def simulate(df, lookback, initial, cost=0.0, ron_rate=0.0, gbp_rate=0.0):
    x = df.copy()
    x["Z"] = z_signal(x, lookback)

    # Z[t-1] is known before day t; it controls today's position.
    x["Momentum_Position"] = (x["Z"].shift(1) > 1).astype(int)
    x["Reversion_Position"] = (x["Z"].shift(1) < -1).astype(int)
    x.loc[
        x["Z"].shift(1).isna(),
        ["Momentum_Position", "Reversion_Position"],
    ] = 0

    # IMPORTANT MODEL ASSUMPTION (temporary simplification):
    # We assume we can trade at yesterday's closing GBP/RON rate, even though
    # this is not literally executable because the daily observation is only
    # known after the close. Therefore, today's holding return is modelled as
    # yesterday's close -> today's close. We can later replace this with a
    # next-day Open price to make execution timing more realistic.
    x["GBP_Day_Return"] = x["Close"].pct_change()

    # NEW: interest accrued between observations. Calendar days are used, so a
    # Friday -> Monday step earns 3 days of interest. Rates are effective annual rates.
    days = x.index.to_series().diff().dt.days
    ron_interest = (1 + ron_rate) ** (days / 365) - 1
    gbp_interest = (1 + gbp_rate) ** (days / 365) - 1

    # NEW: daily return of each state, measured in RON
    ret_in_gbp = (1 + x["GBP_Day_Return"]) * (1 + gbp_interest) - 1  # GBP/RON move + GBP interest
    ret_in_ron = ron_interest                                        # RON interest

    # NEW: a conversion (entry or exit) happens whenever the position changes
    for name, pos in (("Momentum", "Momentum_Position"), ("Reversion", "Reversion_Position")):
        trade = x[pos].diff().abs().fillna(0)
        x[f"{name}_Return"] = (1 + ret_in_gbp.where(x[pos] == 1, ret_in_ron)) * (1 - trade * cost) - 1
        x[f"{name}_Portfolio"] = initial * (1 + x[f"{name}_Return"]).cumprod()

    # Buy & hold pays one conversion (RON -> GBP) at the start
    x["BuyHold_Portfolio"] = initial * (1 - cost) * (1 + ret_in_gbp).cumprod()
    return x


def sharpe(r):
    r = r.dropna()
    s = r.std()
    return np.nan if pd.isna(s) or s == 0 else r.mean() / s * np.sqrt(252)


def _visible_history(x, idx, max_points=900):
    """Limit points sent to Plotly so each frame stays light."""
    v = x.iloc[: idx + 1]
    return v if len(v) <= max_points else v.iloc[-max_points:]


def portfolio_chart(x, idx):
    """Single shared chart for the three portfolio paths."""
    v = _visible_history(x, idx)
    today = x.index[idx]

    f = go.Figure()
    f.add_trace(go.Scatter(
        x=v.index,
        y=v["BuyHold_Portfolio"],
        mode="lines",
        name="Baseline (Buy & Hold)",
    ))
    f.add_trace(go.Scatter(
        x=v.index,
        y=v["Momentum_Portfolio"],
        mode="lines",
        name="Momentum",
    ))
    f.add_trace(go.Scatter(
        x=v.index,
        y=v["Reversion_Portfolio"],
        mode="lines",
        name="Mean Reversal",
    ))
    f.add_vline(
        x=today,
        line_color="#ff2b8a",
        line_dash="dash",
        annotation_text="Today",
    )
    f.update_layout(
        title="Portfolio Value — Baseline vs Momentum vs Mean Reversal",
        height=560,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
        uirevision="portfolio-chart",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return f


def portfolio_metrics(x, idx, initial):
    cur = x.iloc[idx]
    return {
        "Baseline": cur["BuyHold_Portfolio"],
        "Momentum": cur["Momentum_Portfolio"],
        "Mean Reversal": cur["Reversion_Portfolio"],
    }

data = load_data()

if "running" not in st.session_state:
    st.session_state.running = False
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "lookback" not in st.session_state:
    st.session_state.lookback = 10
if "initial" not in st.session_state:
    st.session_state.initial = 10000.0

st.title("GBP/RON Strategy Simulator")
st.caption(
    "Interactive backtest: watch the two hypotheses evolve through the historical data."
)

# ------------------------------------------------------------
# Controls
# ------------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns([2.2, 1.5, 1.1, 1.1, 1.1, 1.2])
with c1:
    lookback = st.slider(
        "Z-score signal horizon (trading days)",
        5,
        60,
        st.session_state.lookback,
    )
with c2:
    initial = st.number_input(
        "Starting capital (RON)",
        100.0,
        1000000.0,
        st.session_state.initial,
        500.0,
    )
with c3:
    speed = st.slider(
        "Simulation speed",
        0.05,
        0.75,
        0.25,
        0.01,
        help="Seconds between animation frames. Higher = slower; lower = faster.",
    )
with c4:
    if st.button("▶ Start Simulation", type="primary", width="stretch"):
        st.session_state.lookback = lookback
        st.session_state.initial = initial
        st.session_state.idx = min(max(lookback + 40, 1), len(data) - 1)
        st.session_state.running = True
        st.rerun()
with c5:
    if st.button("■ Stop Simulation", width="stretch"):
        st.session_state.running = False
        st.rerun()
with c6:
    if st.button("↺ Reset", width="stretch"):
        st.session_state.running = False
        st.session_state.idx = 0
        st.rerun()

# Trading frictions and interest
f1, f2, f3, f4 = st.columns([1.4, 1.4, 1.4, 2.4])
with f1:
    cost_pct = st.number_input(
        "Cost per conversion (%)", 0.0, 5.0, 0.28, 0.05, format="%.2f",
        help="Fee or exchange-rate markup paid on every entry (RON to GBP) and every exit (GBP to RON).",
    )
with f2:
    ron_rate_pct = st.number_input(
        "RON deposit rate (% p.a.)", 0.0, 20.0, 2.00, 0.10, format="%.2f",
        help="Interest earned while the portfolio sits in RON.",
    )
with f3:
    gbp_rate_pct = st.number_input(
        "GBP deposit rate (% p.a.)", 0.0, 20.0, 2.90, 0.10, format="%.2f",
        help="Interest earned while the portfolio sits in GBP.",
    )

x = simulate(data, lookback, initial, cost_pct / 100, ron_rate_pct / 100, gbp_rate_pct / 100)

# ------------------------------------------------------------
# Simulation display
# ------------------------------------------------------------
# Only the metric row and ONE Plotly chart are updated during animation.
# This is deliberately much lighter than rebuilding separate charts for
# exchange rate, position, and portfolio value for both strategies.

metrics_placeholder = st.empty()
chart_placeholder = st.empty()


def render_display(x, idx, lookback, initial):
    values = portfolio_metrics(x, idx, initial)

    with metrics_placeholder.container():
        a, b, c = st.columns(3)
        a.metric(
            "Baseline — Buy & Hold",
            f"{values['Baseline']:,.2f} RON",
            delta=f"{(values['Baseline']/initial-1)*100:.2f}%"
        )
        b.metric(
            "Momentum",
            f"{values['Momentum']:,.2f} RON",
            delta=f"{(values['Momentum']/initial-1)*100:.2f}%"
        )
        c.metric(
            "Mean Reversal",
            f"{values['Mean Reversal']:,.2f} RON",
            delta=f"{(values['Mean Reversal']/initial-1)*100:.2f}%"
        )

    with chart_placeholder.container():
        st.plotly_chart(
            portfolio_chart(x, idx),
            width="stretch",
            key=f"portfolio_{idx}",
        )


# ------------------------------------------------------------
# Animation
# ------------------------------------------------------------
if st.session_state.running:
    start_idx = max(lookback + 40, 1)
    frame_step = 1  # one trading day per frame for smooth progression
    frame_indices = range(start_idx, len(x), frame_step)

    for idx in frame_indices:
        render_display(x, idx, lookback, initial)
        time.sleep(speed)

    # Always finish on the final observation.
    if len(x) - 1 >= start_idx:
        render_display(x, len(x) - 1, lookback, initial)

    st.session_state.running = False
    st.session_state.idx = len(x) - 1
    st.success("Simulation complete.")
else:
    idx = st.session_state.idx
    if idx == 0:
        idx = len(x) - 1
    render_display(x, idx, lookback, initial)

st.divider()
st.caption(
    "Educational backtest. Yahoo Finance daily data. Conversion costs and deposit interest "
    "use the constant assumptions above; slippage, tax and historical changes in interest "
    "rates are not modelled."
)
