import numpy as np
import yfinance as yf

DAYS_10 = 10 # 2 trading weeks, what I'd take as a signal
DAYS_20 = 20
DAYS_40 = 40
DAYS_60 = 60

# Define Trading Rules 
# Strategy 1 - Momentum: if z-score > 1, long; else, wait (assume trend will continue) 
# Strategy 2 - Mean Reversion: if z-score < -1, long; else, do nothing (assume trend will reverse) 

# Download GBP/RON data from Yahoo Finance
data = yf.download(
    "GBPRON=X",
    start="2016-01-01",
    end="2026-09-17",
    interval="1d"
)

# ---- PART 1 - Test if Signal has Predictive Power ----
print("\n========================================")
print("PART 1 - TEST SIGNAL'S PREDICTIVE POWER")
print("========================================")
# signal = average return of future 10 days, standardized over last 40 days


# ------------------------------------------------------------
# 1.1 Calculate 10-day return
# NOTE: this assumes I buy at TODAY's closing price - unrealistic since we would only
#  know the z-score at the end of the day, but simplified for now - to change in future 
data["Return_Past10Days"] = data["Close"].pct_change(periods=DAYS_10) * 100 #NOTE: first 10 values will be NaN since no past data to calculate
data["Return_Future10Days"] = data["Close"].pct_change(periods=DAYS_10).shift(-DAYS_10) * 100 #NOTE: last 10 values will be NaN since no future data to calculate

# ------------------------------------------------------------
# 1.2 Standardize the 10-day return according to mean and std dev over last 40 days (  z-score = (x - mean) / std  )
# NOTE: the 40 observations are made of *overlapping* rolling windows, so NOT independent.. be mindful about conclusions
mean_40Days = data["Return_Past10Days"].rolling(window=DAYS_40).mean();
std_40Days = data["Return_Past10Days"].rolling(window=DAYS_40).std(); # NOTE: use rolling mean, std to compare with recent movements, not overall data
data["Return_Past10Days_Standardized"] = np.where(
    (std_40Days != 0) & std_40Days.notna(),
    (data["Return_Past10Days"] - mean_40Days) / std_40Days,
    np.nan
)

# ------------------------------------------------------------
# 1.3 Calculate average future 10-day returns for each strategy

# S1: buy if z>1 -> calc avg return 
data["Return_Future10Days_Momentum"] = np.where(
    data["Return_Past10Days_Standardized"] > 1, 
    data["Return_Future10Days"],
    np.nan
)
avg_return_S1 = data["Return_Future10Days_Momentum"].mean()

#S1: buy if z<-1 -> calc avg return
data["Return_Future10Days_MeanReversion"] = np.where(
    data["Return_Past10Days_Standardized"] < -1,    
    data["Return_Future10Days"],
    np.nan
)
avg_return_S2 = data["Return_Future10Days_MeanReversion"].mean()

# Baseline: -1 < z < 1 -> calc avg return
data["Return_Future10Days_Baseline"] = np.where(
    (data["Return_Past10Days_Standardized"] >= -1) & (data["Return_Past10Days_Standardized"] <= 1),
    data["Return_Future10Days"],
    np.nan
)
avg_return_S3 = data["Return_Future10Days_Baseline"].mean()

# ------------------------------------------------------------
# 1.4 Print results
print("Average future 10-day returns for each strategy:")
print(f"Average Return - Momentum Strategy: {avg_return_S1}")
print(f"Average Return - Mean Reversion Strategy: {avg_return_S2}")
print(f"Average Return - Baseline Strategy: {avg_return_S3}")


# ============================================================
# PART 2 - RON → GBP → RON PORTFOLIO SIMULATION
# ============================================================

# STRATEGY HYPOTHESES
# ------------------------------------------------------------
# H1 - MOMENTUM 
# ------------------------------------------------------------
#
# H1.1 ENTER / HOLD:
#     Z > 1 (GBP position)
#
# H1.2 EXIT:
#     Z <= 1 (RON position)
#
# Interpretation:
# Assume a strong upward or downward trend in GBP/RON will continue,  
# so buy only when the trend is increasing, i.e. when Z > 1, and hold until Z <= 1
# (if trend is decreasing, assume it will keep decreasing so wait)
#
# ------------------------------------------------------------
# H2 - MEAN REVERSION
# ------------------------------------------------------------
#
# H2.1 ENTER / HOLD:
#     Z < -1 (GBP position)
#
# H2.2 EXIT:
#     Z >= -1 (RON position)
#
# Interpretation:
# Assume a strong upward or downward trend in GBP/RON will recover and return to the mean,  
# so buy only when the trend is decreasing, i.e. when Z < -1, and hold until Z >= -1
# (if trend is increasing, assume it will keep increasing so wait)
#
#
# IMPORTANT:
# H1 and H2 are tested as SEPARATE strategies.
# We do NOT combine them.
#
# Each strategy starts with the same initial RON capital.


# ------------------------------------------------------------
# 2.1 Initial capital
# ------------------------------------------------------------

INITIAL_CAPITAL = 10_000  # RON

# NEW: trading frictions and interest (edit these to test other assumptions)
COST_PER_TRADE = 0.0028   # 0.28% per conversion, i.e. per entry (RON->GBP) and per exit (GBP->RON)
RON_RATE = 0.02          # RON instant-access deposit rate, per year (2.0%)
GBP_RATE = 0.029         # GBP instant-access deposit rate, per year (2.9%)


# ------------------------------------------------------------
# 2.2 Calculate daily GBP/RON return
# ------------------------------------------------------------

data["DailyReturn"] = data["Close"].pct_change()


# Interest gained between observations. Calendar days are used, so a Friday -> Monday
# step earns 3 days of interest. Rates are treated as effective annual rates.
days_elapsed = data.index.to_series().diff().dt.days
data["RON_Interest"] = (1 + RON_RATE) ** (days_elapsed / 365) - 1
data["GBP_Interest"] = (1 + GBP_RATE) ** (days_elapsed / 365) - 1


# Return of each state, measured in RON
#   holding RON -> RON interest
#   holding GBP -> GBP/RON move + GBP interest
data["Return_InRON"] = data["RON_Interest"]
data["Return_InGBP"] = data["DailyReturn"] + data["GBP_Interest"] + (data["DailyReturn"] * data["GBP_Interest"])


# ============================================================
# 2.3 MOMENTUM STRATEGY
# ============================================================
# H1.1 / H1.2
#
# Z > 1:  enter/hold GBP position
#
# Z <= 1: exit GBP position, hold RON
# ------------------------------------------------------------
# NOTE: we use yesterday's Z score to determine today's position. 
# so, since the trade executes TODAY, the returns would have to be calculated using today's Closing vs Opening price;
# however, simplified it assuming trade executes at yesterday's closing price -> to change in future 


# 1 = GBP
# 0 = RON
data["Momentum_Position"] = np.where(
    data["Return_Past10Days_Standardized"].shift(1) > 1,
    1,
    0
)


# ------------------------------------------------------------
# Identify actual entries and exits
# ------------------------------------------------------------

data["Momentum_PositionChange"] = (
    data["Momentum_Position"].diff()
)

# H1.1: RON -> GBP
data["Momentum_Entry"] = np.where(
    data["Momentum_PositionChange"] == 1,
    1,
    0
)

# H1.2: GBP -> RON
data["Momentum_Exit"] = np.where(
    data["Momentum_PositionChange"] == -1,
    1,
    0
)


# ------------------------------------------------------------
# Calculate daily return of strategy
# ------------------------------------------------------------

# If holding GBP:  portfolio changes with GBP/RON plus GBP interest
# If holding RON:  portfolio not exposed to GBP/RON fluctuation - only earns RON interest
# Exchange fees: every entry or exit is a currency conversion that costs COST_PER_TRADE

# Momentum_Trade = 1 on days when entry/exit strategy, 0 when no conversion
data["Momentum_Trade"] = data["Momentum_PositionChange"].abs().fillna(0)
# apply exchange fee if trade is executed that day
Momentum_Exchange_Fee = data["Momentum_Trade"] * COST_PER_TRADE
# select GBP or RON return based on which the strategy is currently holding (remember 1=GBP, 0=RON)
Momentum_GBP_or_RON_Return = data["Return_InGBP"].where(data["Momentum_Position"] == 1, data["Return_InRON"]) 
# Calculate returns
data["Momentum_Strategy_Return"] = (1 + Momentum_GBP_or_RON_Return) * ( 1 - Momentum_Exchange_Fee ) - 1
# No return on first day (no previous day to compare returns to, so holding RON)
data.loc[data.index[0], "Momentum_Strategy_Return"] = 0

# ------------------------------------------------------------
# Build RON portfolio
# ------------------------------------------------------------

data["Momentum_Portfolio"] = (
    INITIAL_CAPITAL *
    (1 + data["Momentum_Strategy_Return"]).cumprod()
)


# ============================================================
# 2.4 MEAN REVERSION STRATEGY
# ============================================================

# H2.1 / H2.2
#
# Yesterday's Z-score determines today's desired position.
#
# Z < -1: desired position = GBP
#
# Z >= -1: desired position = RON
# ------------------------------------------------------------


# 1 = GBP
# 0 = RON
data["MeanReversion_Position"] = np.where(
    data["Return_Past10Days_Standardized"].shift(1) < -1,
    1,
    0
)


# ------------------------------------------------------------
# Identify actual entries and exits
# ------------------------------------------------------------

data["MeanReversion_PositionChange"] = (
    data["MeanReversion_Position"].diff()
)

# H2.1: RON -> GBP
data["MeanReversion_Entry"] = np.where(
    data["MeanReversion_PositionChange"] == 1,
    1,
    0
)

# H2.2: GBP -> RON
data["MeanReversion_Exit"] = np.where(
    data["MeanReversion_PositionChange"] == -1,
    1,
    0
)

# ------------------------------------------------------------
# Calculate daily return
# ------------------------------------------------------------
# Exchange fees: every entry or exit is a currency conversion that costs COST_PER_TRADE
data["MeanReversion_Trade"] = data["MeanReversion_PositionChange"].abs().fillna(0)
# apply exchange fee if trade is executed that day
MeanReversion_Exchange_Fee = data["MeanReversion_Trade"] * COST_PER_TRADE
# select GBP or RON return based on which the strategy is currently holding (remember 1=GBP, 0=RON)
MeanReversion_GBP_or_RON_Return = data["Return_InGBP"].where(data["MeanReversion_Position"] == 1, data["Return_InRON"])
# Calculate returns
data["MeanReversion_Strategy_Return"] = (1 + MeanReversion_GBP_or_RON_Return) * (1 - MeanReversion_Exchange_Fee) - 1
# No return on first day (no previous day to compare returns to, so holding RON)
data.loc[data.index[0], "MeanReversion_Strategy_Return"] = 0

# ------------------------------------------------------------
# Build RON portfolio
# ------------------------------------------------------------
data["MeanReversion_Portfolio"] = (
    INITIAL_CAPITAL *
    (1 + data["MeanReversion_Strategy_Return"]).cumprod()
)

# ============================================================
# 2.5 BUY & HOLD BENCHMARK
# ============================================================

# Convert all RON -> GBP at the beginning and never exit.
#
# This avoids trading (and paying numerous conversion fees),
# allowing the GBP portfolio to accumulate interest over the whole period.
#
# Exchange fee for one conversion (RON -> GBP) at the start; GBP interest accumulates while holding
data["Return_BuyHold"] = data["Return_InGBP"] 
data.loc[data.index[0], "Return_BuyHold"] = -1 * COST_PER_TRADE  # pay conversion fee on first day

data["BuyHold_Portfolio"] = (
    INITIAL_CAPITAL * 
    (1 + data["Return_BuyHold"]).cumprod()
)


# ============================================================
# 2.6 SHARPE & INFORMATION RATIOS
# ============================================================
# Sharpe = how much return each strategy earns per unit of risk, 
#           where risk = volatility of strategy's returns
# Information = how much return each strategy earns per unit of risk, 
#               where risk = tracking error = volatility of difference between strategy returns and buy&hold returns

# NOTE: see README Notes section for justification of using the buy&hold as a "risk-free" benchmark for the ratios

TRADING_DAYS = 252

# Annualized Sharpe ratio on returns in excess of the buy&hold benchmark, 
# i.e. what the strategy earns by exchanging the sum once to GBP and letting it accumulate interest 
# (consider this as the risk-free alternative in our case)

momentum_excess_overBuyHold = data["Momentum_Strategy_Return"] - data["Return_BuyHold"]
mean_reversion_excess_overBuyHold = data["MeanReversion_Strategy_Return"] - data["Return_BuyHold"]

momentum_sharpe = (
    momentum_excess_overBuyHold.mean()
    / data["Momentum_Strategy_Return"].std()
    * np.sqrt(TRADING_DAYS)
)

mean_reversion_sharpe = (
    mean_reversion_excess_overBuyHold.mean()
    / data["MeanReversion_Strategy_Return"].std()
    * np.sqrt(TRADING_DAYS)
)

momentum_information = (
    momentum_excess_overBuyHold.mean()
    / momentum_excess_overBuyHold.std()
    * np.sqrt(TRADING_DAYS)
)

mean_reversion_information = (
    mean_reversion_excess_overBuyHold.mean()
    / mean_reversion_excess_overBuyHold.std()
    * np.sqrt(TRADING_DAYS)
)


# ============================================================
# 2.7 TOTAL RETURNS
# ============================================================

momentum_total_return = (
    data["Momentum_Portfolio"].iloc[-1]
    / INITIAL_CAPITAL
    - 1
)

mean_reversion_total_return = (
    data["MeanReversion_Portfolio"].iloc[-1]
    / INITIAL_CAPITAL
    - 1
)

buyhold_total_return = (
    data["BuyHold_Portfolio"].iloc[-1]
    / INITIAL_CAPITAL
    - 1
)


# ============================================================
# 2.8 TIME IN GBP
# ============================================================

momentum_time_invested = (
    data["Momentum_Position"].mean()
)

mean_reversion_time_invested = (
    data["MeanReversion_Position"].mean()
)


# ============================================================
# 2.9 NUMBER OF ENTRIES / EXITS
# ============================================================

momentum_entries = data["Momentum_Entry"].sum()
momentum_exits = data["Momentum_Exit"].sum()

mean_reversion_entries = data["MeanReversion_Entry"].sum()
mean_reversion_exits = data["MeanReversion_Exit"].sum()


# ============================================================
# 2.10 PRINT RESULTS
# ============================================================

print("\n========================================")
print("PART 2 - PORTFOLIO SIMULATION")
print("========================================")
print(
    f"Assumptions: {COST_PER_TRADE * 100:.2f}% cost per conversion | "
    f"RON deposit {RON_RATE * 100:.2f}% p.a. | GBP deposit {GBP_RATE * 100:.2f}% p.a."
)

print("\n--- H1: MOMENTUM ---")

print(
    f"Final portfolio value: "
    f"{data['Momentum_Portfolio'].iloc[-1]:,.2f} RON"
)

print(
    f"Total return: "
    f"{momentum_total_return * 100:.2f}%"
)

print(
    f"Annualized Sharpe ratio (excess over buy&hold): "
    f"{momentum_sharpe:.2f}"
)

print(
    f"Annualized Information ratio (excess over buy&hold): "
    f"{momentum_information:.2f}"
)

print(
    f"Time holding GBP: "
    f"{momentum_time_invested * 100:.2f}%"
)

print(
    f"Entries: "
    f"{momentum_entries:.0f}"
)

print(
    f"Exits: "
    f"{momentum_exits:.0f}"
)

print(
    f"Total conversions: "
    f"{momentum_entries + momentum_exits:.0f}"
)


print("\n--- H2: MEAN REVERSION ---")

print(
    f"Final portfolio value: "
    f"{data['MeanReversion_Portfolio'].iloc[-1]:,.2f} RON"
)

print(
    f"Total return: "
    f"{mean_reversion_total_return * 100:.2f}%"
)

print(
    f"Annualized Sharpe ratio (excess over buy&hold): "
    f"{mean_reversion_sharpe:.2f}"
)

print(
    f"Annualized Information ratio (excess over buy&hold): "
    f"{mean_reversion_information:.2f}"
)

print(
    f"Time holding GBP: "
    f"{mean_reversion_time_invested * 100:.2f}%"
)

print(
    f"Entries: "
    f"{mean_reversion_entries:.0f}"
)

print(
    f"Exits: "
    f"{mean_reversion_exits:.0f}"
)

print(
    f"Total conversions: "
    f"{mean_reversion_entries + mean_reversion_exits:.0f}"
)


print("\n--- BUY & HOLD ---")

print(
    f"Final portfolio value: "
    f"{data['BuyHold_Portfolio'].iloc[-1]:,.2f} RON"
)

print(
    f"Total return: "
    f"{buyhold_total_return * 100:.2f}%"
)
