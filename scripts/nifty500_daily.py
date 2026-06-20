import yfinance as yf
import pandas as pd
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

IST = ZoneInfo("Asia/Kolkata")


# ===============================
# LOAD NIFTY 500 SYMBOLS
# ===============================
nifty500_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
symbols_df = pd.read_csv(nifty500_url)

sector_map = dict(zip(symbols_df["Symbol"], symbols_df["Industry"]))
symbols = [s + ".NS" for s in symbols_df["Symbol"].tolist()]


# ===============================
# CANDLE PATTERN LOGIC
# ===============================
def candle_pattern(open_p, high_p, low_p, close_p):
    body = abs(close_p - open_p)
    total_range = high_p - low_p

    if total_range == 0:
        return "Flat"

    upper_wick = high_p - max(open_p, close_p)
    lower_wick = min(open_p, close_p) - low_p

    body_pct = body / total_range
    upper_pct = upper_wick / total_range
    lower_pct = lower_wick / total_range

    if body_pct < 0.10:
        return "Doji"

    if body_pct > 0.80:
        return "Bullish Marubozu" if close_p > open_p else "Bearish Marubozu"

    if lower_pct > 0.50 and body_pct < 0.30 and upper_pct < 0.20:
        return "Hammer"

    if upper_pct > 0.50 and body_pct < 0.30 and lower_pct < 0.20:
        return "Shooting Star"

    return "Bullish" if close_p > open_p else "Bearish"


# ===============================
# FETCH DATA
# ===============================
def fetch_data(symbol):
    try:
        base_symbol = symbol.replace(".NS", "")
        ticker = yf.Ticker(symbol)

        daily_full = ticker.history(period="3mo", interval="1d")
        weekly_full = ticker.history(period="8mo", interval="1wk")

        daily_hist = daily_full['Close'].tail(22).iloc[:-1]
        daily_ltp = daily_full['Close'].iloc[-1]

        row = {
            "Symbol": symbol,
            "Sector": sector_map.get(base_symbol, None)
        }

        # ===============================
        # DAILY CLOSING PRICES
        # ===============================
        for i in range(21):
            row[f"D{i+1}"] = round(daily_hist.iloc[i], 2) if i < len(daily_hist) else None
        row["D_LTP"] = round(daily_ltp, 2)

        # ===============================
        # LAST 5 DAILY CANDLES
        # ===============================
        last_5 = daily_full.tail(5)
        completed_4 = last_5.iloc[:-1]
        live_candle = last_5.iloc[-1]

        for idx, (dt, candle) in enumerate(completed_4.iterrows(), start=1):
            row[f"C{idx}_{dt.strftime('%Y-%m-%d')}"] = candle_pattern(
                candle['Open'], candle['High'], candle['Low'], candle['Close']
            )

        row["Day Live Candle"] = candle_pattern(
            live_candle['Open'], live_candle['High'], live_candle['Low'], live_candle['Close']
        )

        # ===============================
        # LAST COMPLETED DAY
        # ===============================
        if len(daily_full) >= 2:
            last_candle = daily_full.iloc[-1]
            prev_candle = daily_full.iloc[-2]
            use_candle = prev_candle if last_candle.name.date() == datetime.now().date() else last_candle
        else:
            use_candle = daily_full.iloc[-1]

        open_p = use_candle['Open']
        high_p = use_candle['High']
        low_p = use_candle['Low']

        row["Open=High"] = "Yes" if abs(open_p - high_p) <= 0.001 * open_p else "No"
        row["Open=Low"] = "Yes" if abs(open_p - low_p) <= 0.001 * open_p else "No"

        prev_close = daily_full['Close'].iloc[-2] if len(daily_full) >= 2 else daily_ltp
        row["LTP_vs_PrevDayClose_%"] = round(((daily_ltp - prev_close) / prev_close) * 100, 2)

        # ===============================
        # RSI DAILY (11-period, Wilder's smoothing)
        # ===============================
        delta_d = daily_full["Close"].diff()
        gain_d = delta_d.where(delta_d > 0, 0.0)
        loss_d = -delta_d.where(delta_d < 0, 0.0)
        avg_gain_d = gain_d.ewm(alpha=1/11, min_periods=11, adjust=False).mean()
        avg_loss_d = loss_d.ewm(alpha=1/11, min_periods=11, adjust=False).mean()
        rs_d = avg_gain_d / avg_loss_d
        rsi_d = 100 - (100 / (1 + rs_d))

        row["RSI_Daily"] = round(rsi_d.iloc[-1], 2)

        # ===============================
        # RSI WEEKLY (11-period, Wilder's smoothing)
        # ===============================
        delta_w = weekly_full["Close"].diff()
        gain_w = delta_w.where(delta_w > 0, 0.0)
        loss_w = -delta_w.where(delta_w < 0, 0.0)
        avg_gain_w = gain_w.ewm(alpha=1/11, min_periods=11, adjust=False).mean()
        avg_loss_w = loss_w.ewm(alpha=1/11, min_periods=11, adjust=False).mean()
        rs_w = avg_gain_w / avg_loss_w
        rsi_w = 100 - (100 / (1 + rs_w))

        row["RSI_Weekly"] = round(rsi_w.iloc[-1], 2)

        # ===============================
        # EMA 9 and EMA 50 (Daily)
        # ===============================
        ema9 = daily_full["Close"].ewm(span=9, adjust=False).mean()
        ema50 = daily_full["Close"].ewm(span=50, adjust=False).mean()

        row["EMA9"] = round(ema9.iloc[-1], 2)
        row["EMA50"] = round(ema50.iloc[-1], 2)
        row["EMA9_vs_EMA50_%"] = round(((ema9.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1]) * 100, 2)

        # ===============================
        # 30 MIN CANDLES (today from 9:15 IST, resampled from 15m)
        # yfinance 30m bars align to UTC :00/:30, so we fetch 15m
        # and resample with offset='15min' to get correct 9:15 bins
        # ===============================
        raw_15m = ticker.history(period="2d", interval="15m")

        if not raw_15m.empty:
            if raw_15m.index.tz is None:
                raw_15m.index = raw_15m.index.tz_localize("UTC").tz_convert(IST)
            else:
                raw_15m.index = raw_15m.index.tz_convert(IST)

            today = datetime.now(IST).date()
            market_open_dt = datetime.combine(today, dtime(9, 15)).replace(tzinfo=IST)

            today_15m = raw_15m[
                (raw_15m.index.date == today) &
                (raw_15m.index >= market_open_dt)
            ]

            if not today_15m.empty:
                intraday = today_15m.resample("30min", offset="15min").agg({
                    "Open": "first", "High": "max", "Low": "min", "Close": "last"
                }).dropna()

                if not intraday.empty:
                    live = intraday.iloc[-1]
                    completed = intraday.iloc[:-1]

                    row["Last_30min_Candle"] = candle_pattern(
                        live["Open"], live["High"], live["Low"], live["Close"]
                    )

                    for idx, (dt, candle) in enumerate(completed.iterrows(), start=1):
                        end = (dt + timedelta(minutes=30)).strftime("%H:%M")
                        col = f"30m_C{idx:02d}_{dt.strftime('%H:%M')}-{end}"
                        row[col] = candle_pattern(candle["Open"], candle["High"], candle["Low"], candle["Close"])
                else:
                    row["Last_30min_Candle"] = None
            else:
                row["Last_30min_Candle"] = None
        else:
            row["Last_30min_Candle"] = None

        # ===============================
        # DAILY METRICS
        # ===============================
        d_avg, d_max, d_min = daily_hist.mean(), daily_hist.max(), daily_hist.min()

        row.update({
            "Avg_21D": round(d_avg, 2),
            "D_Avg_vs_LTP_%": round(((daily_ltp - d_avg) / d_avg) * 100, 2),
            "D_Max_21": round(d_max, 2),
            "D_Min_21": round(d_min, 2),
            "D_LTP_Position": "Above Max" if daily_ltp > d_max else "Below Min" if daily_ltp < d_min else "Between",
            "D_Gap_Min_Max_%": round(((d_max - d_min) / d_min) * 100, 2),
            "D_Gap_Max_LTP_%": round(((daily_ltp - d_max) / d_max) * 100, 2),
            "D_Gap_Min_LTP_%": round(((daily_ltp - d_min) / d_min) * 100, 2),
        })

        return row

    except Exception:
        return {"Symbol": symbol, "Sector": None}


# ===============================
# PARALLEL EXECUTION
# ===============================
final_data = []

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(fetch_data, sym) for sym in symbols]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Stocks", ncols=100):
        final_data.append(future.result())


# ===============================
# DATAFRAME
# ===============================
df = pd.DataFrame(final_data)
df = df.sort_values(by="Symbol").reset_index(drop=True)

candle_cols = sorted([col for col in df.columns if col.startswith("C") and "_" in col])
intraday_30m_cols = sorted([col for col in df.columns if col.startswith("30m_C")])

ordered_cols = (
    ["Symbol", "Sector","Open=High", "Open=Low",]

    + [f"D{i}" for i in range(1, 22)] + ["D_LTP"]
    + [
        "Avg_21D", "D_Avg_vs_LTP_%", "D_Max_21", "D_Min_21",
        "D_Gap_Max_LTP_%", "D_Gap_Min_LTP_%", "D_Gap_Min_Max_%", "D_LTP_Position"
    ]

    + ["Day Live Candle"]
    + candle_cols

    + intraday_30m_cols
    + ["Last_30min_Candle", "LTP_vs_PrevDayClose_%", "RSI_Daily", "RSI_Weekly", "EMA9", "EMA50", "EMA9_vs_EMA50_%"]

)

df = df[ordered_cols]


# ===============================
# SAVE
# ===============================
import os
import json

os.makedirs("data/archive", exist_ok=True)

# Stable file the dashboard will fetch
df.to_excel("data/nifty500_latest.xlsx", index=False)

# Dated archive copy (optional history)
today_date = datetime.now(IST).strftime("%Y-%m-%d")
df.to_excel(f"data/archive/nifty500_daily_scanner_{today_date}.xlsx", index=False)

# Freshness info for dashboard
with open("data/meta.json", "w") as f:
    json.dump({
        "updated_at_ist": datetime.now(IST).isoformat(),
        "rows": len(df),
        "columns": len(df.columns),
    }, f, indent=2)

print(f"Saved: data/nifty500_latest.xlsx ({len(df)} rows)")
