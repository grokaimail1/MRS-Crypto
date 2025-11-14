"""
Delta Exchange crypto breakout scanner with WaveTrend + RSI + Bollinger Bands signals and Telegram alerts.
Auto-runs every 5 minutes.
"""

import time
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime

# -------------------------
# USER CONFIG
# -------------------------

API_KEY = os.environ['API_KEY']
API_SECRET = os.environ['API_SECRET']

TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
CHAT_ID = os.environ['CHAT_ID']

BASE = "https://api.delta.exchange"
INTERVAL = "5m"
CANDLE_LOOKBACK_HOURS = 24
CANDLE_LIMIT = 500
MARKET_MODE = "spot"    # "all" | "spot" | "perpetual"
PAGE_SIZE = 200
SLEEP_BETWEEN_CALLS = 0.25
RUN_INTERVAL = 300      # seconds = 5 minutes

# -------------------------
# AUTH SIGNING
# -------------------------
def sign_headers(method: str, path: str, body: str = "") -> dict:
    if not API_KEY or not API_SECRET:
        return {}
    timestamp = str(int(time.time() * 1000))
    message = timestamp + method.upper() + path + (body or "")
    signature = hmac.new(API_SECRET.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()
    return {"api-key": API_KEY, "timestamp": timestamp, "signature": signature}

def api_request(method, path, params=None, json_body=None, use_auth=False):
    url = BASE + path
    qs = ""
    if params:
        qs = "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    req_path_qs = path + (qs if qs else "")
    body_str = ""
    if json_body:
        import json
        body_str = json.dumps(json_body, separators=(',', ':'))
    headers = sign_headers(method, req_path_qs, body_str) if use_auth else {}

    try:
        if method.upper() == "GET":
            r = requests.get(url, params=params, headers=headers, timeout=15)
        else:
            r = requests.request(method.upper(), url, json=json_body, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# -------------------------
# FETCH PRODUCTS
# -------------------------
def fetch_products():
    print("🔄 Fetching product list...")
    products = {}
    after = None
    while True:
        params = {"page_size": PAGE_SIZE}
        if after:
            params["after"] = after
        resp = api_request("GET", "/v2/products", params=params)
        if "error" in resp:
            print(f"⚠️ Error: {resp['error']}")
            break
        items = resp.get("result", [])
        meta = resp.get("meta", {})
        if not items:
            break
        for p in items:
            display = p.get("display_symbol") or p.get("symbol")
            internal = p.get("symbol")
            state = p.get("state")
            ctype = (p.get("contract_type") or "").lower()
            if "USDT" not in display or state != "live":
                continue
            if MARKET_MODE == "spot" and ctype != "spot":
                continue
            products[display] = internal
        after = meta.get("after")
        print(f"➡️ Total so far: {len(products)} (next cursor: {after})")
        if not after:
            break
        time.sleep(SLEEP_BETWEEN_CALLS)
    return products

# -------------------------
# FETCH CANDLES
# -------------------------
def fetch_candles(symbol):
    end_ts = int(time.time())
    start_ts = end_ts - int(CANDLE_LOOKBACK_HOURS * 3600)
    params = {"symbol": symbol, "resolution": INTERVAL, "start": start_ts, "end": end_ts}
    resp = api_request("GET", "/v2/candles/history", params=params)
    if "error" in resp or not resp.get("result"):
        return None
    df = pd.DataFrame(resp["result"])
    df.rename(columns={"time": "time_s", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    df["date"] = pd.to_datetime(df["time_s"], unit="s")
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    df.sort_values("date", inplace=True)
    return df

def fetch_ticker(symbol):
    resp = api_request("GET", f"/v2/tickers/{symbol}")
    if "error" in resp:
        return None
    data = resp.get("result", {})
    for k in ["mark_price", "last_price", "index_price"]:
        if k in data and data[k]:
            return float(data[k])
    return None

# -------------------------
# INDICATOR FUNCTIONS
# -------------------------
def calculate_wavetrend(df, n1=10, n2=21):
    df["hlc3"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["esa"] = df["hlc3"].ewm(span=n1, adjust=False).mean()
    df["d"] = df["hlc3"].sub(df["esa"]).abs().ewm(span=n1, adjust=False).mean()
    df["ci"] = (df["hlc3"] - df["esa"]) / (0.015 * df["d"])
    df["tci"] = df["ci"].ewm(span=n2, adjust=False).mean()
    df["wt1"] = df["tci"]
    df["wt2"] = df["wt1"].rolling(4).mean()
    return df

def calculate_rsi(df, period=14):
    delta = df["Close"].diff().fillna(0).to_numpy().ravel()
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    gain_series = pd.Series(gain, index=df.index)
    loss_series = pd.Series(loss, index=df.index)
    avg_gain = gain_series.rolling(window=period, min_periods=1).mean()
    avg_loss = loss_series.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2):
    close = df["Close"].astype(float).to_numpy().ravel()
    mid = pd.Series(close, index=df.index).rolling(window=period, min_periods=1).mean()
    std = pd.Series(close, index=df.index).rolling(window=period, min_periods=1).std(ddof=0)
    df["BB_Mid"] = mid
    df["BB_Upper"] = mid + (std_dev * std)
    df["BB_Lower"] = mid - (std_dev * std)
    return df

def generate_signals(df, crossZone=75):
    cross_up = (df["wt1"].shift(1) < df["wt2"].shift(1)) & (df["wt1"] > df["wt2"]) & (df["wt1"] < -crossZone)
    cross_down = (df["wt1"].shift(1) > df["wt2"].shift(1)) & (df["wt1"] < df["wt2"]) & (df["wt1"] > crossZone)
    rsi_buy = df["RSI"].shift(1) < 30
    rsi_sell = df["RSI"].shift(1) > 70
    prev_close_below_lower = df["Close"].shift(1) < df["BB_Lower"].shift(1)
    prev_close_above_upper = df["Close"].shift(1) > df["BB_Upper"].shift(1)
    curr_close_above_lower = df["Close"] > df["BB_Lower"]
    curr_close_below_upper = df["Close"] < df["BB_Upper"]
    green_candle = df["Close"] > df["Open"]
    red_candle = df["Close"] < df["Open"]
    df["BuySignal"] = cross_up & rsi_buy & prev_close_below_lower & curr_close_above_lower & green_candle
    df["SellSignal"] = cross_down & rsi_sell & prev_close_above_upper & curr_close_below_upper & red_candle
    df["Signal"] = np.where(df["BuySignal"], "Buy", np.where(df["SellSignal"], "Sell", ""))
    return df

# -------------------------
# TELEGRAM MESSAGE FUNCTION
# -------------------------
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        return response.ok
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")
        return False

# -------------------------
# SCANNER RUN
# -------------------------
def run_scan():
    products = fetch_products()
    results = []
    for disp, internal in products.items():
        print(f"🔍 Checking {disp} ...")
        df = fetch_candles(internal)
        if df is None or len(df) < 30:
            continue

        # Calculate indicators and signals
        df = calculate_wavetrend(df)
        df = calculate_rsi(df)
        df = calculate_bollinger_bands(df)
        df = generate_signals(df)

        last_signal = df["Signal"].iloc[-1]
        if last_signal in ("Buy", "Sell"):
            price = fetch_ticker(internal)
            signal_time = df["date"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
            results.append({
                "Symbol": disp,
                "Signal": last_signal,
                "Close": df["Close"].iloc[-1],
                "Live_Price": price,
                "Time": signal_time
            })
        time.sleep(SLEEP_BETWEEN_CALLS)
    
    if results:
        for r in results:
            msg = f"*{r['Symbol']}* - {r['Signal']} Signal\nClose: {r['Close']}\nLive Price: {r['Live_Price']}\nTime: {r['Time']}"
            send_telegram_message(msg)
        fname = f"delta_signals_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(results).to_csv(fname, index=False)
        print(f"✅ Saved {len(results)} signals to {fname}")
    else:
        print("❌ No signals found this round.")

# -------------------------
# MAIN (RUNS ONE TIME)
# -------------------------
if __name__ == "__main__":
    print("🚀 Running Delta Crypto Scanner (single run mode)...")
    run_scan()
    print("✅ Scan complete. Exiting.")
