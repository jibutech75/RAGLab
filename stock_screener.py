"""
Daily screener: NSE stocks that are TECHNICALLY turning bearish OR bullish
while remaining FUNDAMENTALLY strong. Produces two separate top-10 lists.

Data source: Yahoo Finance via yfinance (free, stable, no scraping/ToS issues).
Universe: NSE Nifty 500 constituent list (official NSE CSV).

NOT financial advice. Thresholds below are illustrative starting points —
tune them to your own criteria before relying on this for anything real.

Install deps:
    pip install yfinance ta pandas requests --break-system-packages

Run:
    python3 stock_screener.py
"""

import io
import time
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import yfinance as yf
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("screener")

# ---------------------------------------------------------------------------
# Config — tune these thresholds to taste
# ---------------------------------------------------------------------------

NIFTY500_CSV_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
LOCAL_CACHE_CSV = "nifty500_list.csv"     # avoid re-downloading every run
MAX_WORKERS = 8                            # concurrent yfinance fetches
TOP_N = 10

# Technical "turning" thresholds (shared by both directions)
EMA_SHORT = 20
EMA_LONG = 50
RSI_PERIOD = 14
RSI_MOVE_LOOKBACK_DAYS = 5      # window to check for a fast RSI move
RSI_MOVE_THRESHOLD = 10         # RSI must have moved at least this much

# Bearish turning zone: momentum dropping, but not already crashed (RSI < 30)
RSI_BEARISH_UPPER_BOUND = 65
RSI_BEARISH_LOWER_BOUND = 30

# Bullish turning zone: momentum rising, but not already overbought (RSI > 70)
RSI_BULLISH_LOWER_BOUND = 35
RSI_BULLISH_UPPER_BOUND = 70

# Fundamental "strong" thresholds
MIN_ROE = 0.15          # 15% return on equity
MAX_DEBT_TO_EQUITY = 100  # yfinance reports this as a percentage (100 = 1.0x)
MIN_PROFIT_MARGIN = 0.08 # 8%
MIN_EARNINGS_GROWTH = 0.0  # non-negative YoY earnings growth
PE_RANGE = (5, 45)        # sanity band; excludes negative/extreme P/E
FUNDAMENTAL_CRITERIA_REQUIRED = 4  # how many of the 5 checks above must pass


@dataclass
class StockResult:
    symbol: str
    last_close: float = 0.0
    ema_short: float = 0.0
    ema_long: float = 0.0
    rsi_now: float = 0.0
    rsi_prev: float = 0.0
    macd_bearish_cross: bool = False
    macd_bullish_cross: bool = False
    price_below_ema_long: bool = False
    price_above_ema_long: bool = False
    turning_bearish: bool = False
    turning_bullish: bool = False
    bearish_score: int = 0
    bullish_score: int = 0
    direction: str = None   # set after screening: "bearish" or "bullish"
    roe: float = None
    debt_to_equity: float = None
    profit_margin: float = None
    earnings_growth: float = None
    pe_ratio: float = None
    fundamental_checks_passed: int = 0
    fundamentally_strong: bool = False
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def get_nifty500_symbols() -> list[str]:
    """Fetch (or load cached) Nifty 500 list from NSE, return Yahoo-style tickers (.NS suffix)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(NIFTY500_CSV_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.to_csv(LOCAL_CACHE_CSV, index=False)
        symbols = df["Symbol"].tolist()
    except Exception as e:
        log.warning(f"Live NSE fetch failed ({e}), falling back to local cache if present.")
        df = pd.read_csv(LOCAL_CACHE_CSV)
        symbols = df["Symbol"].tolist()

    return [f"{s.strip()}.NS" for s in symbols]


# ---------------------------------------------------------------------------
# Technical analysis
# ---------------------------------------------------------------------------

def analyze_technicals(symbol: str, result: StockResult) -> None:
    hist = yf.Ticker(symbol).history(period="6mo", interval="1d")
    if hist.empty or len(hist) < EMA_LONG + RSI_MOVE_LOOKBACK_DAYS:
        result.notes.append("insufficient price history")
        return

    close = hist["Close"]

    ema_long_series = EMAIndicator(close, window=EMA_LONG).ema_indicator()
    rsi_series = RSIIndicator(close, window=RSI_PERIOD).rsi()
    macd = MACD(close)
    macd_line = macd.macd()
    macd_signal = macd.macd_signal()

    result.last_close = round(float(close.iloc[-1]), 2)
    result.ema_long = round(float(ema_long_series.iloc[-1]), 2)
    result.rsi_now = round(float(rsi_series.iloc[-1]), 2)
    result.rsi_prev = round(float(rsi_series.iloc[-1 - RSI_MOVE_LOOKBACK_DAYS]), 2)

    # --- Direction-agnostic building blocks ---
    was_above_long_ema = close.iloc[-6] > ema_long_series.iloc[-6]
    was_below_long_ema = close.iloc[-6] < ema_long_series.iloc[-6]
    now_below_long_ema = close.iloc[-1] < ema_long_series.iloc[-1]
    now_above_long_ema = close.iloc[-1] > ema_long_series.iloc[-1]

    recent_macd = macd_line.iloc[-4:]
    recent_signal = macd_signal.iloc[-4:]
    crossed_down = ((recent_macd.shift(1) > recent_signal.shift(1)) & (recent_macd < recent_signal)).any()
    crossed_up = ((recent_macd.shift(1) < recent_signal.shift(1)) & (recent_macd > recent_signal)).any()

    rsi_change = result.rsi_now - result.rsi_prev  # negative = falling, positive = rising

    # --- Bearish signals: price broke below EMA, MACD crossed down, RSI falling but not already crashed ---
    result.price_below_ema_long = bool(was_above_long_ema and now_below_long_ema)
    result.macd_bearish_cross = bool(crossed_down)
    rsi_bearish_signal = bool(
        (-rsi_change) >= RSI_MOVE_THRESHOLD
        and RSI_BEARISH_LOWER_BOUND < result.rsi_now < RSI_BEARISH_UPPER_BOUND
    )
    result.bearish_score = sum([result.price_below_ema_long, result.macd_bearish_cross, rsi_bearish_signal])
    result.turning_bearish = result.bearish_score >= 2

    # --- Bullish signals: price broke above EMA, MACD crossed up, RSI rising but not already overbought ---
    result.price_above_ema_long = bool(was_below_long_ema and now_above_long_ema)
    result.macd_bullish_cross = bool(crossed_up)
    rsi_bullish_signal = bool(
        rsi_change >= RSI_MOVE_THRESHOLD
        and RSI_BULLISH_LOWER_BOUND < result.rsi_now < RSI_BULLISH_UPPER_BOUND
    )
    result.bullish_score = sum([result.price_above_ema_long, result.macd_bullish_cross, rsi_bullish_signal])
    result.turning_bullish = result.bullish_score >= 2


# ---------------------------------------------------------------------------
# Fundamental analysis
# ---------------------------------------------------------------------------

def analyze_fundamentals(symbol: str, result: StockResult) -> None:
    info = yf.Ticker(symbol).info  # can be slow; yfinance caches per-session
    if not info:
        result.notes.append("no fundamental info available")
        return

    result.roe = info.get("returnOnEquity")
    result.debt_to_equity = info.get("debtToEquity")
    result.profit_margin = info.get("profitMargins")
    result.earnings_growth = info.get("earningsGrowth")
    result.pe_ratio = info.get("trailingPE")

    checks = 0
    if result.roe is not None and result.roe >= MIN_ROE:
        checks += 1
    if result.debt_to_equity is not None and result.debt_to_equity <= MAX_DEBT_TO_EQUITY:
        checks += 1
    if result.profit_margin is not None and result.profit_margin >= MIN_PROFIT_MARGIN:
        checks += 1
    if result.earnings_growth is not None and result.earnings_growth >= MIN_EARNINGS_GROWTH:
        checks += 1
    if result.pe_ratio is not None and PE_RANGE[0] <= result.pe_ratio <= PE_RANGE[1]:
        checks += 1

    result.fundamental_checks_passed = checks
    result.fundamentally_strong = checks >= FUNDAMENTAL_CRITERIA_REQUIRED


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def screen_symbol(symbol: str) -> StockResult:
    result = StockResult(symbol=symbol)
    try:
        analyze_technicals(symbol, result)
        # only bother with the fundamentals API call if a technical signal already fired
        if result.turning_bearish or result.turning_bullish:
            analyze_fundamentals(symbol, result)
    except Exception as e:
        result.notes.append(f"error: {e}")
    return result


def run_screen() -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = get_nifty500_symbols()
    log.info(f"Screening {len(symbols)} symbols...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(screen_symbol, s): s for s in symbols}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            if i % 50 == 0:
                log.info(f"  processed {i}/{len(symbols)}")

    df = pd.DataFrame([r.__dict__ for r in results])

    bearish = df[(df["turning_bearish"]) & (df["fundamentally_strong"])].copy()
    bearish.sort_values(by=["fundamental_checks_passed", "bearish_score"],
                         ascending=[False, False], inplace=True)

    bullish = df[(df["turning_bullish"]) & (df["fundamentally_strong"])].copy()
    bullish.sort_values(by=["fundamental_checks_passed", "bullish_score"],
                         ascending=[False, False], inplace=True)

    return bearish.head(TOP_N), bullish.head(TOP_N)


def _print_table(df: pd.DataFrame, title: str, score_col: str) -> None:
    cols = [
        "symbol", "last_close", score_col, "rsi_now", "rsi_prev",
        "fundamental_checks_passed", "roe", "debt_to_equity", "pe_ratio",
    ]
    print(f"\n{title}\n")
    if df.empty:
        print("No matches today with current thresholds.")
    else:
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    bearish_top10, bullish_top10 = run_screen()

    _print_table(bearish_top10, "Top 10 — turning BEARISH, fundamentally strong:", "bearish_score")
    _print_table(bullish_top10, "Top 10 — turning BULLISH, fundamentally strong:", "bullish_score")

    date_str = pd.Timestamp.now().date()
    bearish_top10.to_csv(f"screen_bearish_{date_str}.csv", index=False)
    bullish_top10.to_csv(f"screen_bullish_{date_str}.csv", index=False)
