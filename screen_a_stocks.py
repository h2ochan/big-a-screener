#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf


TODAY = datetime.now()
USER_AGENT = "Mozilla/5.0"


def with_retry(func, *args, retries: int = 4, sleep_seconds: float = 1.5, **kwargs):
    last_error = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(sleep_seconds * (attempt + 1))
    raise last_error


@dataclass
class Candidate:
    code: str
    name: str
    market_cap_yi: float
    peak_date: str
    peak_price: float
    low_date: str
    low_price: float
    low_to_peak_pct: float
    range_5y_start: str
    range_5y_end: str
    range_5y_pct: float
    trend_5y_pct: float
    range_6m_pct: float
    volume_ratio_6m_vs_prev_6m: float
    latest_price: float


@dataclass
class StockRef:
    code: str
    name: str
    yahoo_symbol: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从近到远筛选：先近6月放量与振幅，再近5年，最后上市以来超跌与市值。"
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=15,
        help="兼容参数：现在计算基于 `history-period`（默认 `max`）拉取的实际可得历史，不再依赖该参数。",
    )
    parser.add_argument(
        "--history-period",
        type=str,
        default="max",
        choices=["max", "15y", "20y", "25y", "30y"],
        help="yfinance 拉取历史的 period。建议默认 `max` 以覆盖上市以来最高价。",
    )
    parser.add_argument(
        "--max-current-to-peak-ratio",
        type=float,
        default=0.3,
        help="上市以来最低价相对上市以来最高价的最大比例，默认 0.3（即峰谷回撤约 >=70%）。",
    )
    parser.add_argument(
        "--post-peak-window-years",
        type=float,
        default=5.0,
        help="最近中期窗口年数（当前默认按最近 5 年振幅检查）。",
    )
    parser.add_argument(
        "--max-post-peak-range",
        type=float,
        default=1.5,
        help="最近中期窗口最大振幅，默认 1.5。",
    )
    parser.add_argument(
        "--min-5y-trend-ratio",
        type=float,
        default=1.0,
        help="最近5年趋势要求：窗口末值/窗口起值下限，默认 1.0（即整体上行或持平）。",
    )
    parser.add_argument(
        "--max-market-cap-yi",
        "--min-market-cap-yi",
        dest="max_market_cap_yi",
        type=float,
        default=200.0,
        help="总市值上限（亿元），默认 200。兼容旧参数 --min-market-cap-yi。",
    )
    parser.add_argument("--top-n", type=int, default=10, help="输出数量，默认 10")
    parser.add_argument(
        "--recent-range-months",
        type=int,
        default=6,
        help="最近振幅与成交量窗口月数，默认 6 个月",
    )
    parser.add_argument(
        "--max-range-last-months",
        type=float,
        default=0.5,
        help="最近 6 个月最大振幅，默认 0.5",
    )
    parser.add_argument(
        "--min-volume-ratio",
        type=float,
        default=3.0,
        help="最近 6 个月平均成交量相对前 6 个月的最小倍数，默认 3.0",
    )
    parser.add_argument("--workers", type=int, default=6, help="并发数，默认 6")
    parser.add_argument(
        "--cache-dir",
        default=".cache_hist",
        help="历史行情缓存目录，默认 .cache_hist",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="强制刷新历史行情缓存",
    )
    return parser.parse_args()


def session_get(url: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    merged_headers = {"User-Agent": USER_AGENT, **headers}
    response = requests.get(url, headers=merged_headers, timeout=20, **kwargs)
    response.raise_for_status()
    return response


def load_csv_symbols(url: str, region: str) -> list[StockRef]:
    response = with_retry(session_get, url)
    df = pd.read_csv(pd.io.common.StringIO(response.text))
    refs: list[StockRef] = []
    suffix = ".SS" if region == "SH" else ".SZ"
    for _, row in df.iterrows():
        code = str(row["code"]).zfill(6)
        name = str(row["name"]).strip()
        refs.append(StockRef(code=code, name=name, yahoo_symbol=f"{code}{suffix}"))
    return refs


def load_universe() -> list[StockRef]:
    refs = load_csv_symbols(
        "https://raw.githubusercontent.com/irachex/open-stock-data/main/symbols/SSE.csv", "SH"
    ) + load_csv_symbols(
        "https://raw.githubusercontent.com/irachex/open-stock-data/main/symbols/SZSE.csv", "SZ"
    )
    filtered = [ref for ref in refs if re.match(r"^(0|3|6)\d{5}$", ref.code) and "ST" not in ref.name]
    unique: dict[str, StockRef] = {ref.code: ref for ref in filtered}
    return list(unique.values())


def cache_path(cache_dir: Path, code: str, suffix: str) -> Path:
    return cache_dir / f"{code}.{suffix}"


def fetch_hist(
    stock: StockRef,
    cache_dir: Path,
    history_period: str,
    refresh_cache: bool,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_path(cache_dir, stock.code, "csv")
    cache_ttl_hours = 24
    should_refetch = refresh_cache
    if (
        not should_refetch
        and file_path.exists()
        and (time.time() - file_path.stat().st_mtime) < cache_ttl_hours * 3600
    ):
        df = pd.read_csv(file_path)
        # 兼容旧缓存：如果是月线/周线缓存，自动重拉日线覆盖
        if "日期" in df.columns and len(df) >= 3:
            try:
                _tmp_date = pd.to_datetime(df["日期"], errors="coerce").dropna().sort_values()
                if len(_tmp_date) >= 3:
                    diffs = _tmp_date.diff().dropna().dt.days
                    median_diff = float(diffs.median()) if not diffs.empty else 0.0
                    if median_diff > 7:
                        should_refetch = True
            except Exception:
                should_refetch = True
        else:
            should_refetch = True
    if should_refetch or not file_path.exists():
        df = yf.Ticker(stock.yahoo_symbol).history(
            period=history_period, interval="1d", auto_adjust=False
        )
        df = df.reset_index()[["Date", "Close", "Volume"]].rename(
            columns={"Date": "日期", "Close": "收盘", "Volume": "成交量"}
        )
        df.to_csv(file_path, index=False)
        time.sleep(0.1)
    if df.empty:
        return df
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "日期"})
    if "Close" in df.columns:
        df = df.rename(columns={"Close": "收盘"})
    if "Volume" in df.columns:
        df = df.rename(columns={"Volume": "成交量"})
    df["日期"] = pd.to_datetime(df["日期"]).dt.tz_localize(None)
    df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
    df["成交量"] = pd.to_numeric(df["成交量"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["日期", "收盘"]).sort_values("日期").reset_index(drop=True)
    return df


def fetch_market_cap_yi(stock: StockRef, cache_dir: Path, refresh_cache: bool) -> Optional[float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_path(cache_dir, stock.code, "json")
    cache_ttl_hours = 24
    if (
        not refresh_cache
        and file_path.exists()
        and (time.time() - file_path.stat().st_mtime) < cache_ttl_hours * 3600
    ):
        payload = json.loads(file_path.read_text())
    else:
        info = yf.Ticker(stock.yahoo_symbol).info
        payload = {
            "marketCap": info.get("marketCap"),
            "currentPrice": info.get("currentPrice"),
            "shortName": info.get("shortName"),
        }
        file_path.write_text(json.dumps(payload, ensure_ascii=False))
        time.sleep(0.1)
    market_cap = payload.get("marketCap")
    if market_cap is None:
        return None
    return float(market_cap) / 1e8


def analyze_stock(
    stock: StockRef,
    hist_cache_dir: Path,
    history_period: str,
    refresh_cache: bool,
    max_current_to_peak_ratio: float,
    recent_5y_window_years: float,
    max_range_last_5y: float,
    min_5y_trend_ratio: float,
    recent_range_months: int,
    max_range_last_months: float,
    min_volume_ratio: float,
) -> Optional[Candidate]:
    try:
        hist = fetch_hist(stock, hist_cache_dir, history_period, refresh_cache)
    except Exception:
        return None
    if hist.empty:
        return None

    prices = hist["收盘"]
    dates = hist["日期"]
    latest_price = float(prices.iloc[-1])
    latest_date = dates.iloc[-1]

    # 从近到远：先近 6 月放量与振幅，再近 5 年，最后上市以来超跌
    recent_start = latest_date - pd.DateOffset(months=recent_range_months)
    prev_start = recent_start - pd.DateOffset(months=recent_range_months)
    recent_6m = hist[hist["日期"] >= recent_start]
    prev_6m = hist[(hist["日期"] >= prev_start) & (hist["日期"] < recent_start)]
    if len(recent_6m) < 3 or len(prev_6m) < 3:
        return None

    prev_6m_avg_volume = float(prev_6m["成交量"].mean())
    recent_6m_avg_volume = float(recent_6m["成交量"].mean())
    if prev_6m_avg_volume <= 0:
        return None
    volume_ratio = recent_6m_avg_volume / prev_6m_avg_volume
    if volume_ratio < min_volume_ratio:
        return None

    recent_6m_min = float(recent_6m["收盘"].min())
    recent_6m_max = float(recent_6m["收盘"].max())
    if recent_6m_min <= 0:
        return None
    range_6m = (recent_6m_max - recent_6m_min) / recent_6m_min
    if range_6m > max_range_last_months:
        return None

    recent_5y_start = latest_date - pd.Timedelta(days=int(recent_5y_window_years * 365.25))
    recent_5y = hist[hist["日期"] >= recent_5y_start]
    if len(recent_5y) < 2:
        return None
    recent_5y_min = float(recent_5y["收盘"].min())
    recent_5y_max = float(recent_5y["收盘"].max())
    if recent_5y_min <= 0:
        return None
    range_5y = (recent_5y_max - recent_5y_min) / recent_5y_min
    if range_5y > max_range_last_5y:
        return None
    start_price_5y = float(recent_5y["收盘"].iloc[0])
    end_price_5y = float(recent_5y["收盘"].iloc[-1])
    if start_price_5y <= 0:
        return None
    trend_ratio_5y = end_price_5y / start_price_5y
    if trend_ratio_5y < min_5y_trend_ratio:
        return None

    peak_idx = int(prices.idxmax())
    peak_price = float(prices.iloc[peak_idx])
    peak_date = dates.iloc[peak_idx]
    if peak_price <= 0:
        return None

    low_idx = int(prices.idxmin())
    low_price = float(prices.iloc[low_idx])
    low_date = dates.iloc[low_idx]
    if low_price <= 0:
        return None

    low_to_peak_ratio = low_price / peak_price
    if low_to_peak_ratio > max_current_to_peak_ratio:
        return None

    return Candidate(
        code=stock.code,
        name=stock.name,
        market_cap_yi=float("nan"),
        peak_date=peak_date.strftime("%Y-%m-%d"),
        peak_price=round(peak_price, 2),
        low_date=low_date.strftime("%Y-%m-%d"),
        low_price=round(low_price, 2),
        low_to_peak_pct=round(low_to_peak_ratio * 100, 2),
        range_5y_start=recent_5y["日期"].iloc[0].strftime("%Y-%m-%d"),
        range_5y_end=recent_5y["日期"].iloc[-1].strftime("%Y-%m-%d"),
        range_5y_pct=round(range_5y * 100, 2),
        trend_5y_pct=round(trend_ratio_5y * 100, 2),
        range_6m_pct=round(range_6m * 100, 2),
        volume_ratio_6m_vs_prev_6m=round(volume_ratio, 2),
        latest_price=round(latest_price, 2),
    )


def score(candidate: Candidate) -> tuple[float, float, float]:
    return (
        candidate.market_cap_yi,
        -candidate.low_to_peak_pct,
        candidate.volume_ratio_6m_vs_prev_6m,
    )


def main() -> None:
    args = parse_args()
    universe = load_universe()
    hist_cache_dir = Path(args.cache_dir) / "hist"
    info_cache_dir = Path(args.cache_dir) / "info"
    rough_candidates: list[Candidate] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                analyze_stock,
                stock,
                hist_cache_dir,
                args.history_period,
                args.refresh_cache,
                args.max_current_to_peak_ratio,
                args.post_peak_window_years,
                args.max_post_peak_range,
                args.min_5y_trend_ratio,
                args.recent_range_months,
                args.max_range_last_months,
                args.min_volume_ratio,
            )
            for stock in universe
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                rough_candidates.append(result)

    candidates: list[Candidate] = []
    stock_map = {stock.code: stock for stock in universe}
    rough_candidates = sorted(rough_candidates, key=score, reverse=True)
    market_cap_probe_limit = max(args.top_n * 8, 80)
    for candidate in rough_candidates[:market_cap_probe_limit]:
        stock = stock_map[candidate.code]
        try:
            market_cap_yi = fetch_market_cap_yi(stock, info_cache_dir, args.refresh_cache)
        except Exception:
            continue
        if market_cap_yi is None or market_cap_yi > args.max_market_cap_yi:
            continue
        candidate.market_cap_yi = round(market_cap_yi, 2)
        candidates.append(candidate)

    candidates = sorted(candidates, key=score, reverse=True)[: args.top_n]
    if not candidates:
        print("没有找到符合条件的股票。你可以放宽参数后再试。")
        return

    result_df = pd.DataFrame([candidate.__dict__ for candidate in candidates])
    output_path = Path("screen_results.csv")
    result_df.to_csv(output_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(result_df.to_string(index=False))
    print(f"\n已保存到 {output_path.resolve()}")


if __name__ == "__main__":
    main()
