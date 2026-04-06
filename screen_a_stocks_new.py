#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

USER_AGENT = "Mozilla/5.0"


@dataclass
class StockRef:
    code: str
    name: str
    yahoo_symbol: str


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
    mid_start: str
    mid_end: str
    mid_range_pct: float
    mid_trend_pct: float
    range_6m_pct: float
    volume_ratio_6m_vs_prev_6m: float
    latest_price: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="条件new版筛选器（倒序逻辑）")
    parser.add_argument("--history-period", type=str, default="max", choices=["max", "15y", "20y", "25y", "30y"])
    parser.add_argument("--drawdown-lookback-years", type=float, default=15.0, help="条件2：回看年数，默认15")
    parser.add_argument("--max-low-to-peak-ratio", type=float, default=0.3, help="条件2：低点/高点上限")
    parser.add_argument("--mid-window-years", type=float, default=5.0, help="条件3：中期窗口上限年数")
    parser.add_argument("--max-mid-range", type=float, default=1.5, help="条件3：中期振幅上限")
    parser.add_argument("--min-mid-trend-ratio", type=float, default=1.0, help="条件3：中期趋势下限(末/起)")
    parser.add_argument("--recent-range-months", type=int, default=6, help="条件4/5：近期窗口月数")
    parser.add_argument("--max-range-last-months", type=float, default=0.5, help="条件4：近期振幅上限")
    parser.add_argument("--min-volume-ratio", type=float, default=3.0, help="条件5：近期量比下限")
    parser.add_argument("--max-market-cap-yi", type=float, default=200.0, help="条件1：市值上限(亿元)")
    parser.add_argument("--top-n", type=int, default=10, help="输出数量，0表示全部")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-symbols", type=int, default=0, help="最多扫描股票数，0表示全市场")
    parser.add_argument("--symbol-offset", type=int, default=0, help="扫描起始偏移（配合 max-symbols 分批）")
    parser.add_argument("--cache-dir", default=".cache_hist")
    parser.add_argument("--output", type=str, default="screen_results_new.csv", help="输出CSV路径")
    parser.add_argument("--print-universe-count", action="store_true", help="仅输出股票池数量")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def with_retry(func, *args, retries: int = 4, sleep_seconds: float = 1.0, **kwargs):
    last_error = None
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if i == retries - 1:
                break
            time.sleep(sleep_seconds * (i + 1))
    raise last_error


def session_get(url: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    merged_headers = {"User-Agent": USER_AGENT, **headers}
    resp = requests.get(url, headers=merged_headers, timeout=20, **kwargs)
    resp.raise_for_status()
    return resp


def load_csv_symbols(url: str, region: str) -> list[StockRef]:
    r = with_retry(session_get, url)
    df = pd.read_csv(pd.io.common.StringIO(r.text))
    suffix = ".SS" if region == "SH" else ".SZ"
    refs: list[StockRef] = []
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
    filtered = [x for x in refs if re.match(r"^(0|3|6)\d{5}$", x.code) and "ST" not in x.name]
    return list({x.code: x for x in filtered}.values())


def cache_path(cache_dir: Path, code: str, suffix: str) -> Path:
    return cache_dir / f"{code}.{suffix}"


def fetch_hist(stock: StockRef, cache_dir: Path, history_period: str, refresh_cache: bool) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, stock.code, "csv")
    should_refetch = refresh_cache
    if path.exists() and not refresh_cache and (time.time() - path.stat().st_mtime) < 24 * 3600:
        df = pd.read_csv(path)
        try:
            if "日期" in df.columns and len(df) >= 20:
                d = pd.to_datetime(df["日期"], errors="coerce").dropna().sort_values()
                if len(d) >= 20:
                    med = d.diff().dropna().dt.days.median()
                    if med > 7:
                        should_refetch = True
            else:
                should_refetch = True
        except Exception:
            should_refetch = True
    else:
        should_refetch = True

    if should_refetch:
        df = yf.Ticker(stock.yahoo_symbol).history(period=history_period, interval="1d", auto_adjust=False)
        df = df.reset_index()[["Date", "Close", "Volume"]].rename(
            columns={"Date": "日期", "Close": "收盘", "Volume": "成交量"}
        )
        df.to_csv(path, index=False)
        time.sleep(0.05)

    if df.empty:
        return df
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "日期"})
    if "Close" in df.columns:
        df = df.rename(columns={"Close": "收盘"})
    if "Volume" in df.columns:
        df = df.rename(columns={"Volume": "成交量"})
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.tz_localize(None)
    df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
    df["成交量"] = pd.to_numeric(df["成交量"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["日期", "收盘"]).sort_values("日期").reset_index(drop=True)
    return df


def fetch_market_cap_yi(stock: StockRef, cache_dir: Path, refresh_cache: bool) -> Optional[float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, stock.code, "json")
    if path.exists() and not refresh_cache and (time.time() - path.stat().st_mtime) < 24 * 3600:
        payload = json.loads(path.read_text())
    else:
        info = yf.Ticker(stock.yahoo_symbol).info
        payload = {"marketCap": info.get("marketCap"), "currentPrice": info.get("currentPrice"), "shortName": info.get("shortName")}
        path.write_text(json.dumps(payload, ensure_ascii=False))
    mc = payload.get("marketCap")
    return None if mc is None else float(mc) / 1e8


def analyze_stock(
    stock: StockRef,
    hist_cache_dir: Path,
    history_period: str,
    refresh_cache: bool,
    drawdown_lookback_years: float,
    max_low_to_peak_ratio: float,
    mid_window_years: float,
    max_mid_range: float,
    min_mid_trend_ratio: float,
    recent_range_months: int,
    max_range_last_months: float,
    min_volume_ratio: float,
) -> Optional[Candidate]:
    try:
        hist = fetch_hist(stock, hist_cache_dir, history_period, refresh_cache)
    except Exception:
        return None
    if hist.empty or len(hist) < 30:
        return None

    prices = hist["收盘"]
    dates = hist["日期"]
    latest_date = dates.iloc[-1]
    latest_price = float(prices.iloc[-1])

    # 条件2：近15年(不足则上市以来)先找高点，再从高点往后找低点
    drawdown_start = latest_date - pd.Timedelta(days=int(drawdown_lookback_years * 365.25))
    hist_dd = hist[hist["日期"] >= drawdown_start]
    if len(hist_dd) < 2:
        hist_dd = hist
    if len(hist_dd) < 2:
        return None

    peak_local_idx = int(hist_dd["收盘"].idxmax())
    peak_price = float(hist.loc[peak_local_idx, "收盘"])
    peak_date = hist.loc[peak_local_idx, "日期"]
    if peak_price <= 0:
        return None

    post_peak = hist.loc[peak_local_idx:]
    if len(post_peak) < 2:
        return None
    low_local_idx = int(post_peak["收盘"].idxmin())
    low_price = float(hist.loc[low_local_idx, "收盘"])
    low_date = hist.loc[low_local_idx, "日期"]
    if low_price <= 0:
        return None
    low_to_peak_ratio = low_price / peak_price
    if low_to_peak_ratio > max_low_to_peak_ratio:
        return None

    # 条件3：从 low_date 开始，最多看5年（不足就到现在）
    mid_end_date = min(low_date + pd.Timedelta(days=int(mid_window_years * 365.25)), latest_date)
    mid = hist[(hist["日期"] >= low_date) & (hist["日期"] <= mid_end_date)]
    if len(mid) < 2:
        return None
    mid_min = float(mid["收盘"].min())
    mid_max = float(mid["收盘"].max())
    if mid_min <= 0:
        return None
    mid_range = (mid_max - mid_min) / mid_min
    if mid_range > max_mid_range:
        return None
    mid_start_price = float(mid["收盘"].iloc[0])
    mid_end_price = float(mid["收盘"].iloc[-1])
    if mid_start_price <= 0:
        return None
    mid_trend = mid_end_price / mid_start_price
    if mid_trend < min_mid_trend_ratio:
        return None

    # 条件4：最近6个月振幅
    recent_start = latest_date - pd.DateOffset(months=recent_range_months)
    recent = hist[hist["日期"] >= recent_start]
    if len(recent) < 3:
        return None
    recent_min = float(recent["收盘"].min())
    recent_max = float(recent["收盘"].max())
    if recent_min <= 0:
        return None
    range_6m = (recent_max - recent_min) / recent_min
    if range_6m > max_range_last_months:
        return None

    # 条件5：最近6个月放量（对前6个月）
    prev_start = recent_start - pd.DateOffset(months=recent_range_months)
    prev = hist[(hist["日期"] >= prev_start) & (hist["日期"] < recent_start)]
    if len(prev) < 3:
        return None
    prev_avg = float(prev["成交量"].mean())
    recent_avg = float(recent["成交量"].mean())
    if prev_avg <= 0:
        return None
    vol_ratio = recent_avg / prev_avg
    if vol_ratio < min_volume_ratio:
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
        mid_start=mid["日期"].iloc[0].strftime("%Y-%m-%d"),
        mid_end=mid["日期"].iloc[-1].strftime("%Y-%m-%d"),
        mid_range_pct=round(mid_range * 100, 2),
        mid_trend_pct=round(mid_trend * 100, 2),
        range_6m_pct=round(range_6m * 100, 2),
        volume_ratio_6m_vs_prev_6m=round(vol_ratio, 2),
        latest_price=round(latest_price, 2),
    )


def score(c: Candidate) -> tuple[float, float, float]:
    return (-c.low_to_peak_pct, c.volume_ratio_6m_vs_prev_6m, -c.mid_trend_pct)


def main() -> None:
    args = parse_args()
    universe = load_universe()
    if args.print_universe_count:
        print(len(universe))
        return
    if args.symbol_offset and args.symbol_offset > 0:
        universe = universe[args.symbol_offset :]
    if args.max_symbols and args.max_symbols > 0:
        universe = universe[: args.max_symbols]
    hist_cache_dir = Path(args.cache_dir) / "hist"
    info_cache_dir = Path(args.cache_dir) / "info"
    results: list[Candidate] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                analyze_stock,
                stock,
                hist_cache_dir,
                args.history_period,
                args.refresh_cache,
                args.drawdown_lookback_years,
                args.max_low_to_peak_ratio,
                args.mid_window_years,
                args.max_mid_range,
                args.min_mid_trend_ratio,
                args.recent_range_months,
                args.max_range_last_months,
                args.min_volume_ratio,
            )
            for stock in universe
        ]
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            if r is not None:
                results.append(r)

    # 为避免 Yahoo 401 限流，仅对少量候选再查市值做最终过滤
    stock_map = {s.code: s for s in universe}
    rough = sorted(results, key=score, reverse=True)
    if args.top_n <= 0:
        probe_limit = len(rough)
    else:
        probe_limit = max(args.top_n * 12, 120)
    final_results: list[Candidate] = []
    for cand in rough[:probe_limit]:
        try:
            mc = fetch_market_cap_yi(stock_map[cand.code], info_cache_dir, args.refresh_cache)
        except Exception:
            continue
        if mc is None or mc > args.max_market_cap_yi:
            continue
        cand.market_cap_yi = round(mc, 2)
        final_results.append(cand)

    final_results = sorted(final_results, key=score, reverse=True)
    if args.top_n > 0:
        final_results = final_results[: args.top_n]
    if not final_results:
        print("没有找到符合条件的股票。你可以放宽参数后再试。")
        return

    out = pd.DataFrame([x.__dict__ for x in final_results])
    output = Path(args.output)
    out.to_csv(output, index=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    if len(out) > 200:
        print(out.head(200).to_string(index=False))
        print(f"\n仅展示前200条，共 {len(out)} 条。")
    else:
        print(out.to_string(index=False))
    print(f"\n已保存到 {output.resolve()}")


if __name__ == "__main__":
    main()

