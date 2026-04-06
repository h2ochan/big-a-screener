#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "screen_a_stocks_new.py"
RESULT = ROOT / "screen_results_new.csv"


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)


def fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in [x for x in d.columns if x.endswith("_pct")]:
        d[c] = pd.to_numeric(d[c], errors="coerce").map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
    if "volume_ratio_6m_vs_prev_6m" in d.columns:
        raw = pd.to_numeric(d["volume_ratio_6m_vs_prev_6m"], errors="coerce")
        d["volume_ratio_6m_vs_prev_6m"] = raw.map(lambda x: f"{x:.2f}x ({x*100:.2f}%)" if pd.notna(x) else "-")
    d = d.rename(
        columns={
            "code": "股票代码",
            "name": "股票名称",
            "market_cap_yi": "总市值(亿元)",
            "peak_date": "高点日期(条件2)",
            "peak_price": "高点价格(条件2)",
            "low_date": "低点日期(高点后)",
            "low_price": "低点价格(高点后)",
            "low_to_peak_pct": "低点/高点比例",
            "mid_start": "中期窗口起始(条件3)",
            "mid_end": "中期窗口结束(条件3)",
            "mid_range_pct": "中期窗口振幅",
            "mid_trend_pct": "中期窗口趋势(末/起)",
            "range_6m_pct": "近6个月振幅",
            "volume_ratio_6m_vs_prev_6m": "近6个月量比(对前6个月)",
            "latest_price": "最新价",
        }
    )
    return d


st.set_page_config(page_title="条件new筛选网页", layout="wide")
st.title("条件new筛选网页")
st.caption("按 条件new.md 的倒序逻辑：市值 -> 高点后超跌 -> 低点后中期 -> 近6月振幅 -> 近6月放量")

with st.sidebar:
    st.header("参数")
    history_period = st.selectbox("历史周期", ["max", "15y", "20y", "25y", "30y"], index=0)
    drawdown_lookback_years = st.number_input("条件2回看年数", min_value=1.0, max_value=30.0, value=15.0, step=1.0)
    max_low_to_peak_ratio = st.number_input("条件2 低点/高点 上限", min_value=0.01, max_value=1.5, value=0.3, step=0.01)
    mid_window_years = st.number_input("条件3 中期窗口上限(年)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    max_mid_range = st.number_input("条件3 中期振幅上限", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
    min_mid_trend_ratio = st.number_input("条件3 中期趋势下限(末/起)", min_value=0.1, max_value=10.0, value=1.0, step=0.05)
    recent_range_months = st.number_input("条件4/5 近期窗口(月)", min_value=3, max_value=24, value=6, step=1)
    max_range_last_months = st.number_input("条件4 近6月振幅上限", min_value=0.05, max_value=5.0, value=0.5, step=0.05)
    min_volume_ratio = st.number_input("条件5 近6月量比下限", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
    max_market_cap_yi = st.number_input("条件1 市值上限(亿元)", min_value=1.0, max_value=5000.0, value=200.0, step=10.0)
    top_n = st.number_input("输出数量", min_value=1, max_value=200, value=20, step=1)
    workers = st.number_input("并发数", min_value=1, max_value=32, value=8, step=1)
    refresh_cache = st.checkbox("强制刷新缓存", value=False)
    run = st.button("开始筛选", type="primary")

if run:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--history-period",
        history_period,
        "--drawdown-lookback-years",
        str(float(drawdown_lookback_years)),
        "--max-low-to-peak-ratio",
        str(float(max_low_to_peak_ratio)),
        "--mid-window-years",
        str(float(mid_window_years)),
        "--max-mid-range",
        str(float(max_mid_range)),
        "--min-mid-trend-ratio",
        str(float(min_mid_trend_ratio)),
        "--recent-range-months",
        str(int(recent_range_months)),
        "--max-range-last-months",
        str(float(max_range_last_months)),
        "--min-volume-ratio",
        str(float(min_volume_ratio)),
        "--max-market-cap-yi",
        str(float(max_market_cap_yi)),
        "--top-n",
        str(int(top_n)),
        "--workers",
        str(int(workers)),
    ]
    if refresh_cache:
        cmd.append("--refresh-cache")

    with st.spinner("筛选中..."):
        proc = run_cmd(cmd)

    st.subheader("命令输出")
    if proc.stdout.strip():
        st.code(proc.stdout.strip())
    if proc.stderr.strip():
        st.error(proc.stderr.strip())

    if proc.returncode != 0:
        st.error(f"运行失败，退出码 {proc.returncode}")
    else:
        st.success("运行完成")
        if RESULT.exists():
            df = pd.read_csv(RESULT)
            st.subheader(f"结果（{len(df)} 条）")
            st.dataframe(fmt_df(df), use_container_width=True)
            st.download_button("下载CSV", RESULT.read_bytes(), file_name="screen_results_new.csv", mime="text/csv")
        else:
            st.warning("没有找到 screen_results_new.csv")
else:
    st.info("设置参数后点击“开始筛选”。")

