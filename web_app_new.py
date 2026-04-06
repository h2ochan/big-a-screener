#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "screen_a_stocks_new.py"
RESULT = ROOT / "screen_results_new.csv"
TMP_DIR = ROOT / ".tmp_batch_results"
FIXED_HISTORY_PERIOD = "max"
FIXED_TOP_N = 0


def run_cmd(args: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )


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


def rank_results(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    needed = ["low_to_peak_pct", "volume_ratio_6m_vs_prev_6m", "mid_trend_pct"]
    if not all(c in df.columns for c in needed):
        if top_n <= 0:
            return df.reset_index(drop=True)
        return df.head(top_n).reset_index(drop=True)
    d = df.copy()
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["_score1"] = -d["low_to_peak_pct"]
    d["_score2"] = d["volume_ratio_6m_vs_prev_6m"]
    d["_score3"] = -d["mid_trend_pct"]
    d = d.sort_values(["_score1", "_score2", "_score3"], ascending=False).drop(columns=["_score1", "_score2", "_score3"])
    if top_n <= 0:
        return d.reset_index(drop=True)
    return d.head(top_n).reset_index(drop=True)


@st.cache_data(ttl=24 * 3600)
def get_universe_count() -> int:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--print-universe-count"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if proc.returncode == 0 and proc.stdout.strip().isdigit():
        return int(proc.stdout.strip())
    return 5000


st.set_page_config(page_title="条件new筛选网页", layout="wide")
st.title("条件new筛选网页")
st.caption("按 条件new.md 的倒序逻辑：市值 -> 高点后超跌 -> 低点后中期 -> 近6月振幅 -> 近6月放量")

with st.sidebar:
    st.header("参数")
    drawdown_lookback_years = st.number_input("回看年数", min_value=1, max_value=30, value=15, step=1)
    max_low_to_peak_pct = st.number_input("低点/高点 上限(%)", min_value=1, max_value=150, value=30, step=1)
    mid_window_years = st.number_input("中期窗口上限(年)", min_value=1, max_value=10, value=5, step=1)
    max_mid_range_pct = st.number_input("中期振幅上限(%)", min_value=10, max_value=1000, value=150, step=5)
    min_mid_trend_pct = st.number_input("中期趋势下限(%)", min_value=10, max_value=1000, value=100, step=5)
    recent_range_months = st.number_input("近期窗口(月)", min_value=3, max_value=24, value=6, step=1)
    max_range_last_months_pct = st.number_input("近6月振幅上限(%)", min_value=5, max_value=500, value=50, step=5)
    min_volume_ratio_pct = st.number_input("近6月量比下限(%)", min_value=10, max_value=2000, value=300, step=10)
    max_market_cap_yi = st.number_input("市值上限(亿元)", min_value=1, max_value=5000, value=200, step=10)
    run_mode = st.selectbox("运行模式", ["快速模式（分批+进度）", "全量模式（单次）"], index=0)
    batch_size = st.number_input("快速模式每批扫描数", min_value=50, max_value=2000, value=500, step=50)
    timeout_seconds = st.number_input(
        "单次运行超时（秒）",
        min_value=60,
        max_value=1800,
        value=420,
        step=30,
    )
    workers = st.number_input("并发数", min_value=1, max_value=32, value=8, step=1)
    refresh_cache = st.checkbox("强制刷新缓存", value=False)
    st.caption("历史周期固定为 max；扫描范围固定为全市场；输出固定为全部。")
    run = st.button("开始筛选", type="primary")

if "last_result_df" not in st.session_state:
    st.session_state["last_result_df"] = None
if "last_stdout" not in st.session_state:
    st.session_state["last_stdout"] = ""
if "last_stderr" not in st.session_state:
    st.session_state["last_stderr"] = ""

if run:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--history-period",
        FIXED_HISTORY_PERIOD,
        "--drawdown-lookback-years",
        str(int(drawdown_lookback_years)),
        "--max-low-to-peak-ratio",
        str(int(max_low_to_peak_pct) / 100.0),
        "--mid-window-years",
        str(int(mid_window_years)),
        "--max-mid-range",
        str(int(max_mid_range_pct) / 100.0),
        "--min-mid-trend-ratio",
        str(int(min_mid_trend_pct) / 100.0),
        "--recent-range-months",
        str(int(recent_range_months)),
        "--max-range-last-months",
        str(int(max_range_last_months_pct) / 100.0),
        "--min-volume-ratio",
        str(int(min_volume_ratio_pct) / 100.0),
        "--max-market-cap-yi",
        str(int(max_market_cap_yi)),
        "--top-n",
        str(FIXED_TOP_N),
        "--workers",
        str(int(workers)),
    ]
    if refresh_cache:
        cmd.append("--refresh-cache")
    if run_mode.startswith("全量模式"):
        with st.spinner("筛选中..."):
            try:
                proc = run_cmd(cmd, timeout_seconds=int(timeout_seconds))
            except subprocess.TimeoutExpired:
                st.error(
                    f"运行超时（>{int(timeout_seconds)}秒）。建议把并发调到 2~4 再试。"
                )
                st.stop()

        st.session_state["last_stdout"] = proc.stdout.strip()
        st.session_state["last_stderr"] = proc.stderr.strip()

        if proc.returncode != 0:
            st.error(f"运行失败，退出码 {proc.returncode}")
        else:
            st.success("运行完成")
            if RESULT.exists():
                df = pd.read_csv(RESULT)
                st.session_state["last_result_df"] = df
            else:
                st.warning("没有找到 screen_results_new.csv")
    else:
        total_symbols = get_universe_count()
        step = int(batch_size)
        total_batches = (total_symbols + step - 1) // step
        TMP_DIR.mkdir(parents=True, exist_ok=True)

        progress = st.progress(0, text="准备开始分批扫描...")
        status = st.empty()
        outputs: list[str] = []
        frames: list[pd.DataFrame] = []

        for i in range(total_batches):
            offset = i * step
            current = min(step, total_symbols - offset)
            batch_file = TMP_DIR / f"batch_{uuid4().hex}.csv"
            batch_cmd = cmd.copy()
            batch_cmd.extend(["--symbol-offset", str(offset), "--max-symbols", str(current), "--output", str(batch_file)])
            status.info(f"第 {i + 1}/{total_batches} 批：扫描 {offset + 1}~{offset + current} ...")

            try:
                proc = run_cmd(batch_cmd, timeout_seconds=int(timeout_seconds))
            except subprocess.TimeoutExpired:
                status.error(f"第 {i + 1} 批超时（>{int(timeout_seconds)}秒），已停止。")
                st.stop()

            if proc.stdout.strip():
                outputs.append(f"--- Batch {i + 1}/{total_batches} ---\n{proc.stdout.strip()}")
            if proc.returncode != 0:
                status.error(f"第 {i + 1} 批失败，退出码 {proc.returncode}")
                if proc.stderr.strip():
                    st.error(proc.stderr.strip())
                st.stop()

            if batch_file.exists():
                try:
                    batch_df = pd.read_csv(batch_file)
                    if not batch_df.empty:
                        frames.append(batch_df)
                except Exception:
                    pass
            progress.progress((i + 1) / total_batches, text=f"已完成 {i + 1}/{total_batches} 批")

        status.success(f"快速模式完成，共 {total_batches} 批。")
        if outputs:
            st.session_state["last_stdout"] = "\n\n".join(outputs)
            st.session_state["last_stderr"] = ""

        if frames:
            all_df = pd.concat(frames, ignore_index=True)
            all_df = all_df.drop_duplicates(subset=["code"], keep="first")
            final_df = rank_results(all_df, FIXED_TOP_N)
            final_df.to_csv(RESULT, index=False)
            st.session_state["last_result_df"] = final_df
        else:
            st.warning("分批扫描完成，但没有命中结果。")

if st.session_state.get("last_stdout"):
    st.subheader("命令输出")
    st.code(st.session_state["last_stdout"])
if st.session_state.get("last_stderr"):
    st.error(st.session_state["last_stderr"])

last_df = st.session_state.get("last_result_df")
if isinstance(last_df, pd.DataFrame) and not last_df.empty:
    st.subheader(f"结果（{len(last_df)} 条）")
    st.dataframe(fmt_df(last_df), use_container_width=True)
    st.download_button("下载CSV", RESULT.read_bytes(), file_name="screen_results_new.csv", mime="text/csv")
elif not run:
    st.info("设置参数后点击“开始筛选”。")

