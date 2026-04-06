#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "screen_a_stocks.py"
RESULT_CSV = ROOT / "screen_results.csv"
HISTORY_JSON = ROOT / "screen_run_history.json"
MAX_HISTORY = 30

PRESETS = {
    "严格版": {
        "history_period": "max",
        "max_current_to_peak_ratio": 0.3,
        "recent_5y_window_years": 5.0,
        "max_range_last_5y": 1.5,
        "min_5y_trend_ratio": 1.0,
        "recent_range_months": 6,
        "max_range_last_months": 0.5,
        "min_volume_ratio": 3.0,
        "max_market_cap_yi": 200.0,
        "top_n": 10,
        "workers": 8,
        "refresh_cache": False,
    },
    "C方案": {
        "history_period": "max",
        "max_current_to_peak_ratio": 0.4,
        "recent_5y_window_years": 5.0,
        "max_range_last_5y": 1.5,
        "min_5y_trend_ratio": 1.0,
        "recent_range_months": 6,
        "max_range_last_months": 0.5,
        "min_volume_ratio": 2.0,
        "max_market_cap_yi": 200.0,
        "top_n": 10,
        "workers": 8,
        "refresh_cache": False,
    },
    "宽松版": {
        "history_period": "max",
        "max_current_to_peak_ratio": 0.5,
        "recent_5y_window_years": 5.0,
        "max_range_last_5y": 2.5,
        "min_5y_trend_ratio": 0.9,
        "recent_range_months": 6,
        "max_range_last_months": 0.8,
        "min_volume_ratio": 1.5,
        "max_market_cap_yi": 500.0,
        "top_n": 20,
        "workers": 8,
        "refresh_cache": False,
    },
}


def load_history() -> list[dict]:
    if not HISTORY_JSON.exists():
        return []
    try:
        data = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def append_history(entry: dict) -> None:
    history = load_history()
    history.insert(0, entry)
    HISTORY_JSON.write_text(
        json.dumps(history[:MAX_HISTORY], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def apply_preset_to_state(name: str) -> None:
    preset = PRESETS[name]
    for key, value in preset.items():
        st.session_state[key] = value


def build_command(
    history_period: str,
    max_current_to_peak_ratio: float,
    recent_5y_window_years: float,
    max_range_last_5y: float,
    min_5y_trend_ratio: float,
    recent_range_months: int,
    max_range_last_months: float,
    min_volume_ratio: float,
    max_market_cap_yi: float,
    top_n: int,
    workers: int,
    refresh_cache: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--history-period",
        history_period,
        "--max-current-to-peak-ratio",
        str(max_current_to_peak_ratio),
        "--post-peak-window-years",
        str(recent_5y_window_years),
        "--max-post-peak-range",
        str(max_range_last_5y),
        "--min-5y-trend-ratio",
        str(min_5y_trend_ratio),
        "--recent-range-months",
        str(recent_range_months),
        "--max-range-last-months",
        str(max_range_last_months),
        "--min-volume-ratio",
        str(min_volume_ratio),
        "--max-market-cap-yi",
        str(max_market_cap_yi),
        "--top-n",
        str(top_n),
        "--workers",
        str(workers),
    ]
    if refresh_cache:
        cmd.append("--refresh-cache")
    return cmd


def run_screen(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def format_result_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    percent_columns = [col for col in display_df.columns if col.endswith("_pct")]
    for col in percent_columns:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").map(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "-"
        )

    # 量比既保留倍数语义，也补充百分比，方便直观比较
    ratio_col = "volume_ratio_6m_vs_prev_6m"
    if ratio_col in display_df.columns:
        ratio_raw = pd.to_numeric(display_df[ratio_col], errors="coerce")
        display_df[ratio_col] = ratio_raw.map(
            lambda x: f"{x:.2f}x ({x * 100:.2f}%)" if pd.notna(x) else "-"
        )

    column_name_map = {
        "code": "股票代码",
        "name": "股票名称",
        "market_cap_yi": "总市值(亿元)",
        "peak_date": "上市以来最高点日期",
        "peak_price": "上市以来最高价",
        "low_date": "上市以来最低点日期",
        "low_price": "上市以来最低价",
        "low_to_peak_pct": "历史低点占历史高点比例",
        "range_5y_start": "近5年窗口起始",
        "range_5y_end": "近5年窗口结束",
        "range_5y_pct": "近5年振幅",
        "trend_5y_pct": "近5年趋势(末值/起值)",
        "range_6m_pct": "近6个月振幅",
        "volume_ratio_6m_vs_prev_6m": "近6个月量比(对前6个月)",
        "latest_price": "最新价",
    }
    display_df = display_df.rename(columns=column_name_map)
    return display_df


st.set_page_config(page_title="大A选股参数工具", layout="wide")
st.title("大A选股参数工具")
st.caption("左侧改参数，点击运行后自动调用 screen_a_stocks.py")

with st.sidebar:
    st.header("参数")
    preset_name = st.selectbox("参数预设", list(PRESETS.keys()), index=0)
    if st.button("应用预设"):
        apply_preset_to_state(preset_name)

    if "history_period" not in st.session_state:
        apply_preset_to_state(preset_name)

    history_period = st.selectbox(
        "历史周期",
        ["max", "15y", "20y", "25y", "30y"],
        key="history_period",
    )
    max_current_to_peak_ratio = st.number_input(
        "历史最低价/上市以来高点 上限",
        min_value=0.01,
        max_value=2.0,
        key="max_current_to_peak_ratio",
        step=0.01,
    )
    recent_5y_window_years = st.number_input(
        "中期窗口（年）",
        min_value=1.0,
        max_value=10.0,
        key="recent_5y_window_years",
        step=0.5,
    )
    max_range_last_5y = st.number_input(
        "中期窗口振幅上限 (max-min)/min",
        min_value=0.1,
        max_value=10.0,
        key="max_range_last_5y",
        step=0.1,
    )
    min_5y_trend_ratio = st.number_input(
        "近5年趋势下限 (末值/起值)",
        min_value=0.1,
        max_value=10.0,
        key="min_5y_trend_ratio",
        step=0.05,
    )
    recent_range_months = st.number_input(
        "近期窗口（月）",
        min_value=3,
        max_value=24,
        key="recent_range_months",
        step=1,
    )
    max_range_last_months = st.number_input(
        "近期窗口振幅上限 (max-min)/min",
        min_value=0.05,
        max_value=5.0,
        key="max_range_last_months",
        step=0.05,
    )
    min_volume_ratio = st.number_input(
        "近窗口量比下限 (近N月/前N月)",
        min_value=0.1,
        max_value=20.0,
        key="min_volume_ratio",
        step=0.1,
    )
    max_market_cap_yi = st.number_input(
        "市值上限（亿元）",
        min_value=0.0,
        max_value=5000.0,
        key="max_market_cap_yi",
        step=10.0,
    )
    top_n = st.number_input("输出数量", min_value=1, max_value=100, key="top_n", step=1)
    workers = st.number_input("并发数", min_value=1, max_value=32, key="workers", step=1)
    refresh_cache = st.checkbox("强制刷新缓存", key="refresh_cache")
    run_btn = st.button("开始筛选", type="primary")

if run_btn:
    cmd = build_command(
        history_period=history_period,
        max_current_to_peak_ratio=float(max_current_to_peak_ratio),
        recent_5y_window_years=float(recent_5y_window_years),
        max_range_last_5y=float(max_range_last_5y),
        min_5y_trend_ratio=float(min_5y_trend_ratio),
        recent_range_months=int(recent_range_months),
        max_range_last_months=float(max_range_last_months),
        min_volume_ratio=float(min_volume_ratio),
        max_market_cap_yi=float(max_market_cap_yi),
        top_n=int(top_n),
        workers=int(workers),
        refresh_cache=refresh_cache,
    )
    with st.spinner("筛选中，请稍候..."):
        proc = run_screen(cmd)

    st.subheader("命令输出")
    if proc.stdout.strip():
        st.code(proc.stdout.strip())
    if proc.stderr.strip():
        st.error(proc.stderr.strip())

    if proc.returncode != 0:
        st.error(f"运行失败，退出码: {proc.returncode}")
    else:
        st.success("运行完成")
        result_rows = 0
        if RESULT_CSV.exists():
            df = pd.read_csv(RESULT_CSV)
            result_rows = len(df)
            st.subheader(f"结果（{len(df)} 条）")
            st.dataframe(format_result_df_for_display(df), use_container_width=True)
            st.download_button(
                label="下载结果 CSV",
                data=RESULT_CSV.read_bytes(),
                file_name="screen_results.csv",
                mime="text/csv",
            )
        else:
            st.warning("未找到结果文件 screen_results.csv")

        append_history(
            {
                "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "preset": preset_name,
                "result_rows": result_rows,
                "history_period": history_period,
                "max_current_to_peak_ratio": float(max_current_to_peak_ratio),
                "recent_5y_window_years": float(recent_5y_window_years),
                "max_range_last_5y": float(max_range_last_5y),
                "min_5y_trend_ratio": float(min_5y_trend_ratio),
                "recent_range_months": int(recent_range_months),
                "max_range_last_months": float(max_range_last_months),
                "min_volume_ratio": float(min_volume_ratio),
                "max_market_cap_yi": float(max_market_cap_yi),
                "top_n": int(top_n),
                "workers": int(workers),
                "refresh_cache": bool(refresh_cache),
            }
        )
else:
    st.info("先在左侧设置参数，再点击“开始筛选”。")

history = load_history()
if history:
    st.subheader("最近运行记录")
    st.dataframe(pd.DataFrame(history), use_container_width=True)
