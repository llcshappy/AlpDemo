"""
鱼缸 EVS（事件视觉传感器）行为分析 — 功能演示
展示事件相机在水族监控中的独特优势（所有数据为模拟）
不依赖 pandas，仅需 streamlit + numpy + plotly
"""

from __future__ import annotations

import calendar
from datetime import date
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# 常量 & 样式
# ═══════════════════════════════════════════════════════════════════════════════
SENSOR_W, SENSOR_H = 640, 480
C_ON = "#00e5ff"
C_OFF = "#ff0066"
C_OK = "#4ecdc4"
C_WARN = "#ffe66d"
C_ALERT = "#ff6b6b"

DK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,27,42,0.85)",
    font=dict(color="#c8d6e5"),
)

CSS = """<style>
[data-testid="stMetric"]{background:linear-gradient(135deg,#0d1b2a,#1b2838);
  border:1px solid rgba(0,229,255,.12);border-radius:10px;padding:10px 14px;
  box-shadow:0 0 20px rgba(0,229,255,.04)}
[data-testid="stMetricValue"]{color:#00e5ff;font-size:1.3rem!important}
[data-testid="stMetricLabel"]{color:#8899aa;font-size:.82rem!important}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"]{
  color:#00e5ff!important;border-bottom-color:#00e5ff!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#080c14,#0f1923)}
.hero{text-align:center;padding:10px 18px;
  background:linear-gradient(135deg,#0d1b2a,#162a40);
  border-radius:12px;border:1px solid rgba(0,229,255,.1);margin-bottom:10px}
.hero h3{color:#00e5ff;margin:0;font-size:1.15em}
.hero p{color:#778899;margin:2px 0 0;font-size:.82em}
.block-container{padding-top:1.5rem!important;padding-bottom:1rem!important}
</style>"""


def _hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h3>{title}</h3><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def _render_table(headers: list[str], rows: list[list[str]]) -> None:
    hdr = "".join(f"<th style='padding:5px 10px;text-align:left;border-bottom:"
                  f"2px solid rgba(0,229,255,.25);color:#00e5ff;font-size:.82em'>{h}</th>"
                  for h in headers)
    body = ""
    for r in rows:
        cells = "".join(
            f"<td style='padding:4px 10px;border-bottom:1px solid #1b2838;"
            f"font-size:.82em'>{c}</td>"
            for c in r
        )
        body += f"<tr>{cells}</tr>"
    st.markdown(
        f"<div style='overflow-x:auto'><table style='width:100%;border-collapse:"
        f"collapse'><thead><tr>{hdr}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 模拟数据
# ═══════════════════════════════════════════════════════════════════════════════

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@st.cache_data(show_spinner=False)
def hourly_rates(day: date):
    seed = int(day.strftime("%Y%m%d"))
    rng = _rng(seed)
    hours = np.arange(0, 24, 1 / 6)
    n = len(hours)
    circ = .35 * np.sin((hours - 8) * np.pi / 12)
    rate = 120 * (1 + circ) + rng.normal(0, 25, n)
    rate = np.clip(rate, 10, None).astype(float)

    inject = rng.random() < 0.20
    if inject:
        s = rng.integers(n // 4, n // 3)
        still_len = rng.integers(3, 6)
        for i in range(still_len):
            if s + i < n:
                rate[s + i] = float(rng.uniform(8, 20))
        sp = rng.integers(n * 2 // 3, n * 4 // 5)
        rate[sp] = float(rng.uniform(350, 480))

    lo = float(np.percentile(rate, 8))
    hi = float(np.percentile(rate, 98))
    anom = np.full(n, "normal", dtype=object)
    anom[rate < lo] = "long_still"
    anom[rate > hi] = "sudden_burst"
    labels = [f"{int(h):02d}:{int((h % 1) * 60):02d}" for h in hours]

    hourly_ev = rate.reshape(24, 6).sum(axis=1)
    anom_per_hour = anom.reshape(24, 6)
    hourly_anom = np.array([
        "long_still" if np.any(row == "long_still")
        else ("sudden_burst" if np.any(row == "sudden_burst") else "normal")
        for row in anom_per_hour
    ], dtype=object)

    return hours, rate, anom, labels, hourly_ev, hourly_anom


def _score(rate: np.ndarray, anom: np.ndarray) -> int:
    n_anom = int(np.sum(anom != "normal"))
    base = 92 - n_anom * 4
    if float(np.std(rate) / (np.mean(rate) + 1e-6)) > 0.6:
        base -= 6
    jitter = int(_rng(int(np.sum(rate[:5]) * 100) % 9999).integers(-3, 4))
    return int(np.clip(base + jitter, 0, 100))


@st.cache_data(show_spinner=False)
def month_data(year: int, month: int) -> list[dict]:
    _, dim = calendar.monthrange(year, month)
    seed = year * 100 + month
    rng = _rng(seed)
    rows: list[dict] = []

    base_score = rng.uniform(72, 88)
    trend = rng.uniform(-0.15, 0.15)

    for d in range(1, dim + 1):
        day = date(year, month, d)
        day_rng = _rng(seed * 1000 + d)

        weekday = day.weekday()
        weekend_bump = 3.0 if weekday >= 5 else 0.0

        drift = base_score + trend * (d - dim / 2) + weekend_bump
        daily_noise = day_rng.normal(0, 8)
        score_raw = drift + daily_noise

        roll = day_rng.random()
        if roll < 0.06:
            score_raw = day_rng.uniform(15, 38)
            ar = day_rng.uniform(0.25, 0.45)
        elif roll < 0.22:
            score_raw = day_rng.uniform(42, 68)
            ar = day_rng.uniform(0.12, 0.24)
        else:
            score_raw = np.clip(score_raw, 55, 98)
            ar = day_rng.uniform(0.01, 0.11)

        score = int(np.clip(score_raw, 0, 100))
        ok = ar < 0.12
        base_events = day_rng.uniform(90000, 180000)
        event_factor = 0.5 + 0.5 * (score / 100)
        total_events = int(base_events * event_factor * (1 + day_rng.normal(0, 0.08)))

        rows.append(dict(
            date=day, total_events=total_events,
            ar=ar, ok=ok, score=score,
        ))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 可视化函数
# ═══════════════════════════════════════════════════════════════════════════════

def fig_rate_hourly(hourly_ev: np.ndarray, hourly_anom: np.ndarray) -> go.Figure:
    colors = np.where(hourly_anom == "normal", C_OK,
                      np.where(hourly_anom == "long_still", C_ALERT, C_WARN))
    hours = np.arange(24)
    fig = go.Figure(go.Bar(
        x=hours, y=hourly_ev, marker_color=colors,
        hovertemplate="%{x}:00<br>事件数 %{y:,.0f} / 小时<extra></extra>",
    ))
    fig.update_layout(
        **DK, height=280, bargap=.15,
        title=dict(text="24 小时事件率趋势", font=dict(size=14)),
        xaxis=dict(title="时间", dtick=2, ticksuffix=":00"),
        yaxis=dict(title="事件数 / 小时"),
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def fig_gauge(score: int) -> go.Figure:
    c = C_OK if score >= 70 else (C_WARN if score >= 40 else C_ALERT)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title=dict(text="活跃度评分", font=dict(color="#c8d6e5", size=13)),
        number=dict(font=dict(color=c, size=40)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#445566"),
            bar=dict(color=c, thickness=.75),
            bgcolor="rgba(13,27,42,.8)",
            bordercolor="rgba(0,229,255,.15)",
            steps=[
                dict(range=[0, 40], color="rgba(255,107,107,.08)"),
                dict(range=[40, 70], color="rgba(255,230,109,.08)"),
                dict(range=[70, 100], color="rgba(78,205,196,.08)"),
            ],
        ),
    ))
    fig.update_layout(**DK, height=220, margin=dict(t=36, b=0, l=24, r=24))
    return fig


def fig_month_cal(rows: list[dict], year: int, month: int) -> go.Figure:
    by_d = {r["date"]: r for r in rows}
    cal = calendar.Calendar(firstweekday=0)
    weeks = cal.monthdatescalendar(year, month)
    z, txt = [], []
    for wk in weeks:
        rz, rt = [], []
        for cell in wk:
            if cell.month != month:
                rz.append(None); rt.append("")
            else:
                r = by_d.get(cell)
                if r:
                    rz.append(r["score"] / 100)
                    rt.append(f"{cell.day}日  评分 {r['score']}"
                              f"<br>{'✅ 正常' if r['ok'] else '⚠️ 关注'}")
                else:
                    rz.append(.5); rt.append("")
        z.append(rz); txt.append(rt)
    fig = go.Figure(go.Heatmap(
        z=z, x=["一", "二", "三", "四", "五", "六", "日"],
        y=[f"第{i + 1}周" for i in range(len(z))],
        colorscale=[[0, C_ALERT], [.4, C_WARN], [.7, C_OK], [1, C_ON]],
        zmin=0, zmax=1, text=txt, hoverinfo="text", showscale=False,
    ))
    fig.update_layout(
        **DK, height=240, yaxis_autorange="reversed",
        title=dict(text=f"{year}年{month}月 日活跃度评分", font=dict(size=13)),
        margin=dict(t=36, b=10, l=50, r=10),
    )
    return fig


def fig_month_trend(rows: list[dict]) -> go.Figure:
    ds = [r["date"] for r in rows]
    sc = [r["score"] for r in rows]
    mc = [C_OK if r["ok"] else C_ALERT for r in rows]
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor=C_OK, opacity=.04, line_width=0)
    fig.add_hrect(y0=40, y1=70, fillcolor=C_WARN, opacity=.04, line_width=0)
    fig.add_hrect(y0=0, y1=40, fillcolor=C_ALERT, opacity=.04, line_width=0)
    fig.add_trace(go.Scatter(
        x=ds, y=sc, mode="lines+markers",
        line=dict(color=C_ON, width=2),
        marker=dict(size=6, color=mc, line=dict(color="white", width=.5)),
    ))
    fig.update_layout(
        **DK, height=240,
        title=dict(text="月度评分趋势", font=dict(size=13)),
        xaxis_title="日期",
        yaxis=dict(title="评分", range=[0, 105]),
        margin=dict(t=36, b=40, l=50, r=10),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 页面渲染
# ═══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    _hero("🏠 总览仪表盘", "事件相机实时状态一览")
    today = date.today()
    hrs, rate, anom, _, hourly_ev, hourly_anom = hourly_rates(today)
    score = _score(rate, anom)
    n_anom_hours = int(np.sum(hourly_anom != "normal"))
    total_events = int(np.sum(hourly_ev))
    peak_hour = int(np.argmax(hourly_ev))
    avg_hourly = int(np.mean(hourly_ev))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("今日事件总量", f"{total_events:,}")
    c2.metric("活跃度评分", f"{score} / 100")
    c3.metric("异常小时数", f"{n_anom_hours} / 24")
    c4.metric("平均事件数/时", f"{avg_hourly:,}")

    left, right = st.columns([2.2, 1])
    with left:
        st.plotly_chart(fig_rate_hourly(hourly_ev, hourly_anom),
                        use_container_width=True)
    with right:
        st.plotly_chart(fig_gauge(score), use_container_width=True)

    st.markdown(
        "<span style='font-size:.85em'>"
        "**图例** · <span style='color:#4ecdc4'>■</span> 正常 · "
        "<span style='color:#ff6b6b'>■</span> 长时间不动 · "
        "<span style='color:#ffe66d'>■</span> 突然剧烈运动 · "
        f"峰值时段 **{peak_hour}:00** ({int(hourly_ev[peak_hour]):,} 事件)"
        "</span>",
        unsafe_allow_html=True,
    )

    if n_anom_hours:
        with st.expander(f"今日异常事件 ({n_anom_hours} 个时段)", expanded=False):
            for h in range(24):
                a = str(hourly_anom[h])
                if a == "long_still":
                    st.warning(f"⏱ {h:02d}:00 — 长时间低活动"
                               f"（{int(hourly_ev[h]):,} 事件/小时）")
                elif a == "sudden_burst":
                    st.error(f"⚡ {h:02d}:00 — 突然剧烈运动"
                             f"（{int(hourly_ev[h]):,} 事件/小时）")


def page_monthly():
    _hero("📅 月度健康报告", "按自然月汇总每日行为评分 — 识别长期趋势")
    today = date.today()
    pick = st.date_input("选择月份（任选该月任意一天）", value=today.replace(day=1))
    yr, mo = pick.year, pick.month
    rows = month_data(yr, mo)

    total = len(rows)
    scores = [r["score"] for r in rows]
    ok_days = sum(1 for r in rows if r["ok"])
    warn_days = sum(1 for r in rows if not r["ok"] and r["score"] >= 40)
    alert_days = sum(1 for r in rows if r["score"] < 40)
    avg_score = int(np.mean(scores))
    best_day = max(rows, key=lambda r: r["score"])
    worst_day = min(rows, key=lambda r: r["score"])

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("正常", f"{ok_days}天")
    m2.metric("关注", f"{warn_days}天")
    m3.metric("警告", f"{alert_days}天")
    m4.metric("月均评分", avg_score)
    m5.metric("最佳", f"{best_day['date'].day}日 {best_day['score']}分")
    m6.metric("最差", f"{worst_day['date'].day}日 {worst_day['score']}分")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(fig_month_cal(rows, yr, mo), use_container_width=True)
    with right:
        st.plotly_chart(fig_month_trend(rows), use_container_width=True)

    st.markdown(
        "<span style='font-size:.82em'>"
        "**判定** · 异常占比 < 12% → "
        "<span style='color:#4ecdc4'>✅ 正常</span> · "
        "12%~25% → <span style='color:#ffe66d'>⚠️ 关注</span> · "
        "评分 < 40 → <span style='color:#ff6b6b'>🚨 警告</span>"
        "</span>",
        unsafe_allow_html=True,
    )

    def _day_tag(r: dict) -> str:
        if r["score"] < 40:
            return "🚨 警告"
        return "✅ 正常" if r["ok"] else "⚠️ 关注"

    with st.expander("每日明细", expanded=False):
        _render_table(
            ["日期", "星期", "评分", "异常占比", "判定", "事件总量"],
            [
                [str(r["date"]),
                 ["一", "二", "三", "四", "五", "六", "日"][r["date"].weekday()],
                 str(r["score"]), f"{r['ar']:.1%}",
                 _day_tag(r), f"{r['total_events']:,}"]
                for r in rows
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="EVS 鱼缸行为分析", page_icon="🐟", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

PAGES = {
    "🏠 总览仪表盘": page_dashboard,
    "📅 月度报告": page_monthly,
}

with st.sidebar:
    st.markdown("## 🐟 EVS 鱼缸监控")
    st.caption(f"传感器 {SENSOR_W}×{SENSOR_H} DVS · {date.today():%Y-%m-%d}")
    st.markdown("---")
    page = st.radio("功能导航", list(PAGES.keys()), label_visibility="collapsed")

PAGES[page]()
