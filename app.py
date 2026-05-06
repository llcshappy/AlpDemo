"""
鱼缸 EVS（事件视觉传感器）行为分析 — 功能演示
展示事件相机在水族监控中的独特优势（所有数据为模拟）
不依赖 pandas，仅需 streamlit + numpy + plotly
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
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
FISH_PALETTE = ["#00e5ff", "#ff0066", "#ffe66d", "#4ecdc4", "#ff9f43", "#a29bfe"]

DK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,27,42,0.85)",
    font=dict(color="#c8d6e5"),
)

CSS = """<style>
[data-testid="stMetric"]{background:linear-gradient(135deg,#0d1b2a,#1b2838);
  border:1px solid rgba(0,229,255,.12);border-radius:12px;padding:14px 18px;
  box-shadow:0 0 20px rgba(0,229,255,.04)}
[data-testid="stMetricValue"]{color:#00e5ff}
[data-testid="stMetricLabel"]{color:#8899aa}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"]{
  color:#00e5ff!important;border-bottom-color:#00e5ff!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#080c14,#0f1923)}
.hero{text-align:center;padding:18px 24px;
  background:linear-gradient(135deg,#0d1b2a,#162a40);
  border-radius:14px;border:1px solid rgba(0,229,255,.1);margin-bottom:16px}
.hero h3{color:#00e5ff;margin:0}.hero p{color:#778899;margin:4px 0 0;font-size:.92em}
.adv-card{background:linear-gradient(135deg,#0d1b2a,#162a40);
  border:1px solid rgba(0,229,255,.08);border-radius:12px;padding:18px;margin:8px 0}
.adv-card h4{color:#00e5ff;margin:0 0 6px}.adv-card p{color:#aabbcc;margin:0;font-size:.88em}
</style>"""


def _hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h3>{title}</h3><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def _render_table(headers: list[str], rows: list[list[str]]) -> None:
    """Pure-HTML table — avoids st.dataframe which requires pyarrow."""
    hdr = "".join(f"<th style='padding:8px 14px;text-align:left;border-bottom:"
                  f"2px solid rgba(0,229,255,.25);color:#00e5ff'>{h}</th>"
                  for h in headers)
    body = ""
    for r in rows:
        cells = "".join(
            f"<td style='padding:6px 14px;border-bottom:1px solid #1b2838'>{c}</td>"
            for c in r
        )
        body += f"<tr>{cells}</tr>"
    st.markdown(
        f"<div style='overflow-x:auto'><table style='width:100%;border-collapse:"
        f"collapse;font-size:.9em'><thead><tr>{hdr}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 模拟数据
# ═══════════════════════════════════════════════════════════════════════════════

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@st.cache_data(show_spinner=False)
def fish_paths(n_fish: int = 4, n_pts: int = 300, dur: float = 10.0, seed: int = 42):
    rng = _rng(seed)
    t = np.linspace(0, dur, n_pts)
    out: list[dict[str, Any]] = []
    for i in range(n_fish):
        fx, fy = rng.uniform(.08, .4), rng.uniform(.08, .4)
        px, py = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
        ax, ay = rng.uniform(80, 250), rng.uniform(60, 180)
        cx = rng.uniform(180, SENSOR_W - 180)
        cy = rng.uniform(140, SENSOR_H - 140)
        x = cx + ax * np.sin(2 * np.pi * fx * t + px) + rng.normal(0, 3, n_pts)
        y = cy + ay * np.sin(2 * np.pi * fy * t + py) + rng.normal(0, 3, n_pts)
        x, y = np.clip(x, 12, SENSOR_W - 12), np.clip(y, 12, SENSOR_H - 12)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        speed = np.sqrt(dx ** 2 + dy ** 2) / (dur / n_pts)
        out.append(dict(x=x, y=y, t=t, speed=speed, id=i))
    return out


@st.cache_data(show_spinner=False)
def sim_events(n_fish: int = 4, epf: int = 2000, seed: int = 42):
    paths = fish_paths(n_fish=n_fish, seed=seed)
    rng = _rng(seed + 7)
    xs, ys, ts, ps = [], [], [], []
    for p in paths:
        n = len(p["t"])
        w = np.asarray(p["speed"], dtype=float) + 1e-6
        w /= w.sum()
        idx = rng.choice(n, size=epf, p=w)
        xs.append(np.asarray(p["x"])[idx] + rng.normal(0, 6, epf))
        ys.append(np.asarray(p["y"])[idx] + rng.normal(0, 5, epf))
        ts.append(np.asarray(p["t"])[idx] + rng.uniform(-.02, .02, epf))
        ps.append(rng.integers(0, 2, epf))
    nn = 800
    xs.append(rng.uniform(0, SENSOR_W, nn))
    ys.append(rng.uniform(0, SENSOR_H, nn))
    ts.append(rng.uniform(0, 10, nn))
    ps.append(rng.integers(0, 2, nn))
    ex = np.clip(np.concatenate(xs), 0, SENSOR_W)
    ey = np.clip(np.concatenate(ys), 0, SENSOR_H)
    et = np.concatenate(ts)
    ep = np.concatenate(ps).astype(int)
    order = np.argsort(et)
    return ex[order], ey[order], et[order], ep[order]


@st.cache_data(show_spinner=False)
def hourly_rates(day: date):
    seed = int(day.strftime("%Y%m%d"))
    rng = _rng(seed)
    hours = np.arange(0, 24, 1 / 6)
    n = len(hours)
    circ = .35 * np.sin((hours - 8) * np.pi / 12)
    rate = 120 * (1 + circ) + rng.normal(0, 25, n)
    rate = np.clip(rate, 10, None).astype(float)

    # ~20% 的天注入异常，其余天完全正常
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
    return hours, rate, anom, labels


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

# ---------- 事件流动态回放 ----------
def fig_replay(ex, ey, et, ep, n_frames: int = 35, window: float = .4) -> go.Figure:
    t0 = float(et.min()) + window
    t1 = float(et.max())
    fts = np.linspace(t0, t1, n_frames)

    frames: list[go.Frame] = []
    for ft in fts:
        m = (et >= ft - window) & (et < ft)
        c = np.where(ep[m] == 1, C_ON, C_OFF)
        frames.append(go.Frame(
            data=[go.Scatter(
                x=ex[m], y=ey[m], mode="markers",
                marker=dict(size=3, color=c, opacity=.75),
            )],
            name=f"{ft:.2f}",
        ))

    m0 = (et >= fts[0] - window) & (et < fts[0])
    c0 = np.where(ep[m0] == 1, C_ON, C_OFF)
    fig = go.Figure(
        data=[go.Scatter(x=ex[m0], y=ey[m0], mode="markers",
                           marker=dict(size=3, color=c0, opacity=.75))],
        frames=frames,
    )
    fig.update_layout(
        **DK, height=520,
        title="事件流回放 — 滑动时间窗口",
        xaxis=dict(range=[0, SENSOR_W], title="X (px)", showgrid=False),
        yaxis=dict(range=[SENSOR_H, 0], title="Y (px)", showgrid=False, scaleanchor="x"),
        updatemenus=[dict(
            type="buttons", x=.08, y=-.06, direction="left",
            buttons=[
                dict(label="▶ 播放", method="animate",
                     args=[None, dict(frame=dict(duration=70, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸", method="animate",
                     args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name], dict(frame=dict(duration=70), mode="immediate")],
                        label=f.name, method="animate") for f in frames],
            x=.08, len=.84, y=-.02,
            currentvalue=dict(prefix="t = ", suffix=" s", font=dict(color=C_ON)),
        )],
    )
    return fig


# ---------- 运动轨迹动画 ----------
def fig_trajectory_anim(paths: list[dict], n_frames: int = 30) -> go.Figure:
    n_pts = len(paths[0]["t"])
    indices = np.linspace(10, n_pts - 1, n_frames, dtype=int)

    trail_len = max(60, n_pts // 4)

    def _build(end: int) -> list[go.Scatter]:
        traces: list[go.Scatter] = []
        hx, hy, hc = [], [], []
        for p in paths:
            c = FISH_PALETTE[p["id"] % len(FISH_PALETTE)]
            start = max(0, end - trail_len)
            seg_x = np.asarray(p["x"])[start:end]
            seg_y = np.asarray(p["y"])[start:end]
            traces.append(go.Scatter(
                x=seg_x, y=seg_y,
                mode="lines", line=dict(color=c, width=2), opacity=.55,
                showlegend=False,
            ))
            hx.append(float(np.asarray(p["x"])[end - 1]))
            hy.append(float(np.asarray(p["y"])[end - 1]))
            hc.append(c)
        traces.append(go.Scatter(
            x=hx, y=hy, mode="markers",
            marker=dict(size=14, color=hc, line=dict(color="white", width=1.5)),
            showlegend=False,
        ))
        return traces

    frames = [go.Frame(data=_build(int(ei)), name=str(ei)) for ei in indices]
    fig = go.Figure(data=_build(int(indices[0])), frames=frames)
    for i, p in enumerate(paths):
        fig.data[i].name = f"鱼 #{p['id'] + 1}"
        fig.data[i].showlegend = True
    fig.add_shape(type="rect", x0=0, y0=0, x1=SENSOR_W, y1=SENSOR_H,
                  line=dict(color="rgba(0,229,255,.2)", width=1, dash="dot"))
    fig.update_layout(
        **DK, height=520,
        title="鱼体运动轨迹 — 基于 EVS 事件聚类重建",
        xaxis=dict(range=[-10, SENSOR_W + 10], title="X (px)", showgrid=False),
        yaxis=dict(range=[SENSOR_H + 10, -10], title="Y (px)", showgrid=False,
                   scaleanchor="x"),
        updatemenus=[dict(
            type="buttons", x=.08, y=-.06, direction="left",
            buttons=[
                dict(label="▶ 播放", method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸", method="animate",
                     args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
        )],
    )
    return fig


# ---------- 空间活跃热力图 ----------
def fig_heatmap(ex, ey) -> go.Figure:
    H, _, _ = np.histogram2d(ex, ey, bins=[80, 60],
                             range=[[0, SENSOR_W], [0, SENSOR_H]])
    fig = go.Figure(go.Heatmap(
        z=H.T, x=np.linspace(0, SENSOR_W, 80),
        y=np.linspace(0, SENSOR_H, 60),
        colorscale="Hot", showscale=True,
        colorbar=dict(title="密度"),
        hovertemplate="X=%{x:.0f} Y=%{y:.0f}<br>事件密度=%{z:.0f}<extra></extra>",
    ))
    fig.update_layout(**DK, height=500, title="空间活跃热力图 — 事件密度分布",
                      xaxis_title="X (px)", yaxis_title="Y (px)",
                      yaxis=dict(scaleanchor="x"))
    return fig


# ---------- 事件率趋势 ----------
def fig_rate(hours, rate, anom) -> go.Figure:
    colors = np.where(anom == "normal", C_OK,
                      np.where(anom == "long_still", C_ALERT, C_WARN))
    fig = go.Figure(go.Bar(
        x=hours, y=rate, marker_color=colors,
        hovertemplate="时间 %{x:.1f}h<br>事件率 %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(**DK, height=380, bargap=.12,
                      title="24 h 事件率趋势（10 min 桶）",
                      xaxis=dict(title="时间 (h)", dtick=2),
                      yaxis_title="事件数 / 桶")
    return fig


# ---------- 活跃度仪表盘 ----------
def fig_gauge(score: int) -> go.Figure:
    c = C_OK if score >= 70 else (C_WARN if score >= 40 else C_ALERT)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title=dict(text="活跃度评分", font=dict(color="#c8d6e5", size=16)),
        number=dict(font=dict(color=c, size=48)),
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
    fig.update_layout(**DK, height=260, margin=dict(t=50, b=10, l=30, r=30))
    return fig


# ---------- 月历热力 ----------
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
    fig.update_layout(**DK, height=300, yaxis_autorange="reversed",
                      title=f"{year} 年 {month} 月 — 日活跃度评分（绿=优 / 红=关注）")
    return fig


# ---------- 月度趋势 ----------
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
        marker=dict(size=7, color=mc, line=dict(color="white", width=.5)),
    ))
    fig.update_layout(**DK, height=320, title="月度活跃度评分趋势",
                      xaxis_title="日期", yaxis=dict(title="评分", range=[0, 105]))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 页面渲染
# ═══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    _hero("🏠 总览仪表盘", "事件相机实时状态一览")
    today = date.today()
    hrs, rate, anom, _ = hourly_rates(today)
    score = _score(rate, anom)
    n_anom = int(np.sum(anom != "normal"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("今日事件总量", f"{int(np.sum(rate) * 10):,}")
    c2.metric("活跃度评分", f"{score} / 100")
    c3.metric("异常时段", n_anom)
    c4.metric("传感器分辨率", f"{SENSOR_W}×{SENSOR_H}")

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(fig_rate(hrs, rate, anom), use_container_width=True)
    with right:
        st.plotly_chart(fig_gauge(score), use_container_width=True)

    st.markdown("**图例** · <span style='color:#4ecdc4'>■</span> 正常 · "
                "<span style='color:#ff6b6b'>■</span> 长时间不动 · "
                "<span style='color:#ffe66d'>■</span> 突然剧烈运动",
                unsafe_allow_html=True)

    if n_anom:
        st.subheader("今日异常事件")
        idxs = np.where(anom != "normal")[0]
        for i in idxs:
            h = float(hrs[i])
            lbl = f"{int(h):02d}:{int((h % 1) * 60):02d}"
            a = str(anom[i])
            if a == "long_still":
                st.warning(f"⏱ {lbl} — 长时间低活动（事件率 {rate[i]:.0f}）")
            else:
                st.error(f"⚡ {lbl} — 突然剧烈运动（事件率 {rate[i]:.0f}）")



def page_tracking():
    _hero("🐟 运动轨迹重建",
          "基于 EVS 事件聚类，重现 10 秒内鱼缸中每条鱼的游动位置")
    n_fish = st.slider("鱼的数量", 2, 6, 4)
    paths = fish_paths(n_fish=n_fish)
    st.plotly_chart(fig_trajectory_anim(paths, n_frames=30), use_container_width=True)
    st.caption("点击 ▶ 播放观看 10 秒内鱼体游动轨迹动画；圆点 = 鱼当前位置，线条 = 历史路径。")


def page_heatmap():
    _hero("🔥 空间活跃热力图",
          "叠加所有事件的空间密度 — 识别常驻区域、喂食热区与冷僻角落")
    ex, ey, _, _ = sim_events()
    st.plotly_chart(fig_heatmap(ex, ey), use_container_width=True)
    m1, m2 = st.columns(2)
    hot = int(np.sum(ex > SENSOR_W / 2))
    cold = len(ex) - hot
    m1.metric("热区事件占比", f"{hot / len(ex):.0%}")
    m2.metric("冷区事件占比", f"{cold / len(ex):.0%}")


def page_activity():
    _hero("📊 活跃度分析",
          "按日查看事件率趋势、异常标注与活跃度评分")
    pick = st.date_input("选择日期", value=date.today())
    hrs, rate, anom, labels = hourly_rates(pick)
    score = _score(rate, anom)

    g1, g2 = st.columns([2.5, 1])
    with g1:
        st.plotly_chart(fig_rate(hrs, rate, anom), use_container_width=True)
    with g2:
        st.plotly_chart(fig_gauge(score), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("日事件总量", f"{int(np.sum(rate) * 10):,}")
    c2.metric("异常桶数", int(np.sum(anom != "normal")))
    c3.metric("活跃高峰",
              labels[int(np.argmax(rate))] if len(rate) else "–")

    # 时段详情
    st.subheader("时段详情")
    sel = st.selectbox("选择时间段", labels)
    idx = labels.index(sel)
    a = str(anom[idx])
    st.write(f"- **事件率：** {rate[idx]:.0f}")
    if a == "long_still":
        st.warning("该时段事件率极低 — 鱼可能长时间静止或躲藏。")
    elif a == "sudden_burst":
        st.error("事件率骤增 — 可能发生打斗、惊吓或突变光照。")
    else:
        st.success("正常活跃范围。")


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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("总天数", f"{total} 天",
              help=f"正常 {ok_days} · 关注 {warn_days} · 警告 {alert_days}")
    m2.metric("正常天数", f"{ok_days} / {total}",
              delta=f"{ok_days / total:.0%}" if total else "–")
    m3.metric("月均评分", avg_score,
              delta=f"{'优' if avg_score >= 75 else '良' if avg_score >= 60 else '差'}")
    m4.metric("月事件总量", f"{sum(r['total_events'] for r in rows):,}")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("关注天数", f"{warn_days} 天",
              delta=f"{warn_days / total:.0%}" if total else "–",
              delta_color="inverse")
    s2.metric("警告天数", f"{alert_days} 天",
              delta=f"{alert_days / total:.0%}" if total else "–",
              delta_color="inverse")
    s3.metric("最佳日", f"{best_day['date'].day}日 ({best_day['score']}分)")
    s4.metric("最差日", f"{worst_day['date'].day}日 ({worst_day['score']}分)")

    t1, t2 = st.tabs(["月历热力", "趋势曲线"])
    with t1:
        st.plotly_chart(fig_month_cal(rows, yr, mo), use_container_width=True)
    with t2:
        st.plotly_chart(fig_month_trend(rows), use_container_width=True)

    st.markdown(
        "**判定规则** · 异常桶占比 < 12% → "
        "<span style='color:#4ecdc4'>✅ 正常</span> · "
        "12%~25% → <span style='color:#ffe66d'>⚠️ 关注</span> · "
        "评分 < 40 → <span style='color:#ff6b6b'>🚨 警告</span>",
        unsafe_allow_html=True,
    )
    st.subheader("每日明细")

    def _day_tag(r: dict) -> str:
        if r["score"] < 40:
            return "🚨 警告"
        return "✅ 正常" if r["ok"] else "⚠️ 关注"

    _render_table(
        ["日期", "星期", "评分", "异常桶占比", "判定", "事件总量"],
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
    "🐟 运动轨迹重建": page_tracking,
    "🔥 空间热力分析": page_heatmap,
    "📊 活跃度分析": page_activity,
    "📅 月度报告": page_monthly,
}

with st.sidebar:
    st.markdown("## 🐟 EVS 鱼缸监控")
    st.caption(f"传感器 {SENSOR_W}×{SENSOR_H} DVS · {date.today():%Y-%m-%d}")
    st.markdown("---")
    page = st.radio("功能导航", list(PAGES.keys()), label_visibility="collapsed")

PAGES[page]()
