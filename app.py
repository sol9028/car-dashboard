import streamlit as st
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import patches as mpatches
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------
# [Mac] 폰트·마이너스 기호 깨짐 방어
# -------------------------------------------------------
plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------
# ml_pipeline 경로 등록
# -------------------------------------------------------
_PIPELINE_DIR = Path("/Users/spark/매녀졸프/2차 발표 분석/ml_pipeline")
_DATA_ROOT    = Path("/Users/spark/매녀졸프/2차 발표 분석/1.모터_감속기_시계열")
sys.path.insert(0, str(_PIPELINE_DIR))

_ML_CLASSES     = ["DEMAG", "ECC10", "ECC20", "NORMAL", "REDUC"]
CLASS_NAMES     = ["NORMAL", "ECC10", "ECC20", "DEMAG", "REDUC"]
VEHICLES        = ["전체", "IONIQ", "KONA", "NIRO"]
ANOMALY_WARN    = 40    # 경고 임계값
ANOMALY_CAUTION = 20   # 주의 임계값
MAX_HISTORY     = 30   # 트렌드 차트 최대 포인트 수

CLASS_TO_SENSOR = {
    "NORMAL": "Vib_Motor",
    "ECC10":  "Vib_Motor",
    "ECC20":  "Vib_Motor",
    "DEMAG":  "Current_U",
    "REDUC":  "Vib_TM",
}
COLOR_MAP = {
    "NORMAL": "#3B82F6", "ECC10": "#F59E0B",
    "ECC20":  "#EF4444", "DEMAG": "#EC4899", "REDUC": "#8B5CF6",
}
_VID_PREFIX = {"IONIQ": "MOB-I", "KONA": "MOB-K", "NIRO": "MOB-N"}

# -------------------------------------------------------
# ModelRegistry + InferenceRouter 로드
# -------------------------------------------------------
@st.cache_resource
def get_registry_and_router():
    from src.registry         import ModelRegistry
    from src.inference_router import InferenceRouter
    reg = ModelRegistry()
    reg.load_from_config(str(_PIPELINE_DIR / "artifacts" / "config.json"))
    return reg, InferenceRouter(reg)

try:
    _registry, _router = get_registry_and_router()
    model_load_success = True
except Exception:
    model_load_success = False

# -------------------------------------------------------
# 페이지 설정 + 다크 CSS
# -------------------------------------------------------
st.set_page_config(
    page_title="모터워치 — EV 예지보전",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    background-color: #0F172A !important;
    font-family: 'Inter', 'Apple SD Gothic Neo', sans-serif;
    color: #E2E8F0;
}
[data-testid="stHeader"]  { background: rgba(0,0,0,0) !important; }
[data-testid="stSidebar"] { background-color: #0F172A !important; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; }
.kpi-card {
    background: #1E293B; border: 1px solid #334155; border-radius: 12px;
    padding: 18px 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
}
.kpi-title { font-size: 0.75rem; color: #94A3B8; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 5px; }
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #F8FAFC; line-height: 1.2; }
.kpi-sub   { font-size: 0.70rem; color: #38BDF8; margin-top: 5px; font-weight: 500; }
.progress-label { display: flex; justify-content: space-between;
    font-size: 0.80rem; margin-bottom: 4px; font-weight: 500; color: #CBD5E1; }
.progress-bg { background-color: #334155; border-radius: 9999px;
    height: 7px; width: 100%; margin-bottom: 9px; overflow: hidden; }
.progress-fill { height: 100%; border-radius: 9999px; transition: width 0.4s ease-in-out; }
.log-row { display: grid; grid-template-columns: 120px 60px 90px 1fr;
    align-items: center; padding: 8px 14px; border-bottom: 1px solid #1E293B;
    font-size: 0.80rem; font-family: 'Courier New', monospace; }
.log-row:hover { background: rgba(56,189,248,0.04); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 700; text-align: center; }
.badge-warn   { background: rgba(239,68,68,0.18);  color: #EF4444; }
.badge-caution{ background: rgba(245,158,11,0.18); color: #F59E0B; }
.badge-normal { background: rgba(16,185,129,0.15); color: #10B981; }
[data-testid="stExpander"] { background: #1E293B; border: 1px solid #334155; border-radius: 10px; }
[data-testid="stRadio"] > div { gap: 4px !important; }
[data-testid="stRadio"] label {
    background: #1E293B; border: 1px solid #334155; border-radius: 6px;
    padding: 6px 12px !important; font-size: 0.85rem !important; color: #CBD5E1 !important; width: 100%;
}
[data-testid="stButton"] > button {
    background: #1E293B; color: #38BDF8; border: 1px solid #334155;
    border-radius: 8px; font-weight: 600; font-size: 0.85rem;
}
[data-testid="stButton"] > button:hover { border-color: #38BDF8; background: #0F172A; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1E293B; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Fail-Safe
# -------------------------------------------------------
if not model_load_success:
    st.error("🚨 [CRITICAL ALERT] ModelRegistry / InferenceRouter 로드 실패. "
             "하드웨어 인프라 연결을 확인하십시오.")
    st.stop()

# -------------------------------------------------------
# 헬퍼 함수
# -------------------------------------------------------
def _softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def load_rms_signal(csv_path: str, n: int = 2000):
    try:
        import pandas as pd
        return pd.read_csv(csv_path, usecols=["rms"], nrows=n,
                           encoding="utf-8-sig")["rms"].values.astype(np.float32)
    except Exception:
        return None

def generate_fallback_signal(cls: str, n: int = 2000):
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 4 * np.pi, n)
    freq  = {"NORMAL":1.0,"ECC10":2.5,"ECC20":3.5,"DEMAG":1.8,"REDUC":4.0}.get(cls,1.0)
    noise = {"NORMAL":0.05,"ECC10":0.15,"ECC20":0.25,"DEMAG":0.12,"REDUC":0.20}.get(cls,0.1)
    return (np.sin(freq * t) + rng.normal(0, noise, n)).astype(np.float32)

def run_ensemble(sensor_csvs: dict) -> dict:
    per_sensor, total_lat, probs_list = {}, 0.0, []
    for sensor, csv_path in sensor_csvs.items():
        r      = _router.infer(csv_path)
        probs  = _softmax(np.array(r["decision_scores"]).mean(axis=0))
        per_sensor[sensor] = {"probs": probs, "label": r["label"],
                              "latency_ms": r["latency_ms"], "n_chunks": r["n_chunks"]}
        total_lat   += r["latency_ms"]
        probs_list.append(probs)
    avg         = np.mean(probs_list, axis=0)
    model_probs = {cls: float(avg[_ML_CLASSES.index(cls)]) for cls in CLASS_NAMES}
    return {"model_probs": model_probs, "per_sensor": per_sensor,
            "total_latency": round(total_lat, 1)}

def anomaly_score(model_probs: dict) -> float:
    return round((1.0 - model_probs.get("NORMAL", 0.0)) * 100, 1)

def severity(score: float) -> str:
    return "경고" if score >= ANOMALY_WARN else ("주의" if score >= ANOMALY_CAUTION else "정상")

# -------------------------------------------------------
# 세션 카탈로그
# -------------------------------------------------------
@st.cache_data
def build_catalog() -> dict:
    catalog = {}
    for csv_path in sorted(_DATA_ROOT.rglob("*.csv")):
        parts = csv_path.relative_to(_DATA_ROOT).parts
        if len(parts) < 5:
            continue
        vehicle, fault_class, date, sensor = parts[0], parts[1], parts[2], parts[3]
        stem = csv_path.stem.replace(f"_{sensor}_timeseries", "")
        key  = f"{vehicle}|{fault_class}|{date}|{stem}"
        if key not in catalog:
            catalog[key] = {"vehicle": vehicle, "fault_class": fault_class,
                            "date": date, "stem": stem, "sensors": {}}
        catalog[key]["sensors"][sensor] = str(csv_path)
    # 3센서 완비 + 날짜순 정렬
    full = {k: v for k, v in catalog.items() if len(v["sensors"]) == 3}
    return dict(sorted(full.items(), key=lambda x: (x[1]["vehicle"], x[1]["date"])))

catalog = build_catalog()

# -------------------------------------------------------
# 세션 상태 초기화
# -------------------------------------------------------
for _k, _v in [("streaming", False), ("stream_idx", 0),
               ("sel_vehicle", "전체"), ("score_history", []),
               ("alert_log", []), ("ensemble_result", None),
               ("ensemble_key", ""), ("sensor_csvs", {})]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# -------------------------------------------------------
# 헤더
# -------------------------------------------------------
h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(
        '<h1 style="font-size:1.4rem; margin:0; padding:0;">'
        '🚗 MotorWatch 초경량 ML 예지보전 관제 시스템 '
        '<span style="color:#334155; font-size:0.9rem;">Pilot v1.2</span></h1>',
        unsafe_allow_html=True)
with h2:
    s_label = "⏹ 스트리밍 정지" if st.session_state.streaming else "▶ 자동 스트리밍"
    if st.button(s_label, use_container_width=True):
        st.session_state.streaming = not st.session_state.streaming
        st.rerun()

st.write(" ")

# -------------------------------------------------------
# 차종 필터 버튼
# -------------------------------------------------------
st.markdown("**차종 선택**")
v_cols = st.columns(len(VEHICLES))
for col, v in zip(v_cols, VEHICLES):
    with col:
        active = st.session_state.sel_vehicle == v
        style  = ("background:#0EA5E9; color:#fff; border:none;" if active
                  else "background:#1E293B; color:#94A3B8; border:1px solid #334155;")
        if st.button(v, use_container_width=True,
                     key=f"vbtn_{v}",
                     help=f"{v} 차종 세션만 스트리밍"):
            st.session_state.sel_vehicle = v
            st.session_state.stream_idx  = 0
            st.rerun()

# 필터된 세션 리스트 (날짜순)
sel_v      = st.session_state.sel_vehicle
filt_keys  = [k for k, v in catalog.items()
              if sel_v == "전체" or v["vehicle"] == sel_v]
filt_labels = [f"{catalog[k]['vehicle']} / {catalog[k]['fault_class']} / {catalog[k]['date']}"
               for k in filt_keys]

st.caption(f"대상 세션: **{len(filt_keys)}개** | 차종: **{sel_v}** | 날짜순 정렬")
st.write(" ")

# -------------------------------------------------------
# 자동 스트리밍 (날짜순 순차)
# -------------------------------------------------------
if st.session_state.streaming and filt_keys:
    idx     = st.session_state.stream_idx % len(filt_keys)
    cur_key = filt_keys[idx]
    meta    = catalog[cur_key]
    st.info(f"🔄 자동 스트리밍 중 — [{idx+1}/{len(filt_keys)}]  "
            f"`{meta['vehicle']} / {meta['fault_class']} / {meta['date']}`")
    try:
        res = run_ensemble(meta["sensors"])
        st.session_state["ensemble_result"] = res
        st.session_state["ensemble_key"]    = cur_key
        st.session_state["sensor_csvs"]     = meta["sensors"]

        # 이상 점수 히스토리 누적
        a_score = anomaly_score(res["model_probs"])
        sev     = severity(a_score)
        pri     = max(res["model_probs"], key=res["model_probs"].get)
        vid     = _VID_PREFIX.get(meta["vehicle"], "MOB-?") + f"{idx:03d}"
        now_str = time.strftime("%H:%M:%S")
        st.session_state.score_history.append({
            "ts": now_str, "score": a_score, "vehicle": meta["vehicle"],
            "fault_class": meta["fault_class"], "primary": pri,
        })
        if len(st.session_state.score_history) > MAX_HISTORY:
            st.session_state.score_history.pop(0)

        # 알림 로그 누적 (주의 이상만)
        if sev in ("경고", "주의"):
            st.session_state.alert_log.insert(0, {
                "ts": now_str, "severity": sev, "vid": vid,
                "vehicle": meta["vehicle"],
                "msg": f"[자동 스트리밍] {meta['fault_class']}: 이상점수 {a_score}",
                "primary": pri, "conf": res["model_probs"][pri] * 100,
            })
            if len(st.session_state.alert_log) > 50:
                st.session_state.alert_log.pop()

        st.session_state.stream_idx += 1
    except Exception:
        st.session_state.stream_idx += 1
    time.sleep(2.5)
    st.rerun()

# -------------------------------------------------------
# 수동 모드 (드롭다운 + 버튼)
# -------------------------------------------------------
if not st.session_state.streaming:
    with st.expander("📂 녹화 세션 선택 (수동 단발 추론)", expanded=True):
        sel_label = st.selectbox(
            "세션", ["— 세션을 선택하세요 —"] + filt_labels,
            label_visibility="collapsed")
        run_btn = st.button("🚀 3센서 앙상블 추론 실행", use_container_width=True,
                            disabled=(sel_label == "— 세션을 선택하세요 —"))

    if run_btn and sel_label != "— 세션을 선택하세요 —":
        sidx    = filt_labels.index(sel_label)
        cur_key = filt_keys[sidx]
        meta    = catalog[cur_key]
        with st.spinner("MiniRocket 3센서 앙상블 추론 중..."):
            try:
                res = run_ensemble(meta["sensors"])
                st.session_state["ensemble_result"] = res
                st.session_state["ensemble_key"]    = cur_key
                st.session_state["sensor_csvs"]     = meta["sensors"]

                a_score = anomaly_score(res["model_probs"])
                sev     = severity(a_score)
                pri     = max(res["model_probs"], key=res["model_probs"].get)
                vid     = _VID_PREFIX.get(meta["vehicle"], "MOB-?") + f"{sidx:03d}"
                now_str = time.strftime("%H:%M:%S")
                st.session_state.score_history.append({
                    "ts": now_str, "score": a_score, "vehicle": meta["vehicle"],
                    "fault_class": meta["fault_class"], "primary": pri,
                })
                if len(st.session_state.score_history) > MAX_HISTORY:
                    st.session_state.score_history.pop(0)
                if sev in ("경고", "주의"):
                    st.session_state.alert_log.insert(0, {
                        "ts": now_str, "severity": sev, "vid": vid,
                        "vehicle": meta["vehicle"],
                        "msg": f"[수동 조회] {meta['fault_class']}: 이상점수 {a_score}",
                        "primary": pri, "conf": res["model_probs"][pri] * 100,
                    })
            except Exception as e:
                st.error(f"🚨 추론 실패: {e}")
                st.stop()

# -------------------------------------------------------
# Fail-Safe: 결과 없으면 대기
# -------------------------------------------------------
_result = st.session_state.get("ensemble_result")
if _result is None:
    st.info("📂 차종을 선택하고 스트리밍을 시작하거나, 수동으로 세션을 선택·추론하세요.")
    st.stop()

model_probs   = _result["model_probs"]
per_sensor    = _result["per_sensor"]
primary_label = max(model_probs, key=model_probs.get)
primary_conf  = model_probs[primary_label]
a_score       = anomaly_score(model_probs)
sev_now       = severity(a_score)

penalty   = (model_probs["ECC10"]*30 + model_probs["ECC20"]*60 +
             model_probs["DEMAG"]*60 + model_probs["REDUC"]*60)
final_soh = int(np.clip(100 - penalty, 0, 100))
status_str   = "정상" if final_soh >= 85 else ("주의" if final_soh >= 60 else "결함")
status_color = "#10B981" if status_str == "정상" else ("#F59E0B" if status_str == "주의" else "#EF4444")

# -------------------------------------------------------
# 이상 점수 트렌드 차트
# -------------------------------------------------------
history = st.session_state.score_history
if history:
    st.markdown(
        '<div style="background:#1E293B; border:1px solid #334155; '
        'border-radius:12px; padding:16px 20px; margin-bottom:16px;">',
        unsafe_allow_html=True)

    exceed = a_score >= ANOMALY_WARN
    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown('<span style="font-size:1rem; font-weight:600; color:#E2E8F0;">이상 점수 트렌드</span>',
                    unsafe_allow_html=True)
    with top_r:
        if exceed:
            st.markdown(f'<span style="color:#EF4444; font-weight:700; font-size:0.95rem;">'
                        f'임계값 초과!  {a_score}</span>', unsafe_allow_html=True)

    scores = [h["score"] for h in history]
    labels = [h["ts"]    for h in history]

    fig_t, ax_t = plt.subplots(figsize=(12, 2.8))
    fig_t.patch.set_facecolor("#1E293B")
    ax_t.set_facecolor("#1E293B")

    ax_t.plot(range(len(scores)), scores, color="#38BDF8", linewidth=2.0,
              marker="o", markersize=3.5, markerfacecolor="#38BDF8", zorder=3)
    ax_t.fill_between(range(len(scores)), scores, alpha=0.12, color="#38BDF8")

    # 임계선
    ax_t.axhline(ANOMALY_WARN,    color="#EF4444", linestyle="--", linewidth=1.2,
                 alpha=0.8, label=f"임계값 {ANOMALY_WARN}", zorder=2)
    ax_t.axhline(ANOMALY_CAUTION, color="#F59E0B", linestyle=":",  linewidth=1.0,
                 alpha=0.6, label=f"주의선 {ANOMALY_CAUTION}", zorder=2)

    ax_t.set_xlim(0, max(len(scores) - 1, 1))
    ax_t.set_ylim(0, 105)
    ax_t.set_xticks(range(len(labels)))
    ax_t.set_xticklabels(labels, color="#64748B", fontsize=7.5, rotation=30, ha="right")
    ax_t.tick_params(axis="y", colors="#64748B", labelsize=8)
    ax_t.set_ylabel("이상 점수", color="#94A3B8", fontsize=8.5)

    for spine in ["top", "right"]:
        ax_t.spines[spine].set_visible(False)
    ax_t.spines["left"].set_color("#334155")
    ax_t.spines["bottom"].set_color("#334155")
    ax_t.grid(axis="y", color="#334155", linestyle=":", alpha=0.5)

    leg = ax_t.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155",
                      labelcolor="#94A3B8", fontsize=8, framealpha=0.9)

    plt.tight_layout(pad=0.5)
    st.pyplot(fig_t)
    plt.close(fig_t)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# 4구 KPI 카드
# -------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-title">Motor Health Score</div>
      <div class="kpi-value">{final_soh}
        <span style="font-size:0.95rem; color:#94A3B8;">/ 100 점</span></div>
      <div class="kpi-sub" style="color:{status_color};">MiniRocket 앙상블 기반</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-title">Primary Diagnosis</div>
      <div class="kpi-value">{primary_label}</div>
      <div class="kpi-sub">신뢰도: {primary_conf*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with k3:
    sev_color = "#EF4444" if sev_now=="경고" else ("#F59E0B" if sev_now=="주의" else "#10B981")
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-title">이상 점수 (Anomaly Score)</div>
      <div class="kpi-value" style="color:{sev_color};">{a_score}</div>
      <div class="kpi-sub" style="color:{sev_color};">{sev_now} — 임계값 {ANOMALY_WARN}</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-title">System Latency (PSC-04)</div>
      <div class="kpi-value" style="color:#A855F7;">{_result['total_latency']}
        <span style="font-size:0.95rem; color:#94A3B8;">ms</span></div>
      <div class="kpi-sub">3센서 InferenceRouter 누적</div>
    </div>""", unsafe_allow_html=True)

# -------------------------------------------------------
# 중단 분석 영역 (좌: 기여도+파형 / 우: 확률 바)
# -------------------------------------------------------
st.write(" ")
left_col, right_col = st.columns([1.3, 1.0])
sensor_names = ["Current_U", "Vib_Motor", "Vib_TM"]

# ── 우측: 다중 고장 확률 ─────────────────────────────────────
with right_col:
    st.markdown('<div style="background:#1E293B; padding:20px; border-radius:12px; '
                'border:1px solid #334155; min-height:480px;">', unsafe_allow_html=True)
    st.write("📊 **다중 고장 유형 확률 실시간 모니터링**")
    st.caption("항목을 선택하면 좌측 센서 기여도 차트가 해당 클래스로 스위칭됩니다.")
    selected_class = st.radio("타겟 고장 유형:", CLASS_NAMES,
                              index=CLASS_NAMES.index(primary_label),
                              label_visibility="collapsed")
    st.write(" ")
    for name in CLASS_NAMES:
        p_val = model_probs[name]
        p_pct = f"{p_val*100:.1f}%"
        clr   = COLOR_MAP[name]
        sel   = name == selected_class
        border = "border:1px solid #38BDF8; background:rgba(56,189,248,0.07);" if sel else "border:1px solid transparent;"
        fw     = "600" if sel else "400"
        lclr   = "#F8FAFC" if sel else "#CBD5E1"
        st.markdown(f"""<div style="padding:8px 12px; border-radius:8px; margin-bottom:4px; {border}">
          <div class="progress-label">
            <span style="color:{lclr}; font-weight:{fw};">{name}</span>
            <span style="color:{clr}; font-weight:600;">{p_pct}</span>
          </div>
          <div class="progress-bg">
            <div class="progress-fill" style="width:{p_pct}; background-color:{clr};"></div>
          </div></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── 좌측: 센서 기여도 + 파형 ─────────────────────────────────
with left_col:
    st.markdown(f'<div style="background:#1E293B; padding:20px; border-radius:12px; '
                f'border:1px solid #334155; min-height:480px;">', unsafe_allow_html=True)
    st.write(f"🧬 **3센서 모델 기여도 분석 : [ {selected_class} ]**")

    cls_idx      = _ML_CLASSES.index(selected_class)
    baseline     = 1.0 / 5
    contributions = [float(per_sensor[s]["probs"][cls_idx]) - baseline for s in sensor_names]
    sensor_probs  = [float(per_sensor[s]["probs"][cls_idx]) for s in sensor_names]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    fig.patch.set_facecolor("#1E293B"); ax.set_facecolor("#1E293B")
    cumulative = baseline
    for i, (sensor, contrib, rp) in enumerate(zip(sensor_names, contributions, sensor_probs)):
        color = "#EF4444" if contrib >= 0 else "#10B981"
        ax.barh(i, contrib, left=cumulative, color=color, height=0.5, edgecolor="none", zorder=3)
        sign   = "+" if contrib >= 0 else "-"
        ha     = "left" if contrib >= 0 else "right"
        offset = 0.006 if contrib >= 0 else -0.006
        ax.text(cumulative+contrib+offset, i, f"{sign}{abs(contrib):.3f}  (p={rp:.2f})",
                va="center", ha=ha, color="#CBD5E1", fontsize=9)
        cumulative += contrib
    ax.axvline(baseline,   color="#64748B", linestyle="--", linewidth=1.2, alpha=0.8, zorder=2)
    ax.axvline(cumulative, color="#38BDF8", linestyle="-",  linewidth=1.5, alpha=0.6, zorder=2)
    ax.text(baseline,   len(sensor_names)-0.05, f" E[f]={baseline:.2f}", color="#64748B", fontsize=7.5, va="top")
    ax.text(cumulative, -0.55, f" f(x)={cumulative:.3f}", color="#38BDF8", fontsize=8.0, va="bottom")
    ax.set_yticks(range(len(sensor_names)))
    ax.set_yticklabels(sensor_names, color="#94A3B8", fontsize=10)
    ax.tick_params(axis="x", colors="#64748B", labelsize=8.5)
    ax.set_xlim(left=min(0, baseline-0.15))
    for spine in ["top","right"]: ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#334155"); ax.spines["bottom"].set_color("#334155")
    ax.grid(axis="x", color="#334155", linestyle=":", alpha=0.5, zorder=1)
    ax.legend(handles=[mpatches.Patch(color="#EF4444",label="확률 상승 (+)"),
                       mpatches.Patch(color="#10B981",label="확률 억제 (-)")],
              loc="lower right", facecolor="#0F172A", edgecolor="#334155",
              labelcolor="#94A3B8", fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # 파형 차트
    st.write(" ")
    target_sensor = CLASS_TO_SENSOR.get(selected_class, "Vib_Motor")
    target_csv    = st.session_state.get("sensor_csvs", {}).get(target_sensor)
    raw_signal    = load_rms_signal(target_csv) if target_csv else None
    is_real       = raw_signal is not None
    if not is_real:
        raw_signal = generate_fallback_signal(selected_class)
    src_tag  = "실측 rms" if is_real else "합성 Sine+Noise"
    src_name = Path(target_csv).name if (is_real and target_csv) else "폴백 파형"

    fig2, ax2 = plt.subplots(figsize=(6.5, 2.4))
    fig2.patch.set_facecolor("#1E293B"); ax2.set_facecolor("#1E293B")
    ax2.plot(raw_signal, color="#38BDF8", linewidth=0.7, alpha=0.9)
    ax2.set_title(f"[{src_tag}] {target_sensor}  —  {selected_class}  |  {src_name}",
                  color="#94A3B8", fontsize=8, pad=6, fontfamily="Apple SD Gothic Neo")
    ax2.set_xlabel("Time Index  (2,000 samples)", color="#94A3B8", fontsize=7.5,
                   fontfamily="Apple SD Gothic Neo")
    ax2.set_ylabel("Signal Amplitude", color="#94A3B8", fontsize=7.5,
                   fontfamily="Apple SD Gothic Neo")
    ax2.tick_params(axis="both", colors="#64748B", labelsize=7)
    ax2.set_xlim(0, len(raw_signal)-1)
    for spine in ["top","right"]: ax2.spines[spine].set_visible(False)
    ax2.spines["left"].set_color("#334155"); ax2.spines["bottom"].set_color("#334155")
    ax2.grid(color="#334155", linestyle=":", alpha=0.5)
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# 알림 로그 (누적 테이블)
# -------------------------------------------------------
st.write(" ")
alert_log = st.session_state.alert_log
warn_cnt  = sum(1 for a in alert_log if a["severity"] == "경고")
caut_cnt  = sum(1 for a in alert_log if a["severity"] == "주의")

st.markdown(
    f'<div style="margin-bottom:8px; font-size:1rem; font-weight:600; color:#E2E8F0;">'
    f'알림 로그 <span style="color:#94A3B8; font-size:0.8rem;">(실시간)</span>'
    f'&nbsp;&nbsp;<span style="background:rgba(239,68,68,0.2); color:#EF4444; '
    f'padding:2px 9px; border-radius:20px; font-size:0.78rem; font-weight:700;">'
    f'경고 {warn_cnt}</span>&nbsp;'
    f'<span style="background:rgba(245,158,11,0.2); color:#F59E0B; '
    f'padding:2px 9px; border-radius:20px; font-size:0.78rem; font-weight:700;">'
    f'주의 {caut_cnt}</span></div>',
    unsafe_allow_html=True,
)

if not alert_log:
    st.markdown('<div style="background:#1E293B; border:1px solid #334155; '
                'border-radius:8px; padding:20px; color:#475569; text-align:center;">'
                '이상 이벤트 없음</div>', unsafe_allow_html=True)
else:
    header = ('<div style="background:#0F172A; border:1px solid #334155; '
              'border-radius:8px 8px 0 0; overflow:hidden;">'
              '<div class="log-row" style="color:#64748B; font-weight:600; '
              'border-bottom:1px solid #334155; background:#0F172A;">'
              '<span>시간</span><span>등급</span><span>차량 ID</span><span>메시지</span></div>')
    rows = []
    for entry in alert_log[:20]:
        badge_cls = "badge-warn" if entry["severity"] == "경고" else "badge-caution"
        conf_str  = f"{entry['conf']:.1f}%"
        row = (f'<div class="log-row">'
               f'<span style="color:#64748B;">{entry["ts"]}</span>'
               f'<span><span class="badge {badge_cls}">{entry["severity"]}</span></span>'
               f'<span style="color:#38BDF8; font-weight:600;">{entry["vid"]}</span>'
               f'<span style="color:#CBD5E1;">{entry["msg"]} '
               f'<span style="color:{COLOR_MAP.get(entry["primary"],"#94A3B8")}; '
               f'font-weight:600;">[{entry["primary"]} {conf_str}]</span></span>'
               f'</div>')
        rows.append(row)
    st.markdown(header + "".join(rows) + "</div>", unsafe_allow_html=True)
