"""
MT Provider Selection Engine — Light Enterprise UI
Streamlit front-end  |  Backend: FastAPI on localhost:8000

Supports:
  • Multiple CSV uploads
  • Multiple target languages
  • Multiple MT models
  • Explicit Latency & Cost priority dropdowns (Low / Good / High)
Each (CSV × language) pair is evaluated as its own job; results are aggregated.
"""
import json, time, requests, pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys, os, threading
import uvicorn as _uvicorn  # import BEFORE sys.path change to avoid shadowing

# ── Start FastAPI backend in a background thread (single-server mode) ─────────
# Guard: only start once across all Streamlit reruns (check if port is already bound)
_BACKEND_DIR = os.path.join(os.path.dirname(__file__), "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

def _port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

if not _port_in_use(8000):
    def _start_backend():
        _uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="warning")
    _bt = threading.Thread(target=_start_backend, daemon=True, name="fastapi-backend")
    _bt.start()
    time.sleep(1.5)   # give uvicorn a moment to bind the port
# ─────────────────────────────────────────────────────────────────────────────

BACKEND = "http://localhost:8000"

st.set_page_config(
    page_title="Decision Engine — MT Provider Selection",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Premium Enterprise AI Platform Design System
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
.stApp { background: #F8FAFC; color: #0F172A; font-family: 'Inter', sans-serif; }
.block-container { padding: 0 2.5rem 4rem; max-width: 1340px; margin: 0 auto; }
section[data-testid="stSidebar"] { display: none !important; }
button[data-testid="collapsedControl"] { display: none !important; }
footer { visibility: hidden; } #MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { height: 0; min-height: 0; background: transparent; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }

/* ══════════════════════════════════════════════
   NAVBAR
══════════════════════════════════════════════ */
.navbar {
  display: flex; align-items: center; justify-content: space-between;
  background: rgba(255,255,255,0.92);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(226,232,240,0.8);
  padding: 12px 2.5rem; margin: 0 -2.5rem 0;
  position: sticky; top: 0; z-index: 200;
  box-shadow: 0 1px 3px rgba(15,23,42,0.06);
}
.nb-brand {
  display: flex; align-items: center; gap: 10px;
  font-size: 17px; font-weight: 800; color: #0F172A; letter-spacing: -0.3px;
}
.nb-logo {
  width: 32px; height: 32px; border-radius: 8px;
  background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 16px; box-shadow: 0 2px 8px rgba(79,70,229,0.35);
}
.nb-brand .brand-primary { color: #4F46E5; }
.nb-right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.nb-badge {
  display: inline-flex; align-items: center; gap: 5px;
  background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 999px;
  padding: 4px 12px; font-size: 11px; color: #475569; font-weight: 500;
}
.nb-badge.online { border-color: #BBF7D0; background: #F0FDF4; color: #16A34A; }
.nb-badge.offline { border-color: #FECACA; background: #FEF2F2; color: #DC2626; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dot-green { background: #10B981; box-shadow: 0 0 0 2px rgba(16,185,129,0.25); }
.dot-red   { background: #EF4444; box-shadow: 0 0 0 2px rgba(239,68,68,0.25); }
.dot-blue  { background: #4F46E5; }
.dot-amber { background: #F59E0B; }
.nb-tagline {
  font-size: 10px; color: #94A3B8; font-weight: 500;
  background: linear-gradient(90deg, #4F46E5, #7C3AED);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  font-weight: 700; letter-spacing: 0.02em;
}

/* ══════════════════════════════════════════════
   HERO / LANDING
══════════════════════════════════════════════ */
.hero-landing {
  text-align: center; padding: 36px 0 20px;
  background: linear-gradient(180deg, rgba(249,250,251,0) 0%, rgba(249,250,251,0) 100%);
}
.hero-eyebrow {
  display: inline-flex; align-items: center; gap: 6px;
  background: linear-gradient(135deg, rgba(79,70,229,0.08) 0%, rgba(124,58,237,0.08) 100%);
  border: 1px solid rgba(79,70,229,0.2); border-radius: 999px;
  padding: 4px 14px; font-size: 11px; font-weight: 700; color: #4F46E5;
  letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 16px;
}
.hero-title {
  font-size: 40px; font-weight: 900; line-height: 1.15;
  color: #0F172A; margin: 0 0 10px; letter-spacing: -1px;
}
.hero-title .grad {
  background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #06B6D4 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-tagline {
  font-size: 16px; color: #0F172A; font-weight: 700;
  margin: 0 auto 10px; max-width: 680px;
}
.hero-sub {
  font-size: 14px; color: #64748B; max-width: 600px; margin: 0 auto 20px;
  line-height: 1.65; font-weight: 400;
}

/* Hero metric cards */
.hero-metrics { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 20px 0 28px; }
.hm-card {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
  padding: 20px 16px; text-align: center;
  box-shadow: 0 1px 4px rgba(15,23,42,0.06), 0 4px 16px rgba(79,70,229,0.04);
  transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
  position: relative; overflow: hidden;
}
.hm-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #4F46E5, #7C3AED);
}
.hm-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(79,70,229,0.12); border-color: #C7D2FE; }
.hm-icon { font-size: 28px; margin-bottom: 8px; }
.hm-val { font-size: 34px; font-weight: 900; line-height: 1; background: linear-gradient(135deg,#4F46E5,#7C3AED); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hm-label { font-size: 11px; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 6px; }

/* ══════════════════════════════════════════════
   NAVIGATION TABS (premium pill nav)
══════════════════════════════════════════════ */
.nav-tabs-wrap {
  display: flex; gap: 4px; background: #FFFFFF; border: 1px solid #E2E8F0;
  border-radius: 14px; padding: 5px; margin-bottom: 28px;
  box-shadow: 0 1px 4px rgba(15,23,42,0.06);
}
.nav-tab {
  flex: 1; text-align: center; padding: 10px 16px; border-radius: 10px;
  font-size: 13px; font-weight: 600; color: #64748B; cursor: pointer;
  transition: all 0.2s; border: none; background: transparent;
}
.nav-tab.active {
  background: linear-gradient(135deg,#4F46E5,#7C3AED); color: #FFFFFF;
  box-shadow: 0 2px 8px rgba(79,70,229,0.35);
}
.nav-tab:hover:not(.active) { background: #F1F5F9; color: #0F172A; }

/* ══════════════════════════════════════════════
   SECTION CARDS & STEPS
══════════════════════════════════════════════ */
.sec-card {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 18px;
  padding: 24px 26px; margin-bottom: 18px;
  box-shadow: 0 1px 4px rgba(15,23,42,0.05), 0 4px 12px rgba(15,23,42,0.03);
}
.step-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 28px; height: 28px; border-radius: 8px; flex-shrink: 0;
  background: linear-gradient(135deg,#4F46E5,#7C3AED); color: #fff;
  font-size: 12px; font-weight: 800;
}
.step-hdr {
  display: flex; align-items: center; gap: 10px;
  font-size: 15px; font-weight: 800; color: #0F172A;
  margin: 10px 0 14px; letter-spacing: -0.2px;
}
.step-hdr span { color: #4F46E5; font-size: 18px; }
.help-note { font-size: 12px; color: #94A3B8; margin: -6px 0 12px; line-height: 1.5; }

/* ══════════════════════════════════════════════
   FILE UPLOAD ZONE
══════════════════════════════════════════════ */
.upload-zone {
  border: 2px dashed #C7D2FE; border-radius: 16px;
  background: linear-gradient(135deg, rgba(79,70,229,0.03) 0%, rgba(124,58,237,0.03) 100%);
  padding: 30px; text-align: center; transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #4F46E5; }
.file-ok {
  background: linear-gradient(135deg,#F0FDF4,#ECFDF5); border: 1px solid #86EFAC;
  border-radius: 12px; padding: 14px 18px; margin-bottom: 8px;
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  box-shadow: 0 1px 3px rgba(16,185,129,0.1);
}
.fo-name { font-size: 13px; font-weight: 700; color: #15803D; flex: 1; }
.fo-chip {
  background: #FFFFFF; border: 1px solid #86EFAC; border-radius: 6px;
  padding: 2px 9px; font-size: 11px; color: #15803D; font-weight: 600;
}

/* ══════════════════════════════════════════════
   MODEL CARDS — Premium AI Cards
══════════════════════════════════════════════ */
.mc {
  background: #FFFFFF; border: 2px solid #E2E8F0; border-radius: 16px;
  padding: 18px 14px; text-align: center; margin-bottom: 6px;
  transition: all 0.2s; cursor: pointer; position: relative; overflow: hidden;
  box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}
.mc::after {
  content: ''; position: absolute; inset: 0; border-radius: 14px;
  background: linear-gradient(135deg,rgba(79,70,229,0),rgba(124,58,237,0));
  transition: background 0.2s;
}
.mc:hover { border-color: #A5B4FC; box-shadow: 0 4px 16px rgba(79,70,229,0.12); transform: translateY(-2px); }
.mc.sel {
  border-color: #4F46E5 !important;
  background: linear-gradient(135deg,rgba(79,70,229,0.06) 0%,rgba(124,58,237,0.04) 100%) !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,0.12), 0 4px 16px rgba(79,70,229,0.15) !important;
}
.mc-icon { font-size: 28px; margin-bottom: 8px; }
.mc-name { font-size: 13px; font-weight: 800; color: #0F172A; margin-bottom: 6px; letter-spacing: -0.2px; }
.mc-tag {
  display: inline-flex; align-items: center; gap: 4px;
  border-radius: 6px; padding: 2px 8px; font-size: 10px; font-weight: 700; margin-bottom: 8px;
}
.tag-live { background: #DCFCE7; color: #15803D; }
.tag-sim  { background: #F1F5F9; color: #64748B; }
.mc-row { font-size: 10px; color: #94A3B8; line-height: 1.8; }
.mc-hl  { color: #475569; font-weight: 700; }
.mc-sel-check {
  position: absolute; top: 8px; right: 8px; width: 18px; height: 18px;
  border-radius: 50%; background: #4F46E5; display: flex; align-items: center;
  justify-content: center; font-size: 10px; color: #fff; font-weight: 800;
}

/* ══════════════════════════════════════════════
   WINNER / RECOMMENDATION CARDS
══════════════════════════════════════════════ */
.winner-wrap {
  background: linear-gradient(135deg,#EEF2FF 0%,#F0FDF4 60%,#ECFDF5 100%);
  border: 2px solid #A5B4FC; border-radius: 22px; padding: 32px 36px;
  margin-bottom: 22px; position: relative; overflow: hidden;
  box-shadow: 0 4px 24px rgba(79,70,229,0.1);
}
.winner-wrap::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
  background: linear-gradient(90deg, #4F46E5, #7C3AED, #06B6D4);
}
.w-label {
  font-size: 10px; font-weight: 800; color: #4F46E5;
  text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px;
  display: flex; align-items: center; gap: 6px;
}
.w-name {
  font-size: 36px; font-weight: 900; color: #0F172A; margin: 0 0 6px;
  letter-spacing: -1px; line-height: 1.1;
}
.w-sub { font-size: 13px; color: #64748B; margin-bottom: 4px; }
.w-score {
  position: absolute; top: 28px; right: 36px;
  background: #FFFFFF; border: 2px solid #A5B4FC;
  border-radius: 16px; padding: 14px 22px; text-align: center;
  box-shadow: 0 4px 16px rgba(79,70,229,0.15);
}
.ws-val {
  font-size: 42px; font-weight: 900; line-height: 1;
  background: linear-gradient(135deg, #4F46E5, #7C3AED);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.ws-label { font-size: 10px; color: #64748B; text-transform: uppercase; margin-top: 4px; font-weight: 600; letter-spacing: 0.08em; }
.w-kpis { display: flex; gap: 10px; margin-top: 22px; flex-wrap: wrap; }
.wkpi {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
  padding: 12px 16px; text-align: center; min-width: 90px;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04);
  transition: transform 0.2s;
}
.wkpi:hover { transform: translateY(-1px); }
.wkv { font-size: 22px; font-weight: 900; line-height: 1; }
.wkl { font-size: 10px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 4px; font-weight: 600; }
.c-green { color: #10B981; } .c-amber { color: #F59E0B; }
.c-red   { color: #EF4444; } .c-blue  { color: #4F46E5; }
.c-indigo { color: #4F46E5; }
.w-pills { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin-top: 16px; }
.w-pill-l { font-size: 11px; color: #94A3B8; font-weight: 500; }
.w-pill {
  background: rgba(79,70,229,0.08); color: #4F46E5;
  border: 1px solid rgba(79,70,229,0.2); border-radius: 999px;
  padding: 3px 12px; font-size: 11px; font-weight: 700;
}

/* ══════════════════════════════════════════════
   PIPELINE / PROGRESS
══════════════════════════════════════════════ */
.pipeline {
  display: flex; align-items: center; justify-content: center;
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 18px;
  padding: 22px 20px; flex-wrap: nowrap; overflow-x: auto;
  box-shadow: 0 1px 4px rgba(15,23,42,0.05); margin-bottom: 28px;
}
.pipe-title { font-size: 10px; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.14em; text-align: center; margin-bottom: 14px; }
.pipe-step {
  display: flex; flex-direction: column; align-items: center; gap: 5px;
  background: #FFFFFF; border: 2px solid #E2E8F0; border-radius: 14px;
  padding: 14px 18px; min-width: 135px; text-align: center; flex-shrink: 0; transition: all 0.3s;
}
.pipe-step.active { border-color: #4F46E5; background: #EEF2FF; box-shadow: 0 0 0 4px rgba(79,70,229,0.1); }
.pipe-step.done   { border-color: #10B981; background: #F0FDF4; }
.pipe-step.active .ps-name { color: #4F46E5; }
.pipe-step.done   .ps-name { color: #10B981; }
.ps-icon  { font-size: 22px; }
.ps-num   { font-size: 9px; color: #4F46E5; font-weight: 800; letter-spacing: 0.08em; }
.ps-name  { font-size: 12px; font-weight: 800; color: #334155; }
.ps-desc  { font-size: 10px; color: #94A3B8; }
.pipe-arrow { color: #CBD5E1; font-size: 18px; padding: 0 8px; flex-shrink: 0; }

/* ══════════════════════════════════════════════
   AGENT PROCESSING ROWS
══════════════════════════════════════════════ */
.proc-hdr { text-align: center; padding: 32px 0 22px; }
.proc-hdr h2 {
  font-size: 28px; font-weight: 900; color: #0F172A; margin: 0 0 8px; letter-spacing: -0.5px;
}
.proc-hdr p { font-size: 14px; color: #94A3B8; }
.agent-row {
  display: flex; align-items: center; gap: 14px; padding: 14px 20px;
  border-radius: 14px; margin-bottom: 8px; border: 1px solid #E2E8F0;
  background: #FFFFFF; transition: all 0.3s;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
.agent-row.done    { border-color: #86EFAC; background: #F0FDF4; }
.agent-row.running { border-color: #A5B4FC; background: #EEF2FF; animation: agentPulse 1.8s ease-in-out infinite; }
.agent-row.pending { opacity: 0.45; }
@keyframes agentPulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(79,70,229,0.2); }
  50%      { box-shadow: 0 0 0 6px rgba(79,70,229,0); }
}
.ar-icon   { font-size: 22px; width: 34px; text-align: center; flex-shrink: 0; }
.ar-body   { flex: 1; }
.ar-name   { font-size: 14px; font-weight: 700; color: #0F172A; }
.ar-desc   { font-size: 11px; color: #94A3B8; margin-top: 1px; }
.ar-status { font-size: 11px; padding: 3px 12px; border-radius: 999px; flex-shrink: 0; font-weight: 700; }
.ars-done  { background: #DCFCE7; color: #15803D; }
.ars-run   { background: #EEF2FF; color: #4F46E5; }
.ars-pend  { background: #F8FAFC; color: #94A3B8; border: 1px solid #E2E8F0; }

/* ══════════════════════════════════════════════
   EXPLAINABILITY CARDS
══════════════════════════════════════════════ */
.ex-card {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
  padding: 22px; margin-bottom: 12px;
  box-shadow: 0 1px 4px rgba(15,23,42,0.05);
}
.ex-card.winner-ex { border-color: #A5B4FC; background: #EEF2FF; }
.ex-card.reject-ex { border-color: #FCA5A5; background: #FEF2F2; }
.ex-head { display: flex; align-items: center; gap: 14px; margin-bottom: 12px; }
.ex-icon { font-size: 30px; }
.ex-title { font-size: 16px; font-weight: 800; color: #0F172A; letter-spacing: -0.2px; }
.ex-sub { font-size: 12px; color: #94A3B8; margin-top: 2px; }
.ex-body { font-size: 14px; color: #334155; line-height: 1.7; }
.ex-bullets { margin-top: 12px; padding: 0; list-style: none; }
.ex-bullets li {
  position: relative; padding-left: 20px; font-size: 13px; color: #475569;
  margin-bottom: 6px; line-height: 1.5;
}
.ex-bullets li::before { content: "▸"; position: absolute; left: 0; color: #4F46E5; font-size: 11px; top: 2px; }

/* ══════════════════════════════════════════════
   DECISION TRACE
══════════════════════════════════════════════ */
.trace-item { display: flex; gap: 16px; padding: 16px 0; border-bottom: 1px solid #F1F5F9; }
.trace-item:last-child { border-bottom: none; }
.ti-left { display: flex; flex-direction: column; align-items: center; width: 36px; flex-shrink: 0; }
.ti-dot  { width: 13px; height: 13px; border-radius: 50%; background: #10B981; flex-shrink: 0; margin-top: 3px; box-shadow: 0 0 0 3px rgba(16,185,129,0.15); }
.ti-line { width: 2px; flex: 1; background: linear-gradient(180deg,#10B981,#E2E8F0); min-height: 20px; }
.ti-right  { flex: 1; }
.ti-agent  { font-size: 14px; font-weight: 800; color: #0F172A; }
.ti-fact   { font-size: 12px; color: #10B981; font-weight: 700; margin-top: 3px; }
.ti-detail { font-size: 12px; color: #64748B; margin-top: 5px; line-height: 1.6; }

/* ══════════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════════ */
div[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
  border: none !important; border-radius: 12px !important;
  padding: 14px 28px !important; font-size: 15px !important; font-weight: 800 !important;
  box-shadow: 0 4px 16px rgba(79,70,229,0.35) !important;
  transition: transform 0.15s, box-shadow 0.15s !important;
  letter-spacing: -0.1px !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 28px rgba(79,70,229,0.45) !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
  background: #FFFFFF !important; border: 1.5px solid #E2E8F0 !important;
  border-radius: 10px !important; color: #475569 !important;
  font-weight: 600 !important; font-size: 13px !important;
  transition: all 0.15s !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
  border-color: #4F46E5 !important; color: #4F46E5 !important;
  box-shadow: 0 2px 8px rgba(79,70,229,0.12) !important;
}

/* ══════════════════════════════════════════════
   TABS (Streamlit native)
══════════════════════════════════════════════ */
[data-baseweb="tab-list"] {
  background: #F8FAFC !important; border-radius: 12px !important;
  padding: 4px !important; gap: 2px !important; border: 1px solid #E2E8F0 !important;
}
[data-baseweb="tab"] {
  border-radius: 9px !important; color: #64748B !important;
  font-weight: 600 !important; font-size: 13px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  background: linear-gradient(135deg, #4F46E5, #7C3AED) !important; color: #fff !important;
  box-shadow: 0 2px 8px rgba(79,70,229,0.3) !important;
}

/* ══════════════════════════════════════════════
   METRICS & WIDGETS
══════════════════════════════════════════════ */
[data-testid="stMetric"] {
  background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
  border-radius: 14px !important; padding: 16px !important;
  box-shadow: 0 1px 4px rgba(15,23,42,0.05) !important;
}
[data-testid="stMetricLabel"]  { color: #64748B !important; font-size: 11px !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; }
[data-testid="stMetricValue"]  { color: #0F172A !important; font-size: 22px !important; font-weight: 800 !important; }
[data-testid="stProgress"] > div { background: #E2E8F0 !important; border-radius: 999px !important; }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #4F46E5, #7C3AED) !important; }
hr { border-color: #E2E8F0 !important; opacity: 0.6 !important; }

/* ══════════════════════════════════════════════
   DIVIDERS & SECTION SPACING
══════════════════════════════════════════════ */
.sec-divider {
  height: 1px; background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
  margin: 24px 0; border: none;
}
.scard-title {
  font-size: 16px; font-weight: 800; color: #0F172A;
  margin: 0 0 18px; display: flex; align-items: center; gap: 8px;
  letter-spacing: -0.2px;
}
.scard-title span { color: #4F46E5; }

/* ══════════════════════════════════════════════
   HERO section pages
══════════════════════════════════════════════ */
.hero { text-align: center; padding: 10px 0 16px; }
.hero-tag {
  display: inline-flex; align-items: center; gap: 6px;
  background: linear-gradient(135deg, rgba(79,70,229,0.08), rgba(124,58,237,0.08));
  border: 1px solid rgba(79,70,229,0.2); border-radius: 999px;
  padding: 4px 16px; font-size: 10px; font-weight: 800; color: #4F46E5;
  letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 14px;
}
.hero h1 { font-size: 28px; font-weight: 900; margin: 0 0 8px; color: #0F172A; letter-spacing: -0.5px; }
.hero h1 span {
  background: linear-gradient(135deg, #4F46E5, #7C3AED);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { font-size: 14px; color: #64748B; max-width: 580px; margin: 0 auto 8px; line-height: 1.6; }

/* ══════════════════════════════════════════════
   ONBOARDING WIZARD STEPS
══════════════════════════════════════════════ */
.wizard-progress {
  display: flex; align-items: center; gap: 0;
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
  padding: 16px 22px; margin-bottom: 24px;
  box-shadow: 0 1px 4px rgba(15,23,42,0.05);
}
.wz-step {
  display: flex; align-items: center; gap: 8px; flex: 1;
  padding: 6px 10px; border-radius: 10px;
}
.wz-step.active { background: linear-gradient(135deg,rgba(79,70,229,0.08),rgba(124,58,237,0.06)); }
.wz-num {
  width: 26px; height: 26px; border-radius: 8px; flex-shrink: 0;
  background: #E2E8F0; color: #94A3B8;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 800;
}
.wz-step.active .wz-num { background: linear-gradient(135deg,#4F46E5,#7C3AED); color: #fff; }
.wz-step.done .wz-num { background: #10B981; color: #fff; }
.wz-label { font-size: 11px; font-weight: 700; color: #94A3B8; }
.wz-step.active .wz-label { color: #4F46E5; }
.wz-step.done .wz-label { color: #10B981; }
.wz-arrow { color: #E2E8F0; font-size: 16px; flex-shrink: 0; margin: 0 -2px; }

/* ══════════════════════════════════════════════
   FOOTER
══════════════════════════════════════════════ */
.site-footer {
  margin-top: 48px; padding: 28px 0 20px;
  border-top: 1px solid #E2E8F0; text-align: center;
}
.footer-brand {
  font-size: 16px; font-weight: 800; color: #0F172A; margin-bottom: 4px;
}
.footer-brand span { background: linear-gradient(135deg,#4F46E5,#7C3AED); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.footer-sub { font-size: 12px; color: #94A3B8; margin-bottom: 12px; }
.footer-badges { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.footer-badge {
  background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 999px;
  padding: 3px 12px; font-size: 10px; color: #64748B; font-weight: 600;
}

/* ══════════════════════════════════════════════
   DATAFRAME OVERRIDES
══════════════════════════════════════════════ */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }

/* ══════════════════════════════════════════════
   EMPTY STATE
══════════════════════════════════════════════ */
.empty-state {
  text-align: center; padding: 48px 24px;
  background: #FFFFFF; border: 2px dashed #E2E8F0; border-radius: 18px;
  margin: 16px 0;
}
.empty-state-icon { font-size: 52px; margin-bottom: 14px; }
.empty-state-title { font-size: 18px; font-weight: 800; color: #0F172A; margin-bottom: 8px; }
.empty-state-sub { font-size: 13px; color: #94A3B8; max-width: 380px; margin: 0 auto; line-height: 1.6; }

/* ══════════════════════════════════════════════
   SCORING BADGE / KPI INLINE
══════════════════════════════════════════════ */
.score-excellent { color: #10B981; font-weight: 800; }
.score-good      { color: #F59E0B; font-weight: 800; }
.score-poor      { color: #EF4444; font-weight: 800; }
.inline-badge {
  display: inline-flex; align-items: center; gap: 4px;
  border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 700;
}
.badge-green  { background: #DCFCE7; color: #15803D; }
.badge-amber  { background: #FEF3C7; color: #B45309; }
.badge-red    { background: #FEE2E2; color: #DC2626; }
.badge-indigo { background: #EEF2FF; color: #4F46E5; }
.badge-cyan   { background: #ECFEFF; color: #0891B2; }

/* ══════════════════════════════════════════════
   ENTERPRISE ENHANCEMENTS
══════════════════════════════════════════════ */
.block-container { animation: pageFadeIn 0.35s ease-out; }
@keyframes pageFadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

/* Navbar animated gradient accent */
.navbar { position: relative; }
.navbar::after {
  content:''; display:block; position:absolute; bottom:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg,#4F46E5,#7C3AED,#06B6D4,#4F46E5);
  background-size:300% 100%; animation: navGrad 8s linear infinite;
}
@keyframes navGrad { 0%{background-position:0% 50%;} 100%{background-position:300% 50%;} }

/* Metric card staggered entry */
.hm-card:nth-child(1){animation:cardRise .45s .05s ease-out both;}
.hm-card:nth-child(2){animation:cardRise .45s .10s ease-out both;}
.hm-card:nth-child(3){animation:cardRise .45s .15s ease-out both;}
@keyframes cardRise { from{opacity:0;transform:translateY(16px);} to{opacity:1;transform:translateY(0);} }

/* Live API dot pulse */
.live-dot {
  display:inline-block; width:8px; height:8px; border-radius:50%;
  background:#10B981; vertical-align:middle; margin-right:4px;
  animation:liveDot 1.6s ease-in-out infinite;
  box-shadow:0 0 0 0 rgba(16,185,129,.4);
}
@keyframes liveDot {
  0%{box-shadow:0 0 0 0 rgba(16,185,129,.4);}
  70%{box-shadow:0 0 0 7px rgba(16,185,129,0);}
  100%{box-shadow:0 0 0 0 rgba(16,185,129,0);}
}

/* Decision Engine dark winner card */
.de-winner {
  background: linear-gradient(135deg,#0F172A 0%,#1E1B4B 60%,#0F172A 100%);
  border:1px solid rgba(99,102,241,.35); border-radius:24px;
  padding:36px 40px; color:#FFFFFF; position:relative; overflow:hidden;
  box-shadow:0 24px 64px rgba(15,23,42,.35), 0 0 80px rgba(79,70,229,.08);
}
.de-winner::before {
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg,#4F46E5,#7C3AED,#06B6D4);
}
.de-winner::after {
  content:''; position:absolute; top:-80px; right:-80px; width:260px; height:260px;
  border-radius:50%; background:radial-gradient(circle,rgba(79,70,229,.12) 0%,transparent 70%);
  pointer-events:none;
}
.de-label { font-size:10px; font-weight:800; color:#818CF8; text-transform:uppercase; letter-spacing:.14em; margin-bottom:8px; }
.de-name  { font-size:36px; font-weight:900; color:#FFFFFF; letter-spacing:-1.5px; line-height:1.1; margin-bottom:6px; }
.de-provider { font-size:13px; color:#94A3B8; margin-bottom:14px; }
.de-score-box {
  position:absolute; top:32px; right:40px;
  background:rgba(255,255,255,.07); border:1.5px solid rgba(99,102,241,.4);
  border-radius:16px; padding:16px 24px; text-align:center; backdrop-filter:blur(8px);
}
.de-score-val {
  font-size:42px; font-weight:900; line-height:1;
  background:linear-gradient(135deg,#818CF8,#C4B5FD);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.de-score-lbl { font-size:10px; color:#94A3B8; text-transform:uppercase; margin-top:4px; letter-spacing:.08em; }
.de-kpis { display:flex; gap:10px; margin-top:24px; flex-wrap:wrap; }
.de-kpi {
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
  border-radius:12px; padding:12px 16px; text-align:center; min-width:88px;
  backdrop-filter:blur(4px); transition:background .2s;
}
.de-kpi:hover { background:rgba(255,255,255,.1); }
.de-kpi-val { font-size:20px; font-weight:900; color:#FFFFFF; line-height:1; }
.de-kpi-lbl { font-size:10px; color:#64748B; text-transform:uppercase; margin-top:4px; letter-spacing:.07em; }
.de-pills { display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin-top:18px; }
.de-pill { background:rgba(99,102,241,.15); color:#A5B4FC; border:1px solid rgba(99,102,241,.3); border-radius:999px; padding:3px 12px; font-size:11px; font-weight:700; }
.de-badge{ background:rgba(16,185,129,.15); color:#6EE7B7; border:1px solid rgba(16,185,129,.3); border-radius:999px; padding:3px 12px; font-size:11px; font-weight:700; }
.de-gov-pass { background:rgba(16,185,129,.15); color:#6EE7B7; border:1px solid rgba(16,185,129,.3); border-radius:999px; padding:2px 10px; font-size:11px; font-weight:700; }
.de-gov-fail { background:rgba(239,68,68,.15); color:#FCA5A5; border:1px solid rgba(239,68,68,.3); border-radius:999px; padding:2px 10px; font-size:11px; font-weight:700; }

/* Enhanced multiselect pills */
[data-baseweb="tag"] { background:linear-gradient(135deg,#EEF2FF,#E0E7FF) !important; border:1px solid #C7D2FE !important; border-radius:8px !important; }
[data-baseweb="tag"] span { color:#4F46E5 !important; font-weight:700 !important; }

/* File uploader */
[data-testid="stFileUploadDropzone"] { background:linear-gradient(135deg,rgba(79,70,229,.02),rgba(124,58,237,.02)) !important; border:2px dashed #C7D2FE !important; border-radius:16px !important; }

/* Form card */
[data-testid="stForm"] { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:18px; padding:20px !important; box-shadow:0 1px 4px rgba(15,23,42,.05); }

/* Expander */
[data-testid="stExpander"] { border:1px solid #E2E8F0 !important; border-radius:14px !important; overflow:hidden !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DEFS = [
    {"id":"claude-sonnet-4",  "icon":"🟣","label":"Claude Sonnet 4",    "live":True, "quality":"Highest","cost":"High",  "speed":"Medium","default":True},
    {"id":"gemini-2.5-pro",   "icon":"🔵","label":"Gemini 2.5 Pro",     "live":True, "quality":"Highest","cost":"High",  "speed":"Medium","default":True},
    {"id":"deepseek-r1",      "icon":"🧠","label":"DeepSeek R1",        "live":True, "quality":"High",   "cost":"Low",   "speed":"Slow",  "default":False},
    {"id":"glm-4.5",          "icon":"🟢","label":"GLM 4.5",            "live":True, "quality":"High",   "cost":"Low",   "speed":"Medium","default":False},
    {"id":"kimi-k2",          "icon":"🌙","label":"Kimi K2",            "live":True, "quality":"High",   "cost":"Medium","speed":"Medium","default":False},
    {"id":"claude-haiku-4.5", "icon":"⚡","label":"Claude Haiku 4.5",   "live":True, "quality":"High",   "cost":"Medium","speed":"Fast",  "default":True},
    {"id":"qwen3-32b",        "icon":"🔶","label":"Qwen3 32B",          "live":True, "quality":"Good",   "cost":"Lowest","speed":"Fast",  "default":False},
    {"id":"gpt-4o",           "icon":"🤖","label":"GPT-4o",             "live":False,"quality":"Highest","cost":"High",  "speed":"Medium","default":False},
    {"id":"deepl",            "icon":"📚","label":"DeepL",              "live":False,"quality":"High",   "cost":"Medium","speed":"Fast",  "default":False},
    {"id":"azure-mt",         "icon":"☁️","label":"Azure MT",           "live":False,"quality":"Good",   "cost":"Medium","speed":"Fast",  "default":False},
]

LANG_OPTIONS = {
    "German (de)":"de","French (fr)":"fr","Spanish (es)":"es","Italian (it)":"it",
    "Portuguese (pt)":"pt","Dutch (nl)":"nl","Japanese (ja)":"ja","Chinese (zh)":"zh",
    "Korean (ko)":"ko","Arabic (ar)":"ar",
}

PRIORITY_OPTIONS = {"Low":"low", "Good":"good", "High":"high"}

COLORS = ["#2563EB","#0891B2","#16A34A","#D97706","#DC2626"]

# ── Extended model catalog: live first, then benchmark-evaluated ──────────────
ALL_MODELS_EXTENDED = [
    # ── Live API (7) ─────────────────────────────────────────────────────────
    {"id":"claude-sonnet-4",    "icon":"🟣","label":"Claude Sonnet 4",                   "live":True,  "quality":"Highest","cost":"High",   "speed":"Medium","provider":"Anthropic",      "default":True},
    {"id":"gemini-2.5-pro",     "icon":"🔵","label":"Gemini 2.5 Pro",                    "live":True,  "quality":"Highest","cost":"High",   "speed":"Medium","provider":"Google DeepMind","default":True},
    {"id":"claude-haiku-4.5",   "icon":"⚡","label":"Claude Haiku 4.5",                  "live":True,  "quality":"High",   "cost":"Medium", "speed":"Fast",  "provider":"Anthropic",      "default":True},
    {"id":"deepseek-r1",        "icon":"🧠","label":"DeepSeek R1",                       "live":True,  "quality":"High",   "cost":"Low",    "speed":"Slow",  "provider":"DeepSeek AI",    "default":False},
    {"id":"glm-4.5",            "icon":"🟢","label":"GLM 4.5",                           "live":True,  "quality":"High",   "cost":"Low",    "speed":"Medium","provider":"Zhipu AI",       "default":False},
    {"id":"kimi-k2",            "icon":"🌙","label":"Kimi K2",                           "live":True,  "quality":"High",   "cost":"Medium", "speed":"Medium","provider":"Moonshot AI",    "default":False},
    {"id":"qwen3-32b",          "icon":"🔶","label":"Qwen3 32B",                         "live":True,  "quality":"Good",   "cost":"Lowest", "speed":"Fast",  "provider":"Alibaba Cloud", "default":False},
    # ── Benchmark-evaluated models (simulated / proxy API) ───────────────────
    {"id":"gpt-4o",             "icon":"🤖","label":"GPT-4o",                            "live":False, "quality":"Highest","cost":"High",   "speed":"Medium","provider":"OpenAI",         "default":False},
    {"id":"openai-o3",          "icon":"💡","label":"OpenAI o3",                         "live":False, "quality":"Highest","cost":"High",   "speed":"Slow",  "provider":"OpenAI",         "default":False},
    {"id":"openai-o4-mini",     "icon":"⚡","label":"OpenAI o4-mini",                    "live":False, "quality":"High",   "cost":"Low",    "speed":"Fast",  "provider":"OpenAI",         "default":False},
    {"id":"claude-3-7-sonnet",  "icon":"🟣","label":"Claude 3.7 Sonnet (Ext. Thinking)", "live":False, "quality":"Highest","cost":"High",   "speed":"Slow",  "provider":"Anthropic",      "default":False},
    {"id":"claude-3-5-sonnet",  "icon":"🟣","label":"Claude 3.5 Sonnet (Ext. Thinking)", "live":False, "quality":"Highest","cost":"High",   "speed":"Slow",  "provider":"Anthropic",      "default":False},
    {"id":"gemini-flash",       "icon":"🔵","label":"Gemini 2.0 Flash Thinking",         "live":False, "quality":"Good",   "cost":"Low",    "speed":"Fast",  "provider":"Google DeepMind","default":False},
    {"id":"openai-o3-mini",     "icon":"💡","label":"OpenAI o3-mini",                    "live":False, "quality":"High",   "cost":"Low",    "speed":"Medium","provider":"OpenAI",         "default":False},
    {"id":"openai-o1",          "icon":"💡","label":"OpenAI o1",                         "live":False, "quality":"High",   "cost":"High",   "speed":"Slow",  "provider":"OpenAI",         "default":False},
    {"id":"openai-o1-mini",     "icon":"💡","label":"OpenAI o1-mini",                    "live":False, "quality":"Good",   "cost":"Medium", "speed":"Medium","provider":"OpenAI",         "default":False},
    {"id":"qwen-qwq-32b",       "icon":"🔶","label":"Qwen QwQ-32B",                      "live":False, "quality":"High",   "cost":"Low",    "speed":"Medium","provider":"Alibaba Cloud", "default":False},
    {"id":"deepl",              "icon":"📚","label":"DeepL Pro",                         "live":False, "quality":"High",   "cost":"Medium", "speed":"Fast",  "provider":"DeepL SE",       "default":False},
    {"id":"azure-mt",           "icon":"☁️", "label":"Azure Translator",                 "live":False, "quality":"Good",   "cost":"Medium", "speed":"Fast",  "provider":"Microsoft",      "default":False},
    {"id":"llama-3.3-70b",      "icon":"🦙","label":"Llama 3.3 70B",                     "live":False, "quality":"Good",   "cost":"Low",    "speed":"Medium","provider":"Meta AI",        "default":False},
    {"id":"mistral-large",      "icon":"🌟","label":"Mistral Large",                     "live":False, "quality":"High",   "cost":"Medium", "speed":"Fast",  "provider":"Mistral AI",     "default":False},
]

# Onboard catalog with registration metadata
ONBOARD_CATALOG = {
    m["label"]: {
        "model_id":           m["id"],
        "model_name":         m["label"],
        "provider":           m["provider"],
        "api_type":           "openai_compat" if m["live"] else "simulated",
        "cost_per_1k_tokens": {"Highest":0.009,"High":0.003,"Good":0.001,"Lowest":0.0003}.get(m["quality"],0.002),
        "base_latency":       {"Fast":1.0,"Medium":2.5,"Slow":15.0}.get(m["speed"],2.5),
        "color":              "#4F46E5",
        "live":               m["live"],
        "quality":            m["quality"],
        "cost":               m["cost"],
        "speed":              m["speed"],
        "icon":               m["icon"],
    }
    for m in ALL_MODELS_EXTENDED
}


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_cost(v) -> str:
    """Format a USD cost value with appropriate precision."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "$—"
    if v == 0:
        return "$0.00"
    if v >= 1:
        return f"${v:.2f}"
    if v >= 0.01:
        return f"${v:.4f}"
    if v >= 0.001:
        return f"${v:.5f}"
    if v >= 0.0001:
        return f"${v:.6f}"
    if v >= 0.00001:
        return f"${v:.7f}"
    return f"${v:.2e}"

def backend_ok():
    try: return requests.get(f"{BACKEND}/health", timeout=3).status_code == 200
    except Exception: return False

def start_eval(file_bytes, filename, lang, models, lat, cost):
    r = requests.post(f"{BACKEND}/api/evaluate",
        files={"file":(filename, file_bytes, "text/csv")},
        data={"target_language":lang,"models":json.dumps(models),"latency_priority":lat,"cost_priority":cost},
        timeout=30)
    r.raise_for_status()
    return r.json()["job_id"]

def poll(jid):
    r = requests.get(f"{BACKEND}/api/jobs/{jid}", timeout=10); r.raise_for_status(); return r.json()

def get_res(jid):
    r = requests.get(f"{BACKEND}/api/results/{jid}", timeout=10); r.raise_for_status(); return r.json()

def color_c(v, good=85, warn=72):
    if not isinstance(v,(int,float)): return ""
    if v >= good: return "color:#16A34A;font-weight:700"
    if v >= warn: return "color:#D97706"
    return "color:#DC2626"

def color_status(v):
    v = str(v)
    if "Selected" in v: return "background:#DCFCE7;color:#16A34A;font-weight:700"
    if "Review Required" in v: return "background:#FEF3C7;color:#B45309;font-weight:700"
    if "Rejected" in v: return "background:#FEE2E2;color:#DC2626"
    return "background:#DBEAFE;color:#2563EB"


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "results_map": None,
    "active_key": None,
    "error": None, "processing": False,
    "job_queue": None,
    "files": None,
    "lat_priority": "Good", "cost_priority": "Good",
    "page": "router",        # "router" | "evaluate" | "onboard"
    **{m["id"]: m["default"] for m in MODEL_DEFS},
    "langs": ["German (de)"],
}
for k,v in _DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR + NAV
# ══════════════════════════════════════════════════════════════════════════════
ok = backend_ok()
be_cls  = "online" if ok else "offline"
be_text = "Backend Online" if ok else "Backend Offline"
be_dot  = "dot-green" if ok else "dot-red"
st.markdown(f"""
<div class="navbar">
  <div class="nb-brand">
    <div class="nb-logo">🌐</div>
    <span><span class="brand-primary">ModelMatch</span> AI</span>
  </div>
  <div class="nb-right">
    <span class="nb-tagline">MT Provider Selection Engine</span>
    <span class="nb-badge {be_cls}"><span class="status-dot {be_dot}"></span>{be_text}</span>
    <span class="nb-badge"><span class="status-dot dot-blue"></span>504 Decisions</span>
    <span class="nb-badge"><span class="status-dot dot-amber"></span>Hackathon 2026</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Premium Navigation ────────────────────────────────────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
nav1, nav2, nav3, _sp = st.columns([1.4, 1.6, 1.4, 2])
with nav1:
    if st.button("�  Decision Engine", use_container_width=True,
                 type="primary" if st.session_state.page == "router" else "secondary"):
        st.session_state.page = "router"; st.rerun()
with nav2:
    if st.button("📊  Benchmark Center", use_container_width=True,
                 type="primary" if st.session_state.page == "evaluate" else "secondary"):
        st.session_state.page = "evaluate"; st.rerun()
with nav3:
    if st.button("➕  Model Marketplace", use_container_width=True,
                 type="primary" if st.session_state.page == "onboard" else "secondary"):
        st.session_state.page = "onboard"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: AUTO MODEL ROUTER  (benchmark-driven recommendations)
# ══════════════════════════════════════════════════════════════════════════════
LANG_DISPLAY = {
    "deu": "🇩🇪 German (deu)",
    "fra": "🇫🇷 French (fra)",
    "esp": "🇪🇸 Spanish (esp)",
    "kor": "🇰🇷 Korean (kor)",
    "chs": "🇨🇳 Chinese Simplified (chs)",
    "jpa": "🇯🇵 Japanese (jpa)",
}
DOMAIN_LIST    = ["E-commerce Product","Finance/Banking","IT Software","Journals Publishing","Legal Compliance","Multimedia Streaming","Pharma/Healthcare"]
CONTENT_LIST   = ["UI","Info","Warning","Error"]
PRIORITY_LIST  = ["Low","Good","High"]

if st.session_state.page == "router":
    st.markdown("""
    <div class="hero">
      <div class="hero-tag">🎯 DECISION ENGINE</div>
      <h1>Instant <span>AI Routing</span> Decision</h1>
      <p>Configure your language, domain and quality priorities. The Decision Engine queries
         4,200 pre-computed evaluation decisions and surfaces the optimal MT provider — instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ──────────────────────────────────────────────────────────────
    st.markdown('<div class="step-hdr"><span>🔧</span>Configure Your Requirements</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        r_lang = st.selectbox("🌍 Target Language", list(LANG_DISPLAY.keys()),
                              format_func=lambda x: LANG_DISPLAY[x])
        r_content = st.selectbox("📝 Content Type", CONTENT_LIST)
    with fc2:
        r_domain = st.selectbox("🏢 Domain", DOMAIN_LIST)
        r_latency = st.selectbox("⚡ Latency Priority", PRIORITY_LIST, index=1)
    with fc3:
        r_cost = st.selectbox("💰 Cost Priority", PRIORITY_LIST, index=1)
        st.markdown("<br>", unsafe_allow_html=True)
        find_btn = st.button("🚀 Find Best Model", type="primary", use_container_width=True)

    if find_btn or True:   # always show results as filters change
        try:
            resp = requests.get(f"{BACKEND}/api/benchmark",
                params={"language": r_lang, "content_type": r_content, "domain": r_domain},
                timeout=8)
            resp.raise_for_status()
            bench_rows = resp.json().get("results", [])
        except Exception as e:
            st.error(f"Could not load benchmark data: {e}")
            bench_rows = []

        # ── Score rows by user priority weights ───────────────────────────
        # adj "Low" = user doesn't care → near-zero weight → quality dominates
        # adj "High" = user cares a lot → higher weight on that dimension
        adj = {"Low": -0.30, "Good": 0.0, "High": 0.15}
        _q = 0.50
        _c = max(0.0, 0.30 + adj[r_cost])
        _l = max(0.0, 0.20 + adj[r_latency])
        _tot = _q + _c + _l if (_q + _c + _l) > 0 else 1
        w_q, w_c, w_l = _q/_tot, _c/_tot, _l/_tot

        def _rerank(row):
            q_s   = row.get("quality_score", 0)
            # Pre-computed tier scores: Low tier (fast/cheap) = 100, Good = 80, High (slow/expensive) = 55
            lat_s = row.get("latency_score", 70)
            cst_s = row.get("cost_score", 70)
            return round(w_q * q_s + w_c * cst_s + w_l * lat_s, 1)

        for row in bench_rows:
            row["priority_score"] = _rerank(row)
        bench_rows.sort(key=lambda r: r["priority_score"], reverse=True)

        if not bench_rows:
            st.info("No benchmark data found for this exact combination. Try adjusting filters.")
        else:
            winner = bench_rows[0]
            others = bench_rows[1:]

            wq_pct, wc_pct, wl_pct = round(w_q*100), round(w_c*100), round(w_l*100)
            dom_driver = "Quality" if wq_pct >= wc_pct and wq_pct >= wl_pct else ("Latency" if wl_pct >= wc_pct else "Cost")

            # ── Winner card ───────────────────────────────────────────────
            q   = winner["quality_score"]
            lat = winner["avg_latency"]
            cst = winner["total_cost"]
            sc  = winner["priority_score"]
            gov_badge = '<span class="de-gov-pass">✅ Governance Passed</span>' if winner.get("governance_passed") else '<span class="de-gov-fail">⚠️ Check Governance</span>'

            st.markdown(f"""
            <div class="de-winner">
              <div class="de-score-box">
                <div class="de-score-val">{sc:.1f}</div>
                <div class="de-score-lbl">Priority Score</div>
              </div>
              <div class="de-label">🎯 AI ROUTING DECISION</div>
              <div class="de-name">{winner['model_name']}</div>
              <div class="de-provider">{winner['provider']} &nbsp;·&nbsp; {LANG_DISPLAY.get(winner['language'], winner['language'])} &nbsp;·&nbsp; {winner['domain']} &nbsp;·&nbsp; {winner['content_type']}</div>
              {gov_badge}
              <div class="de-kpis">
                <div class="de-kpi"><div class="de-kpi-val" style="color:{'#6EE7B7' if q>=85 else '#FCD34D'};">{q:.1f}</div><div class="de-kpi-lbl">Quality</div></div>
                <div class="de-kpi"><div class="de-kpi-val" style="color:#C4B5FD;">{winner.get('terminology_accuracy',0):.1f}</div><div class="de-kpi-lbl">Terminology</div></div>
                <div class="de-kpi"><div class="de-kpi-val" style="color:#C4B5FD;">{winner.get('fluency_score',0):.1f}</div><div class="de-kpi-lbl">Fluency</div></div>
                <div class="de-kpi"><div class="de-kpi-val" style="color:{'#6EE7B7' if lat<10 else '#FCD34D'};">{lat:.0f}s</div><div class="de-kpi-lbl">Latency</div></div>
                <div class="de-kpi"><div class="de-kpi-val" style="color:{'#6EE7B7' if cst<0.01 else '#FCD34D'};">{fmt_cost(cst)}</div><div class="de-kpi-lbl">Cost/Run</div></div>
                <div class="de-kpi"><div class="de-kpi-val" style="color:#FCA5A5;">{winner.get('hallucination_risk',0)*100:.1f}%</div><div class="de-kpi-lbl">Hall. Risk</div></div>
              </div>
              <div class="de-pills">
                <span style="font-size:11px;color:#64748B;">Applied weights:</span>
                <span class="de-pill">Quality {wq_pct}%</span>
                <span class="de-pill">Cost {wc_pct}%</span>
                <span class="de-pill">Latency {wl_pct}%</span>
                <span class="de-badge">⚡ {dom_driver}-driven</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── All candidates table ──────────────────────────────────────
            if bench_rows:
                st.markdown("#### 📊 All Benchmark Candidates")
                df_r = pd.DataFrame([{
                    "Rank":     f"#{i+1}",
                    "Model":    r["model_name"],
                    "Provider": r["provider"],
                    "Quality":  r["quality_score"],
                    "Terminology": r.get("terminology_accuracy",0),
                    "Fluency":  r.get("fluency_score",0),
                    "Latency(s)": r["avg_latency"],
                    "Cost($)":  r["total_cost"],
                    "Hall.Risk": f"{r.get('hallucination_risk',0)*100:.1f}%",
                    "Priority Score": r["priority_score"],
                    "Status": "✅ Selected" if i==0 else ("⚠️ Gov.Fail" if not r.get("governance_passed") else "🔵 Qualified"),
                } for i, r in enumerate(bench_rows)])
                st.dataframe(
                    df_r.style
                        .map(color_status, subset=["Status"])
                        .map(color_c, subset=["Quality","Fluency","Priority Score"]),
                    use_container_width=True, hide_index=True, height=220)

            # ── Radar comparison chart ────────────────────────────────────
            if len(bench_rows) >= 2:
                st.markdown("#### 🎯 Quality Dimension Comparison")
                cats = ["Quality","Terminology","Fluency","Meaning","Hall.Safety"]
                fig = go.Figure()
                for i, row in enumerate(bench_rows[:4]):
                    vals = [
                        row["quality_score"],
                        row.get("terminology_accuracy",80),
                        row.get("fluency_score",80),
                        row.get("meaning_preservation",80),
                        max(0, 100 - row.get("hallucination_risk",0.1)*200),
                    ]
                    fig.add_trace(go.Scatterpolar(
                        r=vals+[vals[0]], theta=cats+[cats[0]],
                        fill="toself", name=row["model_name"],
                        line=dict(color=COLORS[i%len(COLORS)], width=2), opacity=0.55))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,100], gridcolor="#E2E8F0"),
                               angularaxis=dict(gridcolor="#E2E8F0"), bgcolor="#FFFFFF"),
                    paper_bgcolor="#FFFFFF", font=dict(color="#334155"),
                    margin=dict(l=30,r=30,t=20,b=20), height=340, legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ONBOARD NEW MODEL
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "onboard":
    st.markdown("""
    <div class="hero">
      <div class="hero-tag">➕ MODEL MARKETPLACE</div>
      <h1>Add a Model to the <span>Marketplace</span></h1>
      <p>Select from our curated model catalog, choose your domain &amp; dataset — 
         then instantly benchmark any MT provider against your content.</p>
    </div>
    """, unsafe_allow_html=True)

    # Wizard progress
    st.markdown("""
    <div class="wizard-progress">
      <div class="wz-step active"><div class="wz-num">1</div><div class="wz-label">Select Model</div></div>
      <div class="wz-arrow">›</div>
      <div class="wz-step"><div class="wz-num">2</div><div class="wz-label">Dataset</div></div>
      <div class="wz-arrow">›</div>
      <div class="wz-step"><div class="wz-num">3</div><div class="wz-label">Benchmark</div></div>
      <div class="wz-arrow">›</div>
      <div class="wz-step"><div class="wz-num">4</div><div class="wz-label">Results</div></div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.error:
        st.error(f"⚠️ {st.session_state.error}")
        if st.button("Dismiss", key="dismiss_onb_err"):
            st.session_state.error = None; st.rerun()

    # ── Step 1: Select model from catalog ────────────────────────────────────
    st.markdown('<div class="step-hdr"><span>🤖</span>Step 1 — Select a Model from Catalog</div>', unsafe_allow_html=True)

    catalog_names = list(ONBOARD_CATALOG.keys())
    selected_model_name = st.selectbox(
        "Choose a model to onboard:",
        options=catalog_names,
        format_func=lambda x: f"{ONBOARD_CATALOG[x]['icon']}  {x}  ·  {ONBOARD_CATALOG[x]['provider']}  {'· 🟢 Live API' if ONBOARD_CATALOG[x]['live'] else '· ⚪ Benchmark'}",
        label_visibility="collapsed",
        help="Select the MT model you want to onboard and benchmark."
    )
    sel_meta = ONBOARD_CATALOG.get(selected_model_name, {})

    if sel_meta:
        live_badge = '<span style="background:#DCFCE7;color:#15803D;border-radius:6px;padding:3px 12px;font-size:11px;font-weight:700;">🟢 Live API</span>' if sel_meta["live"] else '<span style="background:#F1F5F9;color:#475569;border-radius:6px;padding:3px 12px;font-size:11px;font-weight:700;">⚪ Benchmark Evaluation</span>'
        q_col = "#16A34A" if sel_meta["quality"] == "Highest" else ("#2563EB" if sel_meta["quality"] == "High" else "#D97706")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#F8FAFC,#EEF2FF);border:1.5px solid #C7D2FE;
                    border-radius:16px;padding:20px 24px;margin:10px 0 18px;
                    display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
          <div style="font-size:36px;">{sel_meta['icon']}</div>
          <div style="flex:1;">
            <div style="font-size:18px;font-weight:900;color:#0F172A;margin-bottom:4px;">{selected_model_name}</div>
            <div style="font-size:13px;color:#64748B;margin-bottom:8px;">{sel_meta['provider']}</div>
            {live_badge}
          </div>
          <div style="display:flex;gap:10px;flex-wrap:wrap;">
            <div style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:10px 14px;text-align:center;min-width:70px;">
              <div style="font-size:13px;font-weight:800;color:{q_col};">{sel_meta['quality']}</div>
              <div style="font-size:10px;color:#94A3B8;text-transform:uppercase;margin-top:2px;">Quality</div>
            </div>
            <div style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:10px 14px;text-align:center;min-width:70px;">
              <div style="font-size:13px;font-weight:800;color:#0F172A;">{sel_meta['cost']}</div>
              <div style="font-size:10px;color:#94A3B8;text-transform:uppercase;margin-top:2px;">Cost</div>
            </div>
            <div style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:10px 14px;text-align:center;min-width:70px;">
              <div style="font-size:13px;font-weight:800;color:#0F172A;">{sel_meta['speed']}</div>
              <div style="font-size:10px;color:#94A3B8;text-transform:uppercase;margin-top:2px;">Speed</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Step 2: Dataset source ────────────────────────────────────────────────
    st.markdown('<div class="step-hdr"><span>📂</span>Step 2 — Choose Your Dataset</div>', unsafe_allow_html=True)
    with st.form("onboard_form", clear_on_submit=False):
        dataset_mode = st.radio("Dataset source:",
                                ["📤 Upload my own CSV", "🧪 Run built-in benchmark suite"],
                                horizontal=True)

        onb_file = None
        bench_domains_sel  = []
        bench_content_sel  = []
        bench_langs_sel    = []

        if dataset_mode == "📤 Upload my own CSV":
            oc1, oc2 = st.columns(2)
            onb_file = oc1.file_uploader("Upload CSV / TSV", type=["csv","tsv","txt"],
                                         accept_multiple_files=False, key="onb_upl")
            onb_lang_sel = oc2.selectbox("Target Language", list(LANG_OPTIONS.keys()), index=0)
            compare_builtin = st.checkbox("Also compare against Claude Haiku & Qwen3 baselines", value=True)
        else:
            st.markdown("**Select benchmark scope** — we'll run our curated datasets through the selected model:")
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                bench_domains_sel = st.multiselect(
                    "🏢 Domains", DOMAIN_LIST, default=["IT Software","E-commerce Product"])
            with bc2:
                bench_content_sel = st.multiselect(
                    "📝 Content Types", CONTENT_LIST, default=["UI","Info"])
            with bc3:
                bench_langs_sel = st.multiselect(
                    "🌍 Target Languages", list(LANG_DISPLAY.keys()),
                    format_func=lambda x: LANG_DISPLAY[x], default=["deu","fra"])
            compare_builtin = st.checkbox("Also compare against Claude Haiku & Qwen3 baselines", value=True)
            onb_lang_sel = list(LANG_OPTIONS.keys())[0]

        submitted = st.form_submit_button("🚀  Run Benchmark", type="primary", use_container_width=True)

    if submitted:
        errs = []
        if not sel_meta: errs.append("Please select a model.")
        if dataset_mode == "📤 Upload my own CSV" and onb_file is None:
            errs.append("Please upload a benchmark dataset.")
        if dataset_mode == "🧪 Run built-in benchmark suite" and not (bench_domains_sel and bench_content_sel and bench_langs_sel):
            errs.append("Select at least one domain, content type, and language.")
        if not backend_ok(): errs.append("Backend not reachable on localhost:8000.")

        if errs:
            st.error("⚠️ " + "  ".join(errs))
        else:
            m_id   = sel_meta["model_id"]
            m_name = sel_meta["model_name"]
            payload = {
                "model_id":           m_id,
                "model_name":         m_name,
                "provider":           sel_meta["provider"],
                "api_type":           sel_meta["api_type"],
                "cost_per_1k_tokens": sel_meta["cost_per_1k_tokens"],
                "base_latency":       sel_meta["base_latency"],
                "color":              sel_meta["color"],
            }
            try:
                with st.spinner(f"Adding {m_name} to marketplace…"):
                    r = requests.post(f"{BACKEND}/api/models", json=payload, timeout=15)
                if r.status_code not in (200, 201, 409):
                    st.error(f"Registration failed: {r.text}"); st.stop()
                if r.status_code != 409:
                    st.toast(f"✅ {m_name} added to Marketplace", icon="➕")

                models_to_run = [m_id] + (["claude-haiku-4.5", "qwen3-32b"] if compare_builtin else [])
                queue = []
                LANG_CODE_MAP = {"deu":"de","fra":"fr","esp":"es","kor":"ko","chs":"zh","jpa":"ja"}

                if dataset_mode == "📤 Upload my own CSV":
                    lang_code = LANG_OPTIONS[onb_lang_sel]
                    jid = start_eval(onb_file.getvalue(), onb_file.name, lang_code, models_to_run,
                                     PRIORITY_OPTIONS["Good"], PRIORITY_OPTIONS["Good"])
                    queue.append({"job_id": jid, "filename": onb_file.name,
                                  "lang": lang_code, "label": f"{onb_file.name} → {onb_lang_sel}"})
                else:
                    progress_ph = st.empty()
                    total_jobs = len(bench_domains_sel) * len(bench_langs_sel)
                    submitted_count = 0
                    for dom in bench_domains_sel:
                        try:
                            csv_resp = requests.get(
                                f"{BACKEND}/api/benchmark/data/{requests.utils.quote(dom)}", timeout=10)
                            csv_resp.raise_for_status()
                            csv_bytes = csv_resp.json()["csv"].encode("utf-8")
                            fname = f"benchmark_{dom.replace('/','_').replace(' ','_')}.csv"
                        except Exception as exc:
                            st.warning(f"Skipping {dom}: {exc}"); continue
                        for blang in bench_langs_sel:
                            lcode = LANG_CODE_MAP.get(blang, "de")
                            jid = start_eval(csv_bytes, fname, lcode, models_to_run,
                                             PRIORITY_OPTIONS["Good"], PRIORITY_OPTIONS["Good"])
                            lbl = f"{dom} → {LANG_DISPLAY.get(blang, blang)}"
                            queue.append({"job_id": jid, "filename": fname, "lang": lcode, "label": lbl})
                            submitted_count += 1
                            progress_ph.info(f"🚀 Submitted {submitted_count}/{total_jobs} jobs…")

                if not queue:
                    st.error("No benchmark jobs could be submitted."); st.stop()

                st.session_state.job_queue  = queue
                st.session_state.processing = True
                st.session_state.page       = "evaluate"
                st.session_state.error      = None
                st.toast(f"🚀 {len(queue)} benchmark job(s) started", icon="🚀")
                time.sleep(0.4); st.rerun()
            except Exception as exc:
                st.error(f"❌ Onboarding failed: {exc}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EVALUATE PROVIDERS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-tag">📊 BENCHMARK CENTER</div>
  <h1>Live <span>Provider</span> Benchmarking</h1>
  <p>Upload your dataset, select models &amp; languages, configure quality priorities — 
     get a data-driven recommendation powered by LLM-as-Judge evaluation.</p>
</div>
""", unsafe_allow_html=True)


# ── ROUTING (within evaluate page) ──────────────────────────────────────────
VIEW = "results" if st.session_state.results_map else ("processing" if st.session_state.processing else "setup")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: SETUP
# ══════════════════════════════════════════════════════════════════════════════
if VIEW == "setup":

    if st.session_state.error:
        st.error(f"⚠️ {st.session_state.error}")
        if st.button("Dismiss", key="dismiss_err"):
            st.session_state.error = None; st.rerun()

    # ── Step 1: Upload (multiple CSVs) ───────────────────────────────────────
    st.markdown('<div class="step-hdr"><span>📂</span>Step 1 — Upload CSV File(s)</div>', unsafe_allow_html=True)
    st.markdown('<div class="help-note">You can upload multiple CSV/TSV files. Each file is evaluated against every selected language.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload CSV / TSV", type=["csv","tsv","txt"],
        accept_multiple_files=True, label_visibility="collapsed",
        help="Column names like source, segment, src, EN are auto-detected.",
    )
    if uploaded:
        files = []
        for uf in uploaded:
            raw = uf.read(); files.append({"name": uf.name, "bytes": raw}); uf.seek(0)
        st.session_state.files = files
        for uf in uploaded:
            try:
                df_p = pd.read_csv(uf, nrows=4, on_bad_lines="skip"); uf.seek(0)
                st.markdown(
                    f'<div class="file-ok"><div class="fo-name">✓ {uf.name}</div>'
                    f'<span class="fo-chip">🔢 {len(df_p.columns)} columns</span>'
                    f'<span class="fo-chip">🏷️ {", ".join(df_p.columns[:5])}</span></div>',
                    unsafe_allow_html=True)
            except Exception:
                st.markdown(f'<div class="file-ok"><div class="fo-name">✓ {uf.name}</div></div>', unsafe_allow_html=True)
    elif st.session_state.files:
        st.success(f"✓ {len(st.session_state.files)} file(s) loaded: " + ", ".join(f['name'] for f in st.session_state.files))
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-state-icon">📂</div>
          <div class="empty-state-title">No Dataset Uploaded Yet</div>
          <div class="empty-state-sub">Upload your translation dataset to begin evaluation. Supports CSV, TSV, and TXT files. Column names like <code>source</code>, <code>segment</code>, <code>src</code> are auto-detected.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Step 2: Models dropdown (sorted live first) ──────────────────────────
    st.markdown('<div class="step-hdr"><span>🤖</span>Step 2 — Select MT Models to Evaluate</div>', unsafe_allow_html=True)
    st.markdown('<div class="help-note">Live API models run real inference. Benchmark-evaluated models use validated proxy scoring. Sorted: 🟢 Live first, then ⚪ Benchmark-evaluated.</div>', unsafe_allow_html=True)

    def _model_option_label(m):
        status = "🟢 Live API" if m["live"] else "⚪ Benchmark"
        return f"{m['icon']}  {m['label']}   ·   {m['provider']}   ·   Quality: {m['quality']}   ·   Cost: {m['cost']}   ·   Speed: {m['speed']}   [{status}]"

    _all_model_opts = {_model_option_label(m): m["id"] for m in ALL_MODELS_EXTENDED}
    _default_labels = [_model_option_label(m) for m in ALL_MODELS_EXTENDED if m.get("default")]

    selected_labels_raw = st.multiselect(
        "Choose models:",
        options=list(_all_model_opts.keys()),
        default=_default_labels,
        label_visibility="collapsed",
        help="Select one or more models. Live API models run actual translations."
    )
    selected_models = [_all_model_opts[l] for l in selected_labels_raw if l in _all_model_opts]

    # Show selected count with live/benchmark breakdown
    _n_live = sum(1 for l in selected_labels_raw if "🟢 Live API" in l)
    _n_bench = len(selected_models) - _n_live
    if selected_models:
        parts = []
        if _n_live:   parts.append(f"<span style='background:#DCFCE7;color:#15803D;border-radius:6px;padding:2px 10px;font-size:11px;font-weight:700;'>🟢 {_n_live} Live</span>")
        if _n_bench:  parts.append(f"<span style='background:#F1F5F9;color:#475569;border-radius:6px;padding:2px 10px;font-size:11px;font-weight:700;'>⚪ {_n_bench} Benchmark</span>")
        st.markdown(f"<div style='display:flex;gap:8px;align-items:center;margin-top:4px;'><span style='font-size:11px;color:#94A3B8;'>Selected:</span>{''.join(parts)}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Select at least one model.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Step 3: Languages + Priorities ───────────────────────────────────────
    c_lang, c_lat, c_cost = st.columns(3)
    with c_lang:
        st.markdown('<div class="step-hdr"><span>🌍</span>Step 3 — Target Languages</div>', unsafe_allow_html=True)
        langs = st.multiselect("Target languages", list(LANG_OPTIONS.keys()),
                               default=st.session_state.langs, label_visibility="collapsed",
                               help="Select one or more target languages.")
        st.session_state.langs = langs
    with c_lat:
        st.markdown('<div class="step-hdr"><span>⚡</span>Latency Priority</div>', unsafe_allow_html=True)
        lat = st.selectbox("Latency priority", list(PRIORITY_OPTIONS.keys()),
                           index=list(PRIORITY_OPTIONS).index(st.session_state.lat_priority),
                           label_visibility="collapsed",
                           help="High = strongly favor fast providers.")
        st.session_state.lat_priority = lat
    with c_cost:
        st.markdown('<div class="step-hdr"><span>💰</span>Cost Priority</div>', unsafe_allow_html=True)
        cost = st.selectbox("Cost priority", list(PRIORITY_OPTIONS.keys()),
                            index=list(PRIORITY_OPTIONS).index(st.session_state.cost_priority),
                            label_visibility="collapsed",
                            help="High = strongly favor cheap providers.")
        st.session_state.cost_priority = cost

    # ── Live weight preview ───────────────────────────────────────────────────
    _adj = {"Low": -0.10, "Good": 0.0, "High": 0.15}
    _q_base, _c_base, _l_base = 0.50, 0.30, 0.20
    _cw = max(0.05, _c_base + _adj.get(st.session_state.cost_priority, 0))
    _lw = max(0.05, _l_base + _adj.get(st.session_state.lat_priority, 0))
    _tot = _q_base + _cw + _lw
    _qp, _cp, _lp = round(_q_base/_tot*100), round(_cw/_tot*100), round(_lw/_tot*100)
    _driver = ("Quality" if _qp >= _cp and _qp >= _lp else "Latency" if _lp >= _cp else "Cost")
    st.markdown(f"""
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
                padding:12px 18px;margin-top:8px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;">
      <span style="font-size:12px;font-weight:700;color:#475569;">📐 Scoring weights preview:</span>
      <span style="background:#DBEAFE;color:#1D4ED8;border-radius:999px;padding:3px 11px;font-size:12px;font-weight:700;">
        Quality {_qp}%</span>
      <span style="background:#DCFCE7;color:#15803D;border-radius:999px;padding:3px 11px;font-size:12px;font-weight:700;">
        Cost {_cp}%</span>
      <span style="background:#FEF3C7;color:#B45309;border-radius:999px;padding:3px 11px;font-size:12px;font-weight:700;">
        Latency {_lp}%</span>
      <span style="background:#F1F5F9;color:#475569;border-radius:999px;padding:3px 11px;font-size:12px;font-weight:600;">
        ⚡ {_driver}-driven recommendation</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Run ──────────────────────────────────────────────────────────────────
    n_files = len(st.session_state.files) if st.session_state.files else 0
    n_langs = len(st.session_state.langs)
    n_jobs  = n_files * n_langs
    if n_jobs:
        st.markdown(f'<div class="help-note">This will run <b>{n_jobs}</b> evaluation job(s): '
                    f'{n_files} file(s) × {n_langs} language(s) × {len(selected_models)} model(s).</div>',
                    unsafe_allow_html=True)

    run_ok = bool(st.session_state.files) and bool(selected_models) and bool(st.session_state.langs)
    if st.button("🚀  Start Evaluation", type="primary", disabled=not run_ok, use_container_width=True):
        if not backend_ok():
            st.error("❌ Backend not reachable on `localhost:8000`. Start it with `start.bat`.")
        else:
            queue = []
            try:
                with st.spinner(f"🚀 Starting evaluation — submitting {n_jobs} job(s)…"):
                    for f in st.session_state.files:
                        for lbl in st.session_state.langs:
                            lang = LANG_OPTIONS[lbl]
                            jid = start_eval(f["bytes"], f["name"], lang, selected_models,
                                             PRIORITY_OPTIONS[st.session_state.lat_priority],
                                             PRIORITY_OPTIONS[st.session_state.cost_priority])
                            queue.append({"job_id":jid, "filename":f["name"], "lang":lang,
                                          "label":f'{f["name"]} → {lbl}'})
                st.session_state.job_queue  = queue
                st.session_state.processing = True
                st.session_state.error      = None
                st.toast(f"✅ Evaluation started — {len(queue)} job(s) running", icon="🚀")
                st.success(f"✅ Evaluation started — running {len(queue)} job(s). Loading live progress…")
                time.sleep(0.6)
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Could not start evaluation: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: PROCESSING  (poll all jobs in the queue)
# ══════════════════════════════════════════════════════════════════════════════
elif VIEW == "processing":
    st.markdown("""
    <div class="proc-hdr">
      <h2>🚀 Evaluation In Progress</h2>
      <p>Your providers are being benchmarked in real-time. AI agents are running in parallel —<br>
         translation, quality evaluation, governance checks, and explainability generation.</p>
    </div>
    """, unsafe_allow_html=True)

    queue = st.session_state.job_queue or []
    st.info(f"🚀 {len(queue)} evaluation job(s) running — please keep this tab open.")
    job_phs = {j["job_id"]: st.empty() for j in queue}
    overall_ph = st.empty()
    results_map = {}

    # Immediate feedback before the first poll completes
    for j in queue:
        job_phs[j["job_id"]].markdown(
            f'<div class="agent-row running"><div class="ar-icon">🔄</div>'
            f'<div class="ar-body"><div class="ar-name">{j["label"]}</div>'
            f'<div class="ar-desc">Initializing…</div></div><span class="ar-status ars-run">Started</span></div>',
            unsafe_allow_html=True)
    overall_ph.progress(0.0, text="**0%** — starting…")

    done = set()
    for _ in range(400):
        all_done = True
        total_pct = 0
        for j in queue:
            jid = j["job_id"]
            if jid in done:
                total_pct += 100; continue
            try:
                job = poll(jid)
            except Exception as e:
                st.session_state.error = f"Lost connection: {e}"; st.session_state.processing=False; st.rerun()
            pct = job.get("progress",0); stage = job.get("stage","")
            total_pct += pct
            if job["status"] == "completed":
                try: results_map[j["label"]] = get_res(jid)
                except Exception as e: st.session_state.error=f"Fetch failed: {e}"
                done.add(jid)
                job_phs[jid].markdown(
                    f'<div class="agent-row done"><div class="ar-icon">✅</div>'
                    f'<div class="ar-body"><div class="ar-name">{j["label"]}</div>'
                    f'<div class="ar-desc">Completed</div></div><span class="ar-status ars-done">Done</span></div>',
                    unsafe_allow_html=True)
            elif job["status"] == "failed":
                done.add(jid)
                results_map[j["label"]] = {"status":"failed","error":job.get("error","failed")}
                job_phs[jid].markdown(
                    f'<div class="agent-row"><div class="ar-icon">❌</div>'
                    f'<div class="ar-body"><div class="ar-name">{j["label"]}</div>'
                    f'<div class="ar-desc">{job.get("error","Failed")}</div></div>'
                    f'<span class="ar-status ars-pend">Failed</span></div>', unsafe_allow_html=True)
            else:
                all_done = False
                job_phs[jid].markdown(
                    f'<div class="agent-row running"><div class="ar-icon">🔄</div>'
                    f'<div class="ar-body"><div class="ar-name">{j["label"]}</div>'
                    f'<div class="ar-desc">{stage}</div></div><span class="ar-status ars-run">{pct}%</span></div>',
                    unsafe_allow_html=True)
        op = int(total_pct / max(1, len(queue)))
        overall_ph.progress(op/100, text=f"**{op}%** — {len(done)}/{len(queue)} job(s) complete")
        if all_done:
            st.session_state.results_map = results_map
            st.session_state.active_key  = next(iter(results_map), None)
            st.session_state.processing  = False
            overall_ph.success("✅ Evaluation complete — opening dashboard…")
            st.toast("✅ Done! Opening dashboard…", icon="📊")
            time.sleep(0.6); st.rerun()
        time.sleep(1.5)
    else:
        st.session_state.error="Evaluation timed out."; st.session_state.processing=False; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif VIEW == "results":
    results_map = st.session_state.results_map

    top_l, top_r = st.columns([4,1])
    with top_r:
        if st.button("🔄 New Evaluation", type="secondary", use_container_width=True):
            for k in ("results_map","active_key","processing","error","job_queue"):
                st.session_state[k] = None
            st.session_state.processing = False
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # RECOMMENDATION DASHBOARD — driven by language × priority settings
    # ══════════════════════════════════════════════════════════════════════
    valid_map = {k:v for k,v in results_map.items() if v.get("status") != "failed" and v.get("model_results")}

    # Collect all unique languages evaluated
    lang_winners = {}   # lang_code → list of (label, winner, weights, domain)
    for label, res in valid_map.items():
        w = next((m for m in res.get("model_results",[]) if m.get("selected")), None)
        dec = res.get("decision", {})
        weights = dec.get("weights_used", {})
        domain  = res.get("dataset_analysis", {}).get("detected_domain","general")
        lang_code = label.split("→")[-1].strip() if "→" in label else label
        lang_winners.setdefault(lang_code, []).append((label, w, weights, domain))

    if lang_winners:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#EFF6FF 0%,#F0FDF4 100%);
                    border:2px solid #BFDBFE;border-radius:18px;padding:20px 26px 14px;margin-bottom:22px;">
          <div style="font-size:11px;font-weight:700;color:#2563EB;text-transform:uppercase;
                      letter-spacing:.12em;margin-bottom:6px;">📊 RECOMMENDATION DASHBOARD</div>
          <div style="font-size:18px;font-weight:800;color:#0F172A;margin-bottom:4px;">
            Best Provider Per Language &amp; Priority</div>
          <div style="font-size:13px;color:#64748B;">
            Ranked by weighted score — Quality · Cost · Latency priorities applied per evaluation.</div>
        </div>
        """, unsafe_allow_html=True)

        for lang_label, entries in lang_winners.items():
            st.markdown(f"#### 🌍 {lang_label}")
            dash_cols = st.columns(len(entries)) if len(entries) > 1 else [st]
            for col, (label, winner, weights, domain) in zip(dash_cols if len(entries)>1 else [st], entries):
                if winner is None:
                    col.warning(f"No winner for `{label}`")
                    continue
                met = winner["metrics"]
                wq  = round(weights.get("quality_weight",0)*100)
                wc  = round(weights.get("cost_weight",0)*100)
                wl  = round(weights.get("latency_weight",0)*100)
                q   = met["quality_score"]
                lat = met["avg_latency"]
                cst = met["total_cost"]
                score = winner["weighted_score"]
                qcol  = "#16A34A" if q>=85 else ("#D97706" if q>=72 else "#DC2626")
                lcol  = "#16A34A" if lat<2 else "#D97706"
                ccol  = "#16A34A" if cst<0.01 else "#D97706"

                # Determine dominant priority label
                max_w = max(wq, wc, wl)
                if max_w == wq:   driver = f"Quality-driven ({wq}%)"
                elif max_w == wl: driver = f"Latency-driven ({wl}%)"
                else:             driver = f"Cost-driven ({wc}%)"

                col.markdown(f"""
                <div style="background:#FFFFFF;border:2px solid #BFDBFE;border-radius:14px;
                            padding:18px 20px;margin-bottom:10px;">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                      <div style="font-size:10px;font-weight:700;color:#2563EB;text-transform:uppercase;
                                  letter-spacing:.1em;margin-bottom:4px;">🏆 RECOMMENDED</div>
                      <div style="font-size:22px;font-weight:900;color:#0F172A;">{winner['model_name']}</div>
                      <div style="font-size:12px;color:#64748B;margin-top:2px;">
                        {winner['provider']} &nbsp;·&nbsp; {domain.capitalize()} domain</div>
                    </div>
                    <div style="text-align:center;background:#EFF6FF;border:2px solid #BFDBFE;
                                border-radius:12px;padding:10px 16px;min-width:72px;">
                      <div style="font-size:28px;font-weight:900;color:#2563EB;line-height:1;">{score:.1f}</div>
                      <div style="font-size:10px;color:#64748B;text-transform:uppercase;">Score</div>
                    </div>
                  </div>
                  <div style="display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;">
                    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
                                padding:8px 12px;text-align:center;min-width:72px;">
                      <div style="font-size:17px;font-weight:800;color:{qcol};">{q:.1f}</div>
                      <div style="font-size:10px;color:#94A3B8;text-transform:uppercase;">Quality</div>
                    </div>
                    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
                                padding:8px 12px;text-align:center;min-width:72px;">
                      <div style="font-size:17px;font-weight:800;color:{lcol};">{lat:.2f}s</div>
                      <div style="font-size:10px;color:#94A3B8;text-transform:uppercase;">Latency</div>
                    </div>
                    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
                                padding:8px 12px;text-align:center;min-width:72px;">
                      <div style="font-size:17px;font-weight:800;color:{ccol};">{fmt_cost(cst)}</div>
                      <div style="font-size:10px;color:#94A3B8;text-transform:uppercase;">Cost/Run</div>
                    </div>
                  </div>
                  <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:12px;align-items:center;">
                    <span style="font-size:11px;color:#64748B;">Weights applied:</span>
                    <span style="background:#DBEAFE;color:#1D4ED8;border-radius:999px;
                                 padding:2px 9px;font-size:11px;font-weight:600;">Quality {wq}%</span>
                    <span style="background:#DCFCE7;color:#15803D;border-radius:999px;
                                 padding:2px 9px;font-size:11px;font-weight:600;">Cost {wc}%</span>
                    <span style="background:#FEF3C7;color:#B45309;border-radius:999px;
                                 padding:2px 9px;font-size:11px;font-weight:600;">Latency {wl}%</span>
                    <span style="background:#F1F5F9;color:#475569;border-radius:999px;
                                 padding:2px 9px;font-size:11px;font-weight:600;">⚡ {driver}</span>
                  </div>
                  <div style="font-size:11px;color:#94A3B8;margin-top:8px;">
                    📁 {label.split('→')[0].strip() if '→' in label else label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

    # ── Cross-combination summary table (multi-result) ───────────────────────
    if len(results_map) > 1:
        st.markdown('<div class="scard-title"><span>🗂️</span>Selection Summary — Winner per File × Language</div>', unsafe_allow_html=True)
        sum_rows = []
        for label, res in results_map.items():
            if res.get("status") == "failed":
                sum_rows.append({"Combination":label,"Domain":"—","Recommended Provider":"❌ Failed","Score":None,"Quality":None})
                continue
            w = next((m for m in res.get("model_results",[]) if m.get("selected")), None)
            d = res.get("dataset_analysis",{})
            sum_rows.append({
                "Combination":label,
                "Domain":d.get("detected_domain","—").capitalize(),
                "Recommended Provider": w["model_name"] if w else "No winner",
                "Score": round(w["weighted_score"],1) if w else None,
                "Quality": round(w["metrics"]["quality_score"],1) if w else None,
            })
        df_sum = pd.DataFrame(sum_rows)
        df_sum["Score"]   = pd.to_numeric(df_sum["Score"],   errors="coerce")
        df_sum["Quality"] = pd.to_numeric(df_sum["Quality"], errors="coerce")
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
        st.divider()

    # ── Combination selector ─────────────────────────────────────────────────
    keys = list(results_map.keys())
    active = st.selectbox("View detailed results for:", keys,
                          index=keys.index(st.session_state.active_key) if st.session_state.active_key in keys else 0)
    st.session_state.active_key = active
    res = results_map[active]

    if res.get("status") == "failed":
        st.error(f"This combination failed: {res.get('error')}")
        st.stop()

    models  = res.get("model_results", [])
    dec     = res.get("decision", {})
    expl    = res.get("explainability", {})
    dinfo   = res.get("dataset_analysis", {})
    weights = dec.get("weights_used", {})
    domain  = dinfo.get("detected_domain","general")
    summary = expl.get("summary","")
    exp_map = {e["model_id"]:e for e in expl.get("model_explanations",[])}
    winner  = next((m for m in models if m.get("selected")), None)
    wq = weights.get("quality_weight",0)*100; wc = weights.get("cost_weight",0)*100; wl = weights.get("latency_weight",0)*100

    # ── Winner banner ────────────────────────────────────────────────────────
    if winner:
        met = winner["metrics"]
        q,trm,flu = met["quality_score"],met["terminology_accuracy"],met["fluency_score"]
        lat_v,cst = met["avg_latency"],met["total_cost"]
        qc = "c-green" if q>=85 else ("c-amber" if q>=72 else "c-red")
        tc = "c-green" if trm>=85 else ("c-amber" if trm>=72 else "c-red")
        fc = "c-green" if flu>=85 else ("c-amber" if flu>=72 else "c-red")
        lc = "c-green" if lat_v<2 else "c-amber"
        cc = "c-green" if cst<0.01 else "c-amber"
        hrr = met.get("human_review_required", False)
        hrr_badge = (
            '<div style="background:#FEF3C7;border:1.5px solid #F59E0B;border-radius:8px;'
            'padding:8px 14px;margin-top:12px;display:flex;align-items:center;gap:8px;">'
            '<span style="font-size:18px;">⚠️</span>'
            '<div><div style="font-size:12px;font-weight:700;color:#B45309;">HUMAN REVIEW REQUIRED</div>'
            '<div style="font-size:11px;color:#92400E;">Rule-based checks detected issues that require human validation before production use.</div>'
            '</div></div>'
        ) if hrr else ""
        hard_fails = met.get("hard_fails", [])
        hf_html = ""
        if hard_fails:
            hf_items = "".join(
                f'<li style="font-size:11px;color:#92400E;margin-bottom:3px;">'
                f'<b>{hf["code"]}</b>: {hf["description"]}</li>'
                for hf in hard_fails[:5]
            )
            hf_html = f'<ul style="margin:6px 0 0 18px;padding:0;">{hf_items}</ul>'
        st.markdown(f"""
        <div class="winner-wrap">
          <div class="w-score"><div class="ws-val">{winner['weighted_score']:.1f}</div><div class="ws-label">Overall Score</div></div>
          <div class="w-label">🏆 RECOMMENDED PROVIDER</div>
          <div class="w-name">{winner['model_name']}</div>
          <div class="w-sub">{winner['provider']} &nbsp;·&nbsp; {domain.capitalize()} domain &nbsp;·&nbsp; {active}</div>
          <div class="w-kpis">
            <div class="wkpi"><div class="wkv {qc}">{q:.1f}</div><div class="wkl">Quality</div></div>
            <div class="wkpi"><div class="wkv {tc}">{trm:.1f}</div><div class="wkl">Terminology</div></div>
            <div class="wkpi"><div class="wkv {fc}">{flu:.1f}</div><div class="wkl">Fluency</div></div>
            <div class="wkpi"><div class="wkv {lc}">{lat_v:.2f}s</div><div class="wkl">Latency</div></div>
            <div class="wkpi"><div class="wkv {cc}">{fmt_cost(cst)}</div><div class="wkl">Cost/run</div></div>
          </div>
          <div class="w-pills"><span class="w-pill-l">Applied weights:</span>
            <span class="w-pill">Quality {wq:.0f}%</span><span class="w-pill">Cost {wc:.0f}%</span><span class="w-pill">Latency {wl:.0f}%</span>
          </div>
          {hrr_badge}{hf_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"No model passed governance for the '{domain}' domain. Consider relaxing business rules.")

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Source Language", dinfo.get("source_language_name","—"))
    m2.metric("Total Rows", dinfo.get("total_rows","—"))
    m3.metric("Unique Rows", dinfo.get("unique_rows","—"))
    m4.metric("Domain", domain.capitalize())
    st.divider()

    t1,t2,t3,t4,t5 = st.tabs(["📊 Rankings","🎯 Charts","🤖 AI Explanation","🧪 Sampling","🔍 Decision Trace"])

    # ─ Tab 1: Rankings ───────────────────────────────────────────────────────
    with t1:
        rows = []
        for mdl in models:
            met, gov = mdl["metrics"], mdl["governance_check"]
            hrr_mdl = met.get("human_review_required", False)
            status = ("✅ Selected" if mdl.get("selected") else "❌ Rejected" if not gov["passed"] else "🔵 Qualified")
            if hrr_mdl and mdl.get("selected"):
                status = "⚠️ Review Required"
            rows.append({
                "Rank":f"#{mdl['rank']}","Model":mdl["model_name"],"Provider":mdl["provider"],
                "Quality":round(met["quality_score"],1),
                "Meaning":round(met.get("meaning_preservation",0),1),
                "Terminology":round(met["terminology_accuracy"],1),
                "Fluency":round(met["fluency_score"],1),
                "Placeholder":round(met.get("placeholder_preservation",0),1),
                "Latency (s)":round(met["avg_latency"],3),"Cost ($)":fmt_cost(met["total_cost"]),
                "Hall. Risk":f"{met['hallucination_risk']*100:.1f}%",
                "Score":round(mdl["weighted_score"],1),"Status":status,
            })
        df_t = pd.DataFrame(rows)
        styled = df_t.style.map(color_status, subset=["Status"]).map(color_c, subset=["Quality","Meaning","Fluency","Placeholder","Score"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=260)
        st.caption("Quality is a composite of 5 localization dimensions (linguistic, localization tokens, domain, cultural, hallucination). Scores are deliberately strict — machine output rarely exceeds the mid-80s.")

    # ─ Tab 2: Charts ─────────────────────────────────────────────────────────
    with t2:
        cr, cb = st.columns(2)
        with cr:
            st.markdown("**Quality Dimensions — Radar**")
            cats = ["Quality","Terminology","Fluency","Meaning","Hall. Safety"]
            fig = go.Figure()
            for i,mdl in enumerate(models):
                met = mdl["metrics"]
                vals = [met["quality_score"],met["terminology_accuracy"],met["fluency_score"],
                        met.get("meaning_preservation",80),max(0,100-met["hallucination_risk"]*200)]
                fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],fill="toself",
                              name=mdl["model_name"],line=dict(color=COLORS[i%len(COLORS)],width=2),opacity=.55))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100],gridcolor="#E2E8F0"),
                              angularaxis=dict(gridcolor="#E2E8F0"),bgcolor="#FFFFFF"),
                              paper_bgcolor="#FFFFFF",font=dict(color="#334155"),
                              margin=dict(l=30,r=30,t=20,b=20),height=380)
            st.plotly_chart(fig, use_container_width=True)
        with cb:
            st.markdown("**Weighted Scores — Leaderboard**")
            bd = sorted(models,key=lambda x:x["weighted_score"],reverse=True)
            bc = ["#16A34A" if m.get("selected") else "#DC2626" if not m["governance_check"]["passed"] else "#2563EB" for m in bd]
            figb = go.Figure(go.Bar(x=[m["model_name"] for m in bd],y=[m["weighted_score"] for m in bd],
                             marker_color=bc,text=[f"{m['weighted_score']:.1f}" for m in bd],textposition="outside"))
            figb.update_layout(paper_bgcolor="#FFFFFF",plot_bgcolor="#F8FAFC",font=dict(color="#334155"),
                               yaxis=dict(range=[0,108],gridcolor="#E2E8F0",title="Score"),
                               xaxis=dict(gridcolor="#E2E8F0"),showlegend=False,margin=dict(l=30,r=30,t=20,b=40),height=380)
            st.plotly_chart(figb, use_container_width=True)

        st.markdown("**Score Component Breakdown**")
        df_c = pd.DataFrame([{
            "Model":m["model_name"],
            "Quality":round(m.get("score_components",{}).get("quality_contribution",0),1),
            "Cost":round(m.get("score_components",{}).get("cost_contribution",0),1),
            "Latency":round(m.get("score_components",{}).get("latency_contribution",0),1),
        } for m in models])
        figs = go.Figure()
        for col_name,color in [("Quality","#2563EB"),("Cost","#16A34A"),("Latency","#D97706")]:
            figs.add_trace(go.Bar(name=col_name,x=df_c["Model"],y=df_c[col_name],marker_color=color))
        figs.update_layout(barmode="stack",paper_bgcolor="#FFFFFF",plot_bgcolor="#F8FAFC",font=dict(color="#334155"),
                           yaxis=dict(gridcolor="#E2E8F0",title="Score Points"),margin=dict(l=30,r=30,t=10,b=40),height=300)
        st.plotly_chart(figs, use_container_width=True)

    # ─ Tab 3: AI Explanation ─────────────────────────────────────────────────
    with t3:
        if summary:
            st.markdown(f'<div class="ex-card winner-ex"><div class="ex-head"><div class="ex-icon">🤖</div>'
                        f'<div><div class="ex-title">AI Recommendation Summary</div>'
                        f'<div class="ex-sub">Generated by the Explainability Agent</div></div></div>'
                        f'<div class="ex-body">{summary}</div></div>', unsafe_allow_html=True)
        for mdl in models:
            exp = exp_map.get(mdl["model_id"],{})
            if not exp: continue
            status = exp.get("status","qualified")
            icon = "🏆" if status=="selected" else ("❌" if status=="rejected" else "🔵")
            card_c = "winner-ex" if status=="selected" else "reject-ex" if status=="rejected" else ""
            bullets = "".join(f"<li>{r}</li>" for r in exp.get("reasons_list",[]))
            st.markdown(f'<div class="ex-card {card_c}"><div class="ex-head"><div class="ex-icon">{icon}</div>'
                        f'<div><div class="ex-title">{exp["model_name"]}</div>'
                        f'<div class="ex-sub">{exp["provider"]} · Rank #{exp["rank"]} · Score {mdl["weighted_score"]:.1f}</div></div></div>'
                        f'<div class="ex-body">{exp.get("reason","")}</div><ul class="ex-bullets">{bullets}</ul></div>',
                        unsafe_allow_html=True)
            gov = mdl.get("governance_check",{})
            if not gov.get("passed") and gov.get("violations"):
                st.error("Governance violations: " + " | ".join(gov["violations"]))

    # ─ Tab 4: Sampling ───────────────────────────────────────────────────────
    with t4:
        st.markdown("**🧪 How evaluation samples were selected**")
        cov = dinfo.get("sample_coverage", {})
        if cov:
            s1,s2,s3,s4 = st.columns(4)
            s1.metric("Unique Segments", cov.get("total_unique","—"))
            s2.metric("Samples Chosen", cov.get("selected","—"))
            s3.metric("With Format Tokens", cov.get("with_format_tokens","—"))
            s4.metric("Avg Words", cov.get("avg_word_count","—"))
            st.caption(f"Strategy: {cov.get('strategy','—')}")
        reasons = dinfo.get("sample_selection_reasons", {})
        if reasons:
            st.dataframe(pd.DataFrame(
                [{"Selected Segment": (t[:80]+"…") if len(t)>80 else t, "Why It Was Chosen": why}
                 for t, why in reasons.items()]),
                use_container_width=True, hide_index=True)
        else:
            st.info("Sample selection details not available for this run.")

    # ─ Tab 5: Decision Trace ─────────────────────────────────────────────────
    with t5:
        st.markdown("**🔍 Decision Trace — How the engine reached this recommendation**")
        best_q = max((m["metrics"]["quality_score"] for m in models), default=0)
        trace = [
            ("📂 Dataset Agent", f"Detected domain: {domain.capitalize()}",
             f"Source: {dinfo.get('source_language_name','?')} · {dinfo.get('total_rows','?')} rows · "
             f"{dinfo.get('unique_rows','?')} unique · {dinfo.get('samples_selected','?')} smart samples"),
            ("🌍 Translation Agent", f"Ran {len(models)} MT provider(s) in parallel",
             ", ".join(m["model_name"] for m in models)),
            ("🎯 Evaluation Agent (LLM-as-Judge)", f"Strict 5-dimension QE — best quality {best_q:.1f}/100",
             "Linguistic · Localization tokens · Domain · Cultural · Hallucination"),
            ("⚖️ Governance Agent", f"Applied {domain.capitalize()} rules — Q {wq:.0f}% · C {wc:.0f}% · L {wl:.0f}%",
             f"{sum(1 for m in models if m['governance_check']['passed'])} passed · "
             f"{sum(1 for m in models if not m['governance_check']['passed'])} rejected"),
            ("🤖 Explainability Agent",
             f"Recommended: {winner['model_name']} — Score {winner['weighted_score']:.1f}" if winner else "No winner",
             (summary[:160]+"…") if len(summary)>160 else summary),
        ]
        trace_html = ""
        for i,(agent,fact,detail) in enumerate(trace):
            line = '<div class="ti-line"></div>' if i<len(trace)-1 else ""
            trace_html += (f'<div class="trace-item"><div class="ti-left"><div class="ti-dot"></div>{line}</div>'
                           f'<div class="ti-right"><div class="ti-agent">{agent}</div>'
                           f'<div class="ti-fact">✓ {fact}</div><div class="ti-detail">{detail}</div></div></div>')
        biz = expl.get("business_rules_applied",[])
        rules_html = ""
        if biz:
            rules_html = "<br><b style='color:#64748B;font-size:12px;text-transform:uppercase;letter-spacing:.1em'>Business Rules Applied</b>"
            for r in biz: rules_html += f"<div class='ti-detail' style='margin-top:6px'>▹ {r}</div>"
        st.markdown(f'<div class="sec-card">{trace_html}{rules_html}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="site-footer">
  <div class="footer-brand"><span>Decision Engine</span> — MT Provider Selection</div>
  <div class="footer-sub">AI-Powered Translation Intelligence Platform &nbsp;·&nbsp; Built for Hackathon 2026</div>
  <div class="footer-badges">
    <span class="footer-badge">🏢 OpenText L10n Innovation Lab</span>
    <span class="footer-badge">🤖 LLM-as-Judge Evaluation</span>
    <span class="footer-badge">🌍 10 Languages</span>
    <span class="footer-badge">⚡ Real-Time Benchmarking</span>
    <span class="footer-badge">🏆 Hackathon 2026</span>
  </div>
</div>
""", unsafe_allow_html=True)
