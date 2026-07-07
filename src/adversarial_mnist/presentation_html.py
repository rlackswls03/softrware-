"""HTML/CSS/JS template for the presentation deck built by `presentation.py`.

The template is a plain string (not an f-string) so CSS/JS braces need no
escaping; the JSON payload is substituted via a single ``__DATA_JSON__``
placeholder, mirroring the pattern used by ``simulation.py``.
"""

from __future__ import annotations

import json
from typing import Any

_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Adversarial Firewall — 발표 시연</title>
<style>
:root {
  --bg: #0a0d13;
  --bg-panel-grad: radial-gradient(circle at 15% -10%, #131b2c 0%, #0a0d13 55%);
  --surface: #131a26;
  --surface-2: #1a2332;
  --surface-3: #212c3d;
  --border: rgba(255,255,255,0.09);
  --border-strong: rgba(255,255,255,0.18);
  --ink: #eef2f8;
  --ink-secondary: #a9b4c6;
  --ink-muted: #6d7A8f;
  --blue: #3987e5;
  --violet: #9085e9;
  --yellow: #c98500;
  --magenta: #d55181;
  --good: #24c479;
  --good-ink: #0a0d13;
  --good-bg: rgba(36,196,121,0.16);
  --warning: #fab219;
  --warning-bg: rgba(250,178,25,0.16);
  --critical: #ef5b5b;
  --critical-bg: rgba(239,91,91,0.16);
  --serious: #ec835a;
  --serious-bg: rgba(236,131,90,0.16);
  --grid: rgba(255,255,255,0.07);
  --shadow: 0 20px 60px rgba(0,0,0,0.45);
  --mono: "Consolas", "SFMono-Regular", Menlo, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", "Malgun Gothic", sans-serif;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  height: 100%;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
  -webkit-font-smoothing: antialiased;
}
body { overflow: hidden; }
body.no-fullscreen-api .fs-only { display: none; }

::selection { background: rgba(57,135,229,0.4); }

button {
  font-family: inherit;
  cursor: pointer;
}

.deck {
  position: relative;
  width: 100vw;
  height: 100vh;
  background: var(--bg-panel-grad);
}

.topbar {
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: rgba(255,255,255,0.06);
  z-index: 60;
}
.topbar-fill {
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, var(--blue), var(--magenta));
  transition: width 320ms ease;
}

.slide {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  padding: clamp(22px, 3.4vw, 52px) clamp(30px, 6vw, 88px) clamp(70px, 8vh, 96px);
  opacity: 0;
  pointer-events: none;
  transform: translateY(18px) scale(0.99);
  transition: opacity 380ms ease, transform 380ms ease;
  overflow-y: auto;
}
.slide.active { opacity: 1; pointer-events: auto; transform: translateY(0) scale(1); }
.slide::-webkit-scrollbar { width: 8px; }
.slide::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 8px; }

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-bottom: 10px;
}
.eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--blue); }

h1.slide-title {
  margin: 0 0 6px;
  font-size: clamp(26px, 3.1vw, 42px);
  font-weight: 800;
  letter-spacing: -0.01em;
  line-height: 1.15;
}
p.slide-sub {
  margin: 0 0 22px;
  max-width: 920px;
  color: var(--ink-secondary);
  font-size: clamp(14px, 1.15vw, 17px);
  line-height: 1.55;
}

.slide-body { flex: 1; display: flex; flex-direction: column; min-height: 0; }

.cols-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 22px;
  min-height: 0;
}
.cols-3 {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
}
@media (max-width: 1080px) {
  .cols-2, .cols-3 { grid-template-columns: 1fr; }
}

.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px 22px;
  min-width: 0;
}
.panel-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 14px;
}
.panel-title h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 700;
  color: var(--ink);
}
.panel-note {
  margin-top: 12px;
  font-size: 12.5px;
  color: var(--ink-muted);
  line-height: 1.5;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 14px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  font-size: 13px;
  font-weight: 600;
  color: var(--ink-secondary);
}
.chip .dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 auto; }

.chip-row { display: flex; flex-wrap: wrap; gap: 10px; }

.toggle-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
.toggle-btn {
  padding: 8px 16px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--ink-secondary);
  font-size: 13px;
  font-weight: 700;
  transition: all 160ms ease;
}
.toggle-btn:hover { border-color: var(--border-strong); color: var(--ink); }
.toggle-btn.active {
  background: linear-gradient(135deg, var(--blue), #2660ac);
  border-color: transparent;
  color: #fff;
  box-shadow: 0 6px 18px rgba(57,135,229,0.35);
}

.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.kpi-grid.cols-2 { grid-template-columns: repeat(2, 1fr); }
.kpi {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
}
.kpi .kpi-label {
  font-size: 12px;
  color: var(--ink-muted);
  margin-bottom: 6px;
  font-weight: 700;
  letter-spacing: 0.02em;
}
.kpi .kpi-value {
  font-size: clamp(24px, 2.6vw, 34px);
  font-weight: 800;
  font-variant-numeric: tabular-nums;
  line-height: 1.1;
}
.kpi .kpi-sub { font-size: 12px; color: var(--ink-muted); margin-top: 4px; }
.kpi .kpi-arrow { color: var(--ink-muted); font-weight: 400; margin: 0 4px; }

.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 11px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
}
.badge.good { background: var(--good-bg); color: var(--good); }
.badge.warning { background: var(--warning-bg); color: var(--warning); }
.badge.critical { background: var(--critical-bg); color: var(--critical); }
.badge.serious { background: var(--serious-bg); color: var(--serious); }
.badge.neutral { background: var(--surface-3); color: var(--ink-secondary); }

table.fact-table { width: 100%; border-collapse: collapse; font-size: 13px; }
table.fact-table th, table.fact-table td {
  text-align: left;
  padding: 9px 10px;
  border-bottom: 1px solid var(--border);
  color: var(--ink-secondary);
}
table.fact-table th {
  color: var(--ink-muted);
  font-weight: 700;
  font-size: 11.5px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
table.fact-table td.strong { color: var(--ink); font-weight: 700; }
table.fact-table tr:last-child td { border-bottom: none; }

.callout {
  border-left: 3px solid var(--blue);
  background: var(--surface-2);
  border-radius: 0 10px 10px 0;
  padding: 12px 16px;
  font-size: 13.5px;
  color: var(--ink-secondary);
  line-height: 1.55;
}
.callout.critical { border-left-color: var(--critical); }
.callout.warning { border-left-color: var(--warning); }
.callout.good { border-left-color: var(--good); }
.callout strong { color: var(--ink); }

.hero-wrap {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 26px;
  max-width: 1180px;
}
.hero-wrap h1 {
  margin: 0;
  font-size: clamp(34px, 4.6vw, 62px);
  font-weight: 800;
  line-height: 1.16;
  letter-spacing: -0.02em;
}
.hero-wrap h1 .hi { color: var(--blue); }
.hero-question {
  max-width: 760px;
  font-size: clamp(15px, 1.3vw, 19px);
  color: var(--ink-secondary);
  line-height: 1.6;
  border-left: 3px solid var(--border-strong);
  padding-left: 16px;
}
.hero-footer {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 18px;
  color: var(--ink-muted);
  font-size: 13px;
}
.hero-hint {
  position: absolute;
  bottom: 90px;
  right: clamp(30px, 6vw, 88px);
  font-size: 12.5px;
  color: var(--ink-muted);
  display: flex;
  align-items: center;
  gap: 8px;
  animation: pulse-hint 2.4s ease-in-out infinite;
}
@keyframes pulse-hint { 0%, 100% { opacity: 0.55; } 50% { opacity: 1; } }
kbd {
  border: 1px solid var(--border-strong);
  border-bottom-width: 2px;
  border-radius: 6px;
  padding: 2px 7px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--ink-secondary);
  background: var(--surface-2);
}

/* ---- chart chrome ---- */
.chart-wrap { width: 100%; }
svg.chart text { fill: var(--ink-muted); font-family: var(--sans); }
svg.chart .axis-title { fill: var(--ink-secondary); font-size: 11px; font-weight: 700; }
svg.chart .tick-label { font-size: 10.5px; }
svg.chart .gridline { stroke: var(--grid); stroke-width: 1; shape-rendering: crispEdges; }
svg.chart .baseline { stroke: var(--border-strong); stroke-width: 1; }
svg.chart .value-label { fill: var(--ink); font-size: 11px; font-weight: 700; font-variant-numeric: tabular-nums; }

.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; font-size: 12px; color: var(--ink-secondary); }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.legend-swatch { width: 11px; height: 11px; border-radius: 3px; flex: 0 0 auto; }
.legend-swatch.line { height: 3px; border-radius: 2px; width: 16px; }
.legend-shape { width: 11px; height: 11px; display: inline-block; }

#tooltip {
  position: fixed;
  z-index: 200;
  display: none;
  max-width: 260px;
  background: #0d1622;
  border: 1px solid var(--border-strong);
  border-radius: 10px;
  padding: 10px 12px;
  font-size: 12.5px;
  line-height: 1.5;
  color: var(--ink);
  box-shadow: var(--shadow);
  pointer-events: none;
}
#tooltip strong { color: var(--ink); }
#tooltip .tt-muted { color: var(--ink-muted); }

/* ---- nav ---- */
.deck-nav {
  position: fixed;
  bottom: 0; left: 0; right: 0;
  z-index: 60;
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 24px 16px;
  background: linear-gradient(0deg, rgba(10,13,19,0.96) 10%, rgba(10,13,19,0));
}
.nav-btn {
  width: 38px; height: 38px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--ink);
  font-size: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: all 140ms ease;
}
.nav-btn:hover { border-color: var(--border-strong); background: var(--surface-3); }
.nav-dots { display: flex; gap: 7px; flex: 1; justify-content: center; }
.nav-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: rgba(255,255,255,0.18);
  border: none;
  padding: 0;
  transition: all 200ms ease;
}
.nav-dot:hover { background: rgba(255,255,255,0.4); }
.nav-dot.active { width: 22px; border-radius: 5px; background: var(--blue); }
.nav-counter {
  font-size: 12px;
  color: var(--ink-muted);
  font-variant-numeric: tabular-nums;
  min-width: 52px;
  text-align: right;
}
.nav-slidename {
  position: absolute;
  bottom: 58px;
  left: 24px;
  font-size: 11.5px;
  color: var(--ink-muted);
  letter-spacing: 0.02em;
}

/* ---- slide 0 title ---- */
.title-chip-row { display: flex; flex-wrap: wrap; gap: 10px; }

/* ---- slide 5 pipeline ---- */
.pipeline-flow {
  display: flex;
  align-items: stretch;
  gap: 0;
  overflow-x: auto;
  padding: 6px 2px 14px;
}
.pipeline-node {
  flex: 1 1 0;
  min-width: 168px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px;
  position: relative;
}
.pipeline-node .node-icon { font-size: 22px; margin-bottom: 8px; }
.pipeline-node .node-title { font-weight: 800; font-size: 13.5px; margin-bottom: 4px; }
.pipeline-node .node-desc { font-size: 12px; color: var(--ink-muted); line-height: 1.45; }
.pipeline-arrow {
  flex: 0 0 auto;
  width: 34px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--ink-muted);
  font-size: 18px;
}
.decision-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 16px; }
.decision-card {
  border-radius: 12px;
  padding: 14px 16px;
  border: 1px solid var(--border);
  background: var(--surface);
}
.decision-card .dc-title { font-weight: 800; font-size: 13.5px; margin-bottom: 6px; }
.decision-card .dc-rule { font-size: 12px; color: var(--ink-muted); line-height: 1.5; }

/* ---- slide 6 live demo ---- */
.demo-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 22px;
  margin-bottom: 16px;
}
.demo-control-group { display: flex; align-items: center; gap: 8px; }
.demo-control-label { font-size: 11.5px; color: var(--ink-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-right: 4px; }
.sample-thumb {
  width: 44px; height: 44px;
  border-radius: 8px;
  border: 2px solid var(--border);
  background: #05070b;
  padding: 0;
  overflow: hidden;
  display: inline-flex;
  transition: all 140ms ease;
}
.sample-thumb img { width: 100%; height: 100%; image-rendering: pixelated; object-fit: cover; }
.sample-thumb:hover { border-color: var(--border-strong); }
.sample-thumb.active { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(57,135,229,0.25); }
.play-btn {
  padding: 9px 18px;
  border-radius: 10px;
  border: none;
  background: linear-gradient(135deg, var(--blue), #2660ac);
  color: #fff;
  font-weight: 800;
  font-size: 13px;
  box-shadow: 0 6px 18px rgba(57,135,229,0.3);
}
.play-btn.secondary {
  background: var(--surface-2);
  color: var(--ink-secondary);
  box-shadow: none;
  border: 1px solid var(--border);
}

.pipeline-stage-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
}
@media (max-width: 1100px) { .pipeline-stage-grid { grid-template-columns: repeat(2, 1fr); } }
.p-stage {
  background: var(--surface);
  border: 2px solid var(--border);
  border-radius: 14px;
  padding: 14px;
  transition: border-color 200ms ease, transform 200ms ease, box-shadow 200ms ease, opacity 200ms ease;
  opacity: 0.4;
}
.p-stage.revealed { opacity: 1; }
.p-stage.active { border-color: var(--blue); box-shadow: 0 10px 26px rgba(57,135,229,0.22); transform: translateY(-2px); }
.p-stage-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.p-stage-head .stage-num {
  width: 24px; height: 24px;
  border-radius: 50%;
  background: var(--surface-3);
  color: var(--ink-secondary);
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 11.5px; font-weight: 800;
}
.p-stage.active .stage-num { background: var(--blue); color: #fff; }
.p-stage-title { font-size: 12.5px; font-weight: 800; color: var(--ink-secondary); }
.p-stage.active .p-stage-title { color: var(--ink); }
.p-stage-img {
  display: grid;
  place-items: center;
  height: 132px;
  background: #05070b;
  border: 1px solid var(--border);
  border-radius: 10px;
  margin-bottom: 10px;
}
.p-stage-img img { width: 108px; height: 108px; image-rendering: pixelated; }
.p-stage-kv { display: grid; grid-template-columns: 1fr auto; gap: 4px 8px; font-size: 12px; }
.p-stage-kv span { color: var(--ink-muted); }
.p-stage-kv strong { font-variant-numeric: tabular-nums; }

.gauge-track {
  position: relative;
  height: 16px;
  border-radius: 999px;
  background: var(--surface-3);
  overflow: hidden;
  border: 1px solid var(--border);
}
.gauge-fill { height: 100%; width: 0%; background: var(--good); transition: width 260ms ease, background 260ms ease; }
.gauge-marker {
  position: absolute; top: -3px; bottom: -3px;
  width: 2px; background: var(--ink);
  left: 50%;
}
.gauge-marker::after {
  content: "임계값";
  position: absolute;
  top: -18px; left: 50%; transform: translateX(-50%);
  font-size: 9.5px; color: var(--ink-muted); white-space: nowrap;
}

.outcome-banner {
  margin-top: 14px;
  border-radius: 12px;
  padding: 14px 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  border: 1px solid var(--border);
  background: var(--surface);
}
.outcome-banner .ob-main { display: flex; align-items: center; gap: 12px; font-size: 14px; font-weight: 700; }
.outcome-banner .ob-detail { font-size: 12.5px; color: var(--ink-muted); }

.log-console {
  margin-top: 14px;
  background: #05070b;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
  font-family: var(--mono);
  font-size: 12px;
  color: #9fe6b5;
  min-height: 96px;
  white-space: pre-wrap;
  line-height: 1.6;
}

/* ---- misc ---- */
.divider { height: 1px; background: var(--border); margin: 18px 0; }
.two-col-list { columns: 2; column-gap: 28px; font-size: 13.5px; color: var(--ink-secondary); line-height: 1.7; }
.two-col-list li { break-inside: avoid; margin-bottom: 6px; }
.ref-list { font-size: 12px; color: var(--ink-muted); line-height: 1.8; }
.final-statement {
  margin-top: auto;
  padding-top: 18px;
  font-size: clamp(17px, 1.7vw, 22px);
  font-weight: 700;
  color: var(--ink);
  line-height: 1.5;
}
.final-statement .hi { color: var(--blue); }

.help-overlay {
  position: fixed; inset: 0; z-index: 300;
  background: rgba(5,7,11,0.82);
  display: none;
  align-items: center; justify-content: center;
}
.help-overlay.open { display: flex; }
.help-card {
  background: var(--surface);
  border: 1px solid var(--border-strong);
  border-radius: 16px;
  padding: 26px 30px;
  box-shadow: var(--shadow);
  min-width: 300px;
}
.help-card h3 { margin: 0 0 14px; font-size: 16px; }
.help-row { display: flex; justify-content: space-between; gap: 20px; padding: 6px 0; font-size: 13px; color: var(--ink-secondary); }
</style>
</head>
<body>
<div class="deck" id="deck">
  <div class="topbar"><div class="topbar-fill" id="topbar-fill"></div></div>

  <!-- ================= SLIDE 0 : TITLE ================= -->
  <section class="slide" data-index="0" data-name="시작">
    <div class="hero-wrap">
      <div class="eyebrow"><span class="dot"></span>ADVERSARIAL ROBUSTNESS STUDY · MNIST</div>
      <h1>FGSM 적대적 훈련의 한계와<br><span class="hi">Adversarial Firewall</span> 방어 파이프라인</h1>
      <div class="hero-question" id="hero-question"></div>
      <div class="chip-row title-chip-row" id="title-model-chips"></div>
      <div class="hero-footer" id="hero-footer"></div>
    </div>
    <div class="hero-hint"><kbd>&#8594;</kbd> 또는 클릭으로 시작 · <kbd>F</kbd> 전체화면 · <kbd>?</kbd> 단축키</div>
  </section>

  <!-- ================= SLIDE 1 : SETUP ================= -->
  <section class="slide" data-index="1" data-name="실험 설계">
    <div class="eyebrow"><span class="dot"></span>PART 1 · 실험 설계</div>
    <h1 class="slide-title">2개 구조 × 2개 학습 방식, 4개의 모델</h1>
    <p class="slide-sub">LeNet과 SmallCNN을 각각 표준 학습과 FGSM 적대적 훈련으로 학습해 총 4개 모델을 비교한다. 모든 결과는 seed 42·123·2026 세 번의 독립 실행 평균이다.</p>
    <div class="slide-body">
      <div class="cols-2">
        <div class="panel">
          <div class="panel-title"><h3>모델 구조 비교</h3></div>
          <table class="fact-table" id="arch-table"></table>
        </div>
        <div class="panel">
          <div class="panel-title"><h3>모델별 Clean 정확도</h3></div>
          <div class="kpi-grid cols-2" id="setup-kpis"></div>
          <div class="divider"></div>
          <div id="setup-config-list" class="two-col-list"></div>
        </div>
      </div>
    </div>
  </section>

  <!-- ================= SLIDE 2 : FGSM WORKS ================= -->
  <section class="slide" data-index="2" data-name="FGSM 방어 효과">
    <div class="eyebrow"><span class="dot"></span>PART 1 · 핵심 결과 1</div>
    <h1 class="slide-title">FGSM 적대적 훈련, 같은 공격엔 확실히 통한다</h1>
    <p class="slide-sub">정상 정확도를 거의 잃지 않으면서 FGSM ε=0.25 강건 정확도가 크게 향상된다.</p>
    <div class="slide-body">
      <div class="cols-2">
        <div class="panel">
          <div class="panel-title"><h3>Clean vs FGSM ε=0.25 강건 정확도</h3>
            <div class="legend" id="headline-legend"></div>
          </div>
          <div class="chart-wrap" id="chart-headline"></div>
        </div>
        <div class="panel">
          <div class="panel-title">
            <h3>ε 증가에 따른 FGSM 강건 정확도</h3>
            <label class="toggle-btn" id="toggle-anomaly" style="display:inline-flex; align-items:center; gap:6px; cursor:pointer;">
              <input type="checkbox" id="anomaly-checkbox" style="accent-color:#3987e5;"> 이상 현상 표시
            </label>
          </div>
          <div class="chart-wrap" id="chart-fgsm-curve"></div>
          <div class="legend" id="curve-legend"></div>
        </div>
      </div>
      <div class="cols-2" style="margin-top:16px;">
        <div class="callout good" id="retention-callout"></div>
        <div class="callout" id="anomaly-callout">체크박스를 눌러 ε이 커져도 정확도가 오히려 오르는 비단조 지점을 확인하세요.</div>
      </div>
    </div>
  </section>

  <!-- ================= SLIDE 3 : PGD BREAKS IT ================= -->
  <section class="slide" data-index="3" data-name="PGD 불안정성">
    <div class="eyebrow"><span class="dot"></span>PART 1 · 핵심 결과 2 — 반전</div>
    <h1 class="slide-title">그런데 반복 공격 PGD 앞에서는 무너진다</h1>
    <p class="slide-sub">같은 방어 모델이 seed마다 완전히 다른 PGD 강건 정확도를 보인다. 평균만 보면 방어가 통한 것처럼 보이지만, 표준편차가 평균을 압도한다.</p>
    <div class="slide-body">
      <div class="toggle-row">
        <button class="toggle-btn active" data-pgd="pgd10" id="btn-pgd10">PGD-10, 1회 (최초 발견)</button>
        <button class="toggle-btn" data-pgd="pgd20" id="btn-pgd20">PGD-20 · restart×5 (강화 검증)</button>
      </div>
      <div class="cols-2">
        <div class="panel" style="grid-column: span 2;">
          <div class="panel-title"><h3 id="pgd-chart-title">모델별 PGD 강건 정확도 — 평균 · 표준편차 · seed별 값</h3></div>
          <div class="chart-wrap" id="chart-pgd-scatter"></div>
          <div class="legend" id="pgd-legend"></div>
        </div>
      </div>
      <div class="cols-2" style="margin-top:16px;">
        <div class="callout critical" id="pgd-callout-1"></div>
        <div class="callout" id="pgd-callout-2"></div>
      </div>
    </div>
  </section>

  <!-- ================= SLIDE 4 : TRANSFER HEATMAP ================= -->
  <section class="slide" data-index="4" data-name="전이 공격 비대칭">
    <div class="eyebrow"><span class="dot"></span>PART 1 · 핵심 결과 3</div>
    <h1 class="slide-title">모델 간 전이 공격도 대칭적이지 않다</h1>
    <p class="slide-sub">행은 공격을 생성한 source 모델, 열은 공격을 받은 target 모델이다. 대각선은 자기 자신의 gradient로 만든 white-box FGSM이다. (조건부 전이 성공률, ε=0.25 평균)</p>
    <div class="slide-body">
      <div class="cols-2">
        <div class="panel">
          <div class="panel-title"><h3>조건부 전이 성공률 행렬</h3></div>
          <div class="chart-wrap" id="chart-heatmap"></div>
        </div>
        <div class="panel" style="display:flex; flex-direction:column; gap:14px;">
          <div class="panel-title"><h3>읽는 법 &amp; 주목할 지점</h3></div>
          <div class="callout" id="heatmap-callout-1"></div>
          <div class="callout warning" id="heatmap-callout-2"></div>
          <div class="callout" id="heatmap-callout-3"></div>
        </div>
      </div>
    </div>
  </section>

  <!-- ================= SLIDE 5 : PIVOT / ARCHITECTURE ================= -->
  <section class="slide" data-index="5" data-name="Firewall 구조">
    <div class="eyebrow"><span class="dot"></span>PART 2 · 전환</div>
    <h1 class="slide-title">모델을 더 강하게 만드는 것만으로는 부족하다</h1>
    <p class="slide-sub">입력 단계에서 적대적 입력을 탐지 · 정화 · 거부하는 시스템 수준 방어, <strong>Adversarial Firewall</strong>을 추가한다.</p>
    <div class="slide-body">
      <div class="panel">
        <div class="panel-title"><h3>파이프라인 흐름</h3></div>
        <div class="pipeline-flow" id="pipeline-flow"></div>
      </div>
      <div class="panel" style="margin-top:16px;">
        <div class="panel-title"><h3>판정 정책 (Reject Option)</h3></div>
        <div class="decision-grid" id="decision-grid"></div>
      </div>
    </div>
  </section>

  <!-- ================= SLIDE 6 : LIVE DEMO ================= -->
  <section class="slide" data-index="6" data-name="라이브 데모">
    <div class="eyebrow"><span class="dot"></span>PART 2 · 라이브 데모</div>
    <h1 class="slide-title">Adversarial Firewall 파이프라인 실시간 재생</h1>
    <p class="slide-sub" id="demo-sub"></p>
    <div class="slide-body">
      <div class="demo-controls">
        <div class="demo-control-group">
          <span class="demo-control-label">조건</span>
          <div class="toggle-row" id="demo-condition-buttons" style="margin:0;"></div>
        </div>
        <div class="demo-control-group">
          <span class="demo-control-label">샘플</span>
          <div class="chip-row" id="demo-sample-thumbs"></div>
        </div>
        <div class="demo-control-group">
          <button class="play-btn" id="demo-play">▶ 재생</button>
          <button class="play-btn secondary" id="demo-autocycle">자동 순환</button>
        </div>
      </div>
      <div class="pipeline-stage-grid" id="demo-stages"></div>
      <div class="outcome-banner" id="demo-outcome"></div>
      <div class="log-console" id="demo-log"></div>
    </div>
  </section>

  <!-- ================= SLIDE 7 : RECOVERY RESULTS ================= -->
  <section class="slide" data-index="7" data-name="정확도 회복">
    <div class="eyebrow"><span class="dot"></span>PART 2 · 핵심 결과</div>
    <h1 class="slide-title">방화벽 적용 전후, 정확도가 크게 회복된다</h1>
    <p class="slide-sub" id="recovery-sub"></p>
    <div class="slide-body">
      <div class="toggle-row" id="recovery-model-buttons"></div>
      <div class="cols-2">
        <div class="panel">
          <div class="panel-title"><h3>조건별 정확도: 방화벽 적용 전 vs 최종 안전 정확도</h3></div>
          <div class="chart-wrap" id="chart-recovery"></div>
          <div class="legend" id="recovery-legend"></div>
        </div>
        <div class="panel">
          <div class="panel-title"><h3>Reconstruction Error 분포 (탐지 근거)</h3></div>
          <div class="chart-wrap" id="chart-score-hist"></div>
          <div class="legend" id="hist-legend"></div>
        </div>
      </div>
      <div class="kpi-grid" id="recovery-kpis" style="margin-top:16px;"></div>
    </div>
  </section>

  <!-- ================= SLIDE 8 : CONCLUSION ================= -->
  <section class="slide" data-index="8" data-name="결론">
    <div class="eyebrow"><span class="dot"></span>결론</div>
    <h1 class="slide-title">단일 방어로는 부족하다 — 다층 방어가 현실적이다</h1>
    <div class="slide-body">
      <div class="cols-2">
        <div class="panel">
          <div class="panel-title"><h3>Part 1 — 강건성 일반화의 한계</h3></div>
          <p style="font-size:13.5px; color:var(--ink-secondary); line-height:1.6;">FGSM 적대적 훈련은 동일 공격엔 매우 효과적이지만, PGD와 모델 간 전이 공격에서는 안정적으로 일반화되지 않으며 seed에 따른 편차가 크다. 단일 FGSM 공격에 대한 높은 정확도만으로 일반적 강건성을 판단할 수 없다.</p>
        </div>
        <div class="panel">
          <div class="panel-title"><h3>Part 2 — Adversarial Firewall</h3></div>
          <p style="font-size:13.5px; color:var(--ink-secondary); line-height:1.6;">재구성 오차 기반 탐지는 FGSM·PGD를 AUC 1.0 수준으로 탐지했고, 정화·거부 정책이 공격 상황의 최종 안전 정확도를 크게 회복시켰다. 다만 seed 42 단일 실험이며 adaptive attack에 대한 보장은 없다.</p>
        </div>
      </div>
      <div class="panel" style="margin-top:16px;">
        <div class="panel-title"><h3>주요 한계</h3></div>
        <ul class="two-col-list" id="limitations-list"></ul>
      </div>
      <div class="final-statement">모델 수준 방어와 입력 단계 방어를 결합한 <span class="hi">다층 방어 구조</span>가 더 현실적인 접근이다.</div>
      <div class="divider"></div>
      <div class="ref-list" id="references-list"></div>
    </div>
  </section>

</div>

<div class="deck-nav">
  <button class="nav-btn" id="nav-prev" title="이전 (←)">&#8592;</button>
  <div class="nav-dots" id="nav-dots"></div>
  <span class="nav-counter" id="nav-counter">1 / 9</span>
  <button class="nav-btn" id="nav-next" title="다음 (→)">&#8594;</button>
  <button class="nav-btn fs-only" id="nav-fullscreen" title="전체화면 (F)">&#9974;</button>
</div>
<div class="nav-slidename" id="nav-slidename"></div>

<div id="tooltip"></div>

<div class="help-overlay" id="help-overlay">
  <div class="help-card">
    <h3>단축키</h3>
    <div class="help-row"><span>다음 / 이전 슬라이드</span><span><kbd>&#8594;</kbd> <kbd>&#8592;</kbd> <kbd>Space</kbd></span></div>
    <div class="help-row"><span>전체화면 전환</span><span><kbd>F</kbd></span></div>
    <div class="help-row"><span>단축키 닫기</span><span><kbd>Esc</kbd></span></div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
</script>
<script>
__APP_JS__
</script>
</body>
</html>
"""


def render_presentation_html(payload: dict[str, Any]) -> str:
    json_data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return _TEMPLATE.replace("__DATA_JSON__", json_data).replace("__APP_JS__", _APP_JS)


_APP_JS = r"""
/* ============================================================
   State
   ============================================================ */
let currentSlide = 0;
let currentPgdMode = "pgd10";
let recoveryModel = "smallcnn_fgsm_at";
let demoCondition = "Clean";
let demoSampleIndex = 0;
let demoStageTimer = null;
let demoAutoTimer = null;
let demoAutoRunning = false;

const CONDITION_LABELS = { Clean: "정상 입력", FGSM: "FGSM 공격", PGD: "PGD 공격" };
const SEED_SHAPES = ["circle", "diamond", "triangle"];

/* ============================================================
   Utilities
   ============================================================ */
const SVGNS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs) {
  const node = document.createElementNS(SVGNS, tag);
  if (attrs) {
    for (const key in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, key)) node.setAttribute(key, attrs[key]);
    }
  }
  return node;
}

function clearNode(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function fmtPct(value, digits) {
  const d = digits === undefined ? 1 : digits;
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(d)}%`;
}

function fmtPctPoints(value, digits) {
  const d = digits === undefined ? 1 : digits;
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(d)}%p`;
}

function fmtNum(value, digits) {
  const d = digits === undefined ? 2 : digits;
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(d);
}

function modelMeta(modelKey) {
  return DATA.models.find((m) => m.model === modelKey) || null;
}
function modelColor(modelKey) {
  const m = modelMeta(modelKey);
  return m ? m.color : "#8899aa";
}
function modelLabel(modelKey) {
  const m = modelMeta(modelKey);
  return m ? m.label : modelKey;
}
function modelShortLabel(modelKey) {
  const m = modelMeta(modelKey);
  return m ? m.shortLabel : modelKey;
}
function seedShape(seed) {
  const idx = DATA.seeds.indexOf(seed);
  return SEED_SHAPES[idx] || "circle";
}
function kv(label, value) {
  return `<span>${label}</span><strong>${value}</strong>`;
}

/* ---- tooltip ---- */
const tooltipEl = document.getElementById("tooltip");
function showTooltip(evt, html) {
  tooltipEl.innerHTML = html;
  tooltipEl.style.display = "block";
  positionTooltip(evt);
}
function positionTooltip(evt) {
  const pad = 16;
  const rect = tooltipEl.getBoundingClientRect();
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + rect.width > window.innerWidth - 8) x = evt.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = evt.clientY - rect.height - pad;
  tooltipEl.style.left = `${x}px`;
  tooltipEl.style.top = `${y}px`;
}
function hideTooltip() {
  tooltipEl.style.display = "none";
}
function attachTooltip(node, htmlFn) {
  node.addEventListener("mousemove", (evt) => showTooltip(evt, htmlFn()));
  node.addEventListener("mouseleave", hideTooltip);
}

function renderLegend(containerId, items) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = items.map((it) => {
    if (it.shapeSvg) {
      return `<span class="legend-item">${it.shapeSvg}<span>${it.label}</span></span>`;
    }
    const cls = it.shape === "line" ? "legend-swatch line" : "legend-swatch";
    return `<span class="legend-item"><span class="${cls}" style="background:${it.color}"></span>${it.label}</span>`;
  }).join("");
}

function shapeLegendSvg(shape) {
  if (shape === "circle") {
    return '<svg width="12" height="12"><circle cx="6" cy="6" r="4.6" fill="#0a0d13" stroke="#fff" stroke-width="1.4"/></svg>';
  }
  if (shape === "diamond") {
    return '<svg width="12" height="12"><rect x="2.4" y="2.4" width="7.2" height="7.2" fill="#0a0d13" stroke="#fff" stroke-width="1.4" transform="rotate(45 6 6)"/></svg>';
  }
  return '<svg width="12" height="12"><polygon points="6,1.6 10.6,9.6 1.4,9.6" fill="#0a0d13" stroke="#fff" stroke-width="1.4"/></svg>';
}

/* ============================================================
   Chart: grouped bar (percentages, two series per category)
   ============================================================ */
function renderGroupedBar(containerId, opts) {
  const container = document.getElementById(containerId);
  if (!container) return;
  clearNode(container);
  const categories = opts.categories;
  const seriesA = opts.seriesA;
  const seriesB = opts.seriesB;
  const W = opts.width || 560;
  const H = opts.height || 300;
  const margin = { top: 20, right: 12, bottom: 40, left: 46 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;
  const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, style: "width:100%; height:auto; display:block;" });

  const yScale = (v) => margin.top + plotH - Math.max(0, Math.min(1, v)) * plotH;
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const y = yScale(t);
    svg.appendChild(svgEl("line", { class: "gridline", x1: margin.left, x2: margin.left + plotW, y1: y, y2: y }));
    const label = svgEl("text", { class: "tick-label", x: margin.left - 8, y: y + 3, "text-anchor": "end" });
    label.textContent = `${Math.round(t * 100)}%`;
    svg.appendChild(label);
  });
  svg.appendChild(svgEl("line", { class: "baseline", x1: margin.left, x2: margin.left + plotW, y1: margin.top + plotH, y2: margin.top + plotH }));

  const bandWidth = plotW / categories.length;
  const barGap = 7;
  const barWidth = (bandWidth - barGap * 3) / 2;
  const baseY = margin.top + plotH;

  categories.forEach((cat, i) => {
    const bandX = margin.left + i * bandWidth;
    const valueA = seriesA.values[cat.key];
    const valueB = seriesB.values[cat.key];
    const colorA = seriesA.color;
    const colorB = seriesB.colorFn ? seriesB.colorFn(cat.key) : seriesB.color;

    if (valueA !== undefined && valueA !== null) {
      const xA = bandX + barGap;
      const yA = yScale(valueA);
      const rect = svgEl("rect", { x: xA, y: yA, width: barWidth, height: Math.max(0, baseY - yA), fill: colorA, rx: 3, opacity: 0.85 });
      attachTooltip(rect, () => `<strong>${cat.label}</strong><br>${seriesA.label}: ${fmtPct(valueA, 2)}`);
      svg.appendChild(rect);
      const lbl = svgEl("text", { class: "value-label", x: xA + barWidth / 2, y: yA - 6, "text-anchor": "middle" });
      lbl.textContent = fmtPct(valueA, 1);
      svg.appendChild(lbl);
    }
    if (valueB !== undefined && valueB !== null) {
      const xB = bandX + barGap * 2 + barWidth;
      const yB = yScale(valueB);
      const rect = svgEl("rect", { x: xB, y: yB, width: barWidth, height: Math.max(0, baseY - yB), fill: colorB, rx: 3 });
      attachTooltip(rect, () => `<strong>${cat.label}</strong><br>${seriesB.label}: ${fmtPct(valueB, 2)}`);
      svg.appendChild(rect);
      const lbl = svgEl("text", { class: "value-label", x: xB + barWidth / 2, y: yB - 6, "text-anchor": "middle" });
      lbl.textContent = fmtPct(valueB, 1);
      svg.appendChild(lbl);
    }
    const catLabel = svgEl("text", { class: "tick-label", x: bandX + bandWidth / 2, y: baseY + 20, "text-anchor": "middle" });
    catLabel.textContent = cat.label;
    svg.appendChild(catLabel);
  });

  container.appendChild(svg);
}

/* ============================================================
   Chart: multi-series line (epsilon sweep)
   ============================================================ */
function renderLineChart(containerId, opts) {
  const container = document.getElementById(containerId);
  if (!container) return;
  clearNode(container);
  const xValues = opts.xValues;
  const series = opts.series;
  const W = opts.width || 560;
  const H = opts.height || 300;
  const margin = { top: 16, right: 18, bottom: 34, left: 42 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;
  const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, style: "width:100%; height:auto; display:block;" });

  const xMin = xValues[0];
  const xMax = xValues[xValues.length - 1];
  const xScale = (v) => margin.left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
  const yScale = (v) => margin.top + plotH - Math.max(0, Math.min(1, v)) * plotH;

  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const y = yScale(t);
    svg.appendChild(svgEl("line", { class: "gridline", x1: margin.left, x2: margin.left + plotW, y1: y, y2: y }));
    const label = svgEl("text", { class: "tick-label", x: margin.left - 8, y: y + 3, "text-anchor": "end" });
    label.textContent = `${Math.round(t * 100)}%`;
    svg.appendChild(label);
  });
  xValues.forEach((xv) => {
    const x = xScale(xv);
    const label = svgEl("text", { class: "tick-label", x, y: margin.top + plotH + 18, "text-anchor": "middle" });
    label.textContent = xv.toFixed(2);
    svg.appendChild(label);
  });
  svg.appendChild(svgEl("line", { class: "baseline", x1: margin.left, x2: margin.left + plotW, y1: margin.top + plotH, y2: margin.top + plotH }));

  series.forEach((s) => {
    const points = xValues
      .map((xv, i) => ({ x: xScale(xv), y: yScale(s.values[i]), v: s.values[i], eps: xv }))
      .filter((p) => p.v !== null && p.v !== undefined);
    if (!points.length) return;
    const d = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");
    svg.appendChild(svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 2.5 }));
    points.forEach((p) => {
      const dot = svgEl("circle", { cx: p.x, cy: p.y, r: 4, fill: s.color, stroke: "#0a0d13", "stroke-width": 1.5 });
      attachTooltip(dot, () => `<strong>${s.label}</strong><br>ε=${p.eps.toFixed(2)} → ${fmtPct(p.v, 2)}`);
      svg.appendChild(dot);
    });
  });

  if (opts.annotations) {
    opts.annotations.forEach((a) => {
      const x = xScale(a.epsilon);
      const y = yScale(a.value);
      const ring = svgEl("circle", { cx: x, cy: y, r: 8, fill: "none", stroke: "#ffffff", "stroke-width": 2, "stroke-dasharray": "3,2" });
      svg.appendChild(ring);
      const text = svgEl("text", { x, y: y + (a.labelDy || -14), "text-anchor": "middle", class: "value-label" });
      text.style.fill = "#ffffff";
      text.textContent = a.label;
      svg.appendChild(text);
    });
  }

  container.appendChild(svg);
}

/* ============================================================
   Chart: seed scatter (mean bar + std whisker + seed markers)
   ============================================================ */
function renderSeedScatter(containerId, panelData) {
  const container = document.getElementById(containerId);
  if (!container) return;
  clearNode(container);
  const models = DATA.models.map((m) => m.model).filter((m) => panelData[m]);
  // This chart spans the full slide width (not a half-width panel like the
  // other charts), so its viewBox is proportionally wider to keep the
  // rendered aspect ratio (and therefore text/marker size) consistent.
  const W = 1300;
  const H = 56 + models.length * 66;
  const margin = { top: 20, right: 24, bottom: 34, left: 150 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;
  const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, style: "width:100%; height:auto; display:block;" });

  const xScale = (v) => margin.left + Math.max(0, Math.min(1, v)) * plotW;
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const x = xScale(t);
    svg.appendChild(svgEl("line", { class: "gridline", x1: x, x2: x, y1: margin.top, y2: margin.top + plotH }));
    const label = svgEl("text", { class: "tick-label", x, y: margin.top + plotH + 18, "text-anchor": "middle" });
    label.textContent = `${Math.round(t * 100)}%`;
    svg.appendChild(label);
  });

  const rowHeight = plotH / models.length;

  models.forEach((model, idx) => {
    const rowY = margin.top + idx * rowHeight + rowHeight / 2;
    const d = panelData[model];
    const color = modelColor(model);

    const label = svgEl("text", { class: "tick-label", x: margin.left - 12, y: rowY + 4, "text-anchor": "end" });
    label.textContent = modelShortLabel(model);
    label.style.fill = color;
    label.setAttribute("font-weight", "700");
    svg.appendChild(label);

    const lo = Math.max(0, d.mean - d.std);
    const hi = Math.min(1, d.mean + d.std);
    svg.appendChild(svgEl("line", {
      x1: xScale(lo), x2: xScale(hi), y1: rowY, y2: rowY,
      stroke: color, "stroke-width": 4, opacity: 0.3, "stroke-linecap": "round",
    }));
    svg.appendChild(svgEl("line", {
      x1: xScale(d.mean), x2: xScale(d.mean), y1: rowY - 12, y2: rowY + 12,
      stroke: color, "stroke-width": 3,
    }));

    const jitters = [-10, 0, 10];
    d.seeds.forEach((s, i) => {
      const cx = xScale(s.value);
      const cy = rowY + (jitters[i] || 0);
      const shape = seedShape(s.seed);
      let node;
      if (shape === "circle") {
        node = svgEl("circle", { cx, cy, r: 5, fill: "#0a0d13", stroke: "#ffffff", "stroke-width": 1.6 });
      } else if (shape === "diamond") {
        node = svgEl("rect", {
          x: cx - 4.2, y: cy - 4.2, width: 8.4, height: 8.4,
          fill: "#0a0d13", stroke: "#ffffff", "stroke-width": 1.6, transform: `rotate(45 ${cx} ${cy})`,
        });
      } else {
        const r = 5.8;
        node = svgEl("polygon", {
          points: `${cx},${cy - r} ${cx + r},${cy + r * 0.8} ${cx - r},${cy + r * 0.8}`,
          fill: "#0a0d13", stroke: "#ffffff", "stroke-width": 1.6,
        });
      }
      attachTooltip(node, () => `<strong>${modelShortLabel(model)}</strong><br>seed ${s.seed} → ${fmtPct(s.value, 2)}`);
      svg.appendChild(node);
    });

    const meanLbl = svgEl("text", { class: "value-label", x: xScale(d.mean), y: rowY - 20, "text-anchor": "middle" });
    meanLbl.textContent = `평균 ${fmtPct(d.mean, 1)}`;
    svg.appendChild(meanLbl);
  });

  svg.appendChild(svgEl("line", { class: "baseline", x1: margin.left, x2: margin.left + plotW, y1: margin.top + plotH, y2: margin.top + plotH }));
  container.appendChild(svg);
}

/* ============================================================
   Chart: transferability heatmap
   ============================================================ */
function seqColor(v) {
  const stops = [[0.0, [16, 22, 33]], [0.5, [57, 135, 229]], [1.0, [235, 244, 253]]];
  let a = stops[0];
  let b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i += 1) {
    if (v >= stops[i][0] && v <= stops[i + 1][0]) {
      a = stops[i];
      b = stops[i + 1];
      break;
    }
  }
  const t = (v - a[0]) / ((b[0] - a[0]) || 1);
  const mix = (c1, c2) => Math.round(c1 + (c2 - c1) * t);
  return `rgb(${mix(a[1][0], b[1][0])},${mix(a[1][1], b[1][1])},${mix(a[1][2], b[1][2])})`;
}
function textColorFor(v) {
  return v > 0.62 ? "#0a0d13" : "#eef2f8";
}

function renderHeatmap(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  clearNode(container);
  const t = DATA.transfer;
  const n = t.models.length;
  const W = 560;
  const H = 560;
  const margin = { top: 46, right: 14, bottom: 14, left: 118 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;
  const cellW = plotW / n;
  const cellH = plotH / n;
  const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, style: "width:100%; height:auto; display:block;" });

  const colHead = svgEl("text", { class: "axis-title", x: margin.left + plotW / 2, y: 14, "text-anchor": "middle" });
  colHead.textContent = "Target (공격을 받은 모델)";
  svg.appendChild(colHead);

  t.labels.forEach((label, j) => {
    const x = margin.left + j * cellW + cellW / 2;
    const text = svgEl("text", { class: "tick-label", x, y: margin.top - 12, "text-anchor": "middle" });
    text.textContent = label;
    text.style.fill = modelColor(t.models[j]);
    text.setAttribute("font-weight", "700");
    svg.appendChild(text);
  });

  t.models.forEach((source, i) => {
    const y = margin.top + i * cellH + cellH / 2;
    const text = svgEl("text", { class: "tick-label", x: margin.left - 10, y: y + 4, "text-anchor": "end" });
    text.textContent = t.labels[i];
    text.style.fill = modelColor(source);
    text.setAttribute("font-weight", "700");
    svg.appendChild(text);

    t.models.forEach((target, j) => {
      const value = t.matrix[i][j];
      const x = margin.left + j * cellW;
      const yy = margin.top + i * cellH;
      const gap = 2;
      const rect = svgEl("rect", {
        x: x + gap / 2, y: yy + gap / 2, width: cellW - gap, height: cellH - gap,
        rx: 6, fill: value === null ? "#131a26" : seqColor(value),
      });
      if (i === j) {
        rect.setAttribute("stroke", "#ffffff");
        rect.setAttribute("stroke-width", "2");
      }
      attachTooltip(rect, () => `<strong>${t.labels[i]}</strong> → <strong>${t.labels[j]}</strong>${i === j ? ' <span class="tt-muted">(white-box)</span>' : ""}<br>조건부 전이 성공률: ${fmtPct(value, 2)}`);
      svg.appendChild(rect);
      if (value !== null) {
        const label = svgEl("text", { class: "value-label", x: x + cellW / 2, y: yy + cellH / 2 + 4, "text-anchor": "middle" });
        label.textContent = `${(value * 100).toFixed(0)}%`;
        label.style.fill = textColorFor(value);
        svg.appendChild(label);
      }
    });
  });

  container.appendChild(svg);
}

/* ============================================================
   Chart: reconstruction-error histogram
   ============================================================ */
function renderScoreHistogram(containerId, model) {
  const container = document.getElementById(containerId);
  if (!container) return;
  clearNode(container);
  const hist = DATA.firewall.scoreHist;
  const edges = hist.binEdges;
  const threshold = hist.threshold;
  const conditions = [
    { key: "Clean", stroke: "#24c479" },
    { key: "FGSM", stroke: "#ec835a" },
    { key: "PGD", stroke: "#ef5b5b" },
  ];
  const series = hist.series[model] || {};
  let maxCount = 1;
  conditions.forEach((c) => {
    const counts = series[c.key] || [];
    counts.forEach((v) => { if (v > maxCount) maxCount = v; });
  });

  const W = 560;
  const H = 300;
  const margin = { top: 16, right: 14, bottom: 34, left: 42 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;
  const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, style: "width:100%; height:auto; display:block;" });

  const xMin = edges[0];
  const xMax = edges[edges.length - 1];
  const xScale = (v) => margin.left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
  const yScale = (v) => margin.top + plotH - (v / maxCount) * plotH;

  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const y = margin.top + plotH - t * plotH;
    svg.appendChild(svgEl("line", { class: "gridline", x1: margin.left, x2: margin.left + plotW, y1: y, y2: y }));
  });
  [xMin, (xMin + xMax) / 2, xMax].forEach((xv) => {
    const x = xScale(xv);
    const label = svgEl("text", { class: "tick-label", x, y: margin.top + plotH + 18, "text-anchor": "middle" });
    label.textContent = xv.toFixed(3);
    svg.appendChild(label);
  });
  svg.appendChild(svgEl("line", { class: "baseline", x1: margin.left, x2: margin.left + plotW, y1: margin.top + plotH, y2: margin.top + plotH }));

  conditions.forEach((c) => {
    const counts = series[c.key];
    if (!counts) return;
    const baseY = margin.top + plotH;
    let areaD = `M${xScale(edges[0]).toFixed(2)},${baseY.toFixed(2)}`;
    let lineD = "";
    counts.forEach((count, i) => {
      const xLeft = xScale(edges[i]);
      const xRight = xScale(edges[i + 1]);
      const y = yScale(count);
      areaD += ` L${xLeft.toFixed(2)},${y.toFixed(2)} L${xRight.toFixed(2)},${y.toFixed(2)}`;
      lineD += `${i === 0 ? "M" : "L"}${xLeft.toFixed(2)},${y.toFixed(2)} L${xRight.toFixed(2)},${y.toFixed(2)} `;
    });
    areaD += ` L${xScale(edges[edges.length - 1]).toFixed(2)},${baseY.toFixed(2)} Z`;
    svg.appendChild(svgEl("path", { d: areaD, fill: c.stroke, opacity: 0.22, stroke: "none" }));
    svg.appendChild(svgEl("path", { d: lineD, fill: "none", stroke: c.stroke, "stroke-width": 1.8 }));
  });

  const tx = xScale(threshold);
  svg.appendChild(svgEl("line", { x1: tx, x2: tx, y1: margin.top, y2: margin.top + plotH, stroke: "#ffffff", "stroke-width": 1.5, "stroke-dasharray": "4,3" }));
  const tlabel = svgEl("text", { class: "value-label", x: tx, y: margin.top - 4, "text-anchor": "middle" });
  tlabel.textContent = "임계값";
  tlabel.style.fill = "#ffffff";
  svg.appendChild(tlabel);

  container.appendChild(svg);
}

/* ============================================================
   Gauge (live demo)
   ============================================================ */
function setGauge(gaugeFillEl, gaugeMarkerEl, score, threshold) {
  const scaleMax = Math.max(threshold * 2.2, score * 1.05, 0.0001);
  const pct = Math.min(100, (score / scaleMax) * 100);
  const markerPct = Math.min(100, (threshold / scaleMax) * 100);
  gaugeFillEl.style.width = `${pct}%`;
  gaugeMarkerEl.style.left = `${markerPct}%`;
  gaugeFillEl.style.background = score > threshold ? "var(--critical)" : "var(--good)";
}

/* ============================================================
   Slide 0 — title
   ============================================================ */
function renderTitle() {
  document.getElementById("hero-question").textContent =
    '"FGSM 적대적 훈련으로 향상된 강건성이 다른 구조로의 전이 공격과 더 강한 반복 공격(PGD)에도 유지되는가?"';
  document.getElementById("title-model-chips").innerHTML = DATA.models.map((m) => `
    <span class="chip"><span class="dot" style="background:${m.color}"></span>${m.shortLabel} · Clean ${fmtPct(m.cleanAccuracy, 2)}</span>
  `).join("");
  document.getElementById("hero-footer").innerHTML = `
    <span>Dataset ${DATA.experimentConfig.dataset}</span>
    <span>Seeds ${DATA.seeds.join(" · ")}</span>
    <span>Part 1 강건성 분석 + Part 2 Adversarial Firewall</span>
  `;
}

/* ============================================================
   Slide 1 — setup
   ============================================================ */
function renderSetup() {
  const archs = DATA.architectures;
  const rows = [
    ["구조", archs.lenet.label, archs.smallcnn.label],
    ["합성곱 레이어 수", archs.lenet.convLayers, archs.smallcnn.convLayers],
    ["필터 수", archs.lenet.filters, archs.smallcnn.filters],
    ["커널 크기", archs.lenet.kernel, archs.smallcnn.kernel],
    ["풀링", archs.lenet.pooling, archs.smallcnn.pooling],
    ["BatchNorm", archs.lenet.batchNorm, archs.smallcnn.batchNorm],
    ["Dropout", archs.lenet.dropout, archs.smallcnn.dropout],
    ["파라미터 수", `${archs.lenet.paramCount.toLocaleString()}개`, `${archs.smallcnn.paramCount.toLocaleString()}개`],
  ];
  document.getElementById("arch-table").innerHTML = `
    <tr><th>항목</th><th>LeNet</th><th>SmallCNN</th></tr>
    ${rows.map((r) => `<tr><td>${r[0]}</td><td class="strong">${r[1]}</td><td class="strong">${r[2]}</td></tr>`).join("")}
  `;

  document.getElementById("setup-kpis").innerHTML = DATA.models.map((m) => `
    <div class="kpi">
      <div class="kpi-label" style="color:${m.color}">${m.shortLabel}</div>
      <div class="kpi-value">${fmtPct(m.cleanAccuracy, 2)}</div>
    </div>
  `).join("");

  const cfg = DATA.experimentConfig;
  const eps = DATA.fgsmCurve.epsilons.map((e) => e.toFixed(2)).join(", ");
  const items = [
    `Dataset: ${cfg.dataset}`,
    `Seeds: ${DATA.seeds.join(", ")}`,
    `FGSM 평가 ε: ${eps}`,
    `FGSM 훈련 ε: ${cfg.fgsmAtEpsilon.toFixed(2)}`,
    `PGD-10 평가: ε=${cfg.pgd10.epsilon.toFixed(2)}, steps=${cfg.pgd10.steps}, n=${cfg.pgd10.evaluatedSamples.toLocaleString()}`,
    `PGD-20 restart×5: ε=${cfg.pgd20.epsilon.toFixed(2)}, steps=${cfg.pgd20.steps}, restarts=${cfg.pgd20.restarts}, n=${cfg.pgd20.evaluatedSamples.toLocaleString()}`,
  ];
  document.getElementById("setup-config-list").innerHTML = `<ul style="margin:0;padding-left:18px;">${items.map((it) => `<li>${it}</li>`).join("")}</ul>`;
}

/* ============================================================
   Slide 2 — FGSM works
   ============================================================ */
function renderFgsmWorks() {
  const epsIndex25 = DATA.fgsmCurve.epsilons.findIndex((e) => Math.abs(e - 0.25) < 1e-9);
  const seriesA = { label: "Clean", color: "#5b6576", values: {} };
  const seriesB = { label: "FGSM ε=0.25", colorFn: modelColor, values: {} };
  DATA.models.forEach((m) => {
    seriesA.values[m.model] = m.cleanAccuracy;
    seriesB.values[m.model] = DATA.fgsmCurve.series[m.model][epsIndex25];
  });
  renderGroupedBar("chart-headline", {
    categories: DATA.models.map((m) => ({ key: m.model, label: m.shortLabel })),
    seriesA,
    seriesB,
    height: 300,
  });
  renderLegend("headline-legend", [
    { label: "Clean", color: "#5b6576" },
    { label: "FGSM ε=0.25", color: "#3987e5" },
  ]);

  renderFgsmCurve(false);
  document.getElementById("anomaly-checkbox").addEventListener("change", (evt) => renderFgsmCurve(evt.target.checked));

  const retentions = DATA.cleanRetention
    .map((r) => `${r.architecture === "lenet" ? "LeNet" : "SmallCNN"} ${fmtNum(r.retentionPct, 2)}%`)
    .join(" · ");
  document.getElementById("retention-callout").innerHTML =
    `<strong>Clean 정확도 유지율</strong> ${retentions} — FGSM 훈련이 정상 성능을 거의 희생시키지 않는다.`;
}

function renderFgsmCurve(showAnomaly) {
  const curve = DATA.fgsmCurve;
  const series = DATA.models.map((m) => ({ key: m.model, label: m.shortLabel, color: m.color, values: curve.series[m.model] }));

  const a1 = DATA.fgsmNonMonotonic.lenet_fgsm_at_seed42;
  const a2 = DATA.fgsmNonMonotonic.smallcnn_fgsm_at_seed2026;
  const findEps = (arr, eps) => arr.find((r) => Math.abs(r.epsilon - eps) < 1e-9);
  const l1a = findEps(a1, 0.05);
  const l1b = findEps(a1, 0.2);
  const s2clean = findEps(a2, 0.0);
  const s2b = findEps(a2, 0.2);

  const annotations = [];
  if (showAnomaly) {
    if (l1a) annotations.push({ epsilon: l1a.epsilon, value: l1a.robustAccuracy, label: "LeNet-AT seed42 저점", labelDy: 16 });
    if (l1b) annotations.push({ epsilon: l1b.epsilon, value: l1b.robustAccuracy, label: "ε0.20에서 반등", labelDy: -14 });
    if (s2b) annotations.push({ epsilon: s2b.epsilon, value: s2b.robustAccuracy, label: "SmallCNN-AT seed2026: clean 초과", labelDy: -14 });
  }

  renderLineChart("chart-fgsm-curve", { xValues: curve.epsilons, series, annotations, height: 300 });
  renderLegend("curve-legend", DATA.models.map((m) => ({ label: m.shortLabel, color: m.color, shape: "line" })));

  const calloutEl = document.getElementById("anomaly-callout");
  if (showAnomaly && l1a && l1b && s2clean && s2b) {
    calloutEl.innerHTML = `<strong>비단조 현상</strong> lenet_fgsm_at seed42는 ε=${l1a.epsilon.toFixed(2)}→${l1b.epsilon.toFixed(2)} 구간에서 ${fmtPct(l1a.robustAccuracy, 1)} → ${fmtPct(l1b.robustAccuracy, 1)}로 오히려 상승한다. smallcnn_fgsm_at seed2026은 ε=${s2b.epsilon.toFixed(2)}에서 ${fmtPct(s2b.robustAccuracy, 2)}로, 같은 seed의 clean 정확도 ${fmtPct(s2clean.robustAccuracy, 2)}보다 높다.`;
  } else {
    calloutEl.textContent = "체크박스를 눌러 ε이 커져도 정확도가 오히려 오르는 비단조 지점을 확인하세요.";
  }
}

/* ============================================================
   Slide 3 — PGD instability
   ============================================================ */
function renderPgdInstability() {
  document.getElementById("btn-pgd10").addEventListener("click", () => setPgdMode("pgd10"));
  document.getElementById("btn-pgd20").addEventListener("click", () => setPgdMode("pgd20"));
  setPgdMode("pgd10");
}

function setPgdMode(mode) {
  currentPgdMode = mode;
  document.getElementById("btn-pgd10").classList.toggle("active", mode === "pgd10");
  document.getElementById("btn-pgd20").classList.toggle("active", mode === "pgd20");
  const panel = DATA.pgd[mode];
  const cfg = mode === "pgd10" ? DATA.experimentConfig.pgd10 : DATA.experimentConfig.pgd20;
  document.getElementById("pgd-chart-title").textContent = mode === "pgd10"
    ? `PGD-10, 1회 white-box (n=${cfg.evaluatedSamples.toLocaleString()})`
    : `PGD-20 · restart×${cfg.restarts} white-box (n=${cfg.evaluatedSamples.toLocaleString()})`;

  renderSeedScatter("chart-pgd-scatter", panel);
  renderLegend("pgd-legend", DATA.seeds.map((seed) => ({ label: `seed ${seed}`, shapeSvg: shapeLegendSvg(seedShape(seed)) })));

  const fgsmAtModels = DATA.models.filter((m) => m.training === "fgsm_at").map((m) => m.model);
  let worst = null;
  fgsmAtModels.forEach((model) => {
    if (!worst || panel[model].std > panel[worst].std) worst = model;
  });
  if (worst) {
    const d = panel[worst];
    const seedText = d.seeds.map((s) => `seed${s.seed} ${fmtPct(s.value, 2)}`).join(", ");
    document.getElementById("pgd-callout-1").innerHTML =
      `<strong>${modelShortLabel(worst)}</strong>: ${seedText} — 평균 ${fmtPct(d.mean, 2)} ± ${fmtPctPoints(d.std, 2)}${d.std > d.mean ? " · 표준편차가 평균보다 큼" : ""}`;
  }
  const standardModels = DATA.models.filter((m) => m.training === "standard").map((m) => m.model);
  const stdMeans = standardModels.map((m) => `${modelShortLabel(m)} ${fmtPct(panel[m].mean, 2)}`).join(" · ");
  document.getElementById("pgd-callout-2").innerHTML =
    `<strong>표준 학습 모델 참고값</strong> ${stdMeans} — FGSM 훈련 없이도 PGD 앞에서는 거의 항상 붕괴한다.`;
}

/* ============================================================
   Slide 4 — transfer heatmap
   ============================================================ */
function renderTransferHeatmap() {
  renderHeatmap("chart-heatmap");
  const t = DATA.transfer;
  const idxLenetStd = t.models.indexOf("lenet_standard");
  const idxSmallStd = t.models.indexOf("smallcnn_standard");
  const idxSmallAt = t.models.indexOf("smallcnn_fgsm_at");

  document.getElementById("heatmap-callout-1").innerHTML =
    `<strong>대각선(White-box)</strong> 표준 학습 모델은 자기 gradient로 만든 공격에 매우 취약하다 (${t.labels[idxLenetStd]} ${fmtPct(t.matrix[idxLenetStd][idxLenetStd], 1)}, ${t.labels[idxSmallStd]} ${fmtPct(t.matrix[idxSmallStd][idxSmallStd], 1)}).`;

  const selfAttack = t.matrix[idxSmallAt][idxSmallAt];
  const incomingFromLenetStd = t.matrix[idxLenetStd][idxSmallAt];
  document.getElementById("heatmap-callout-2").innerHTML =
    `<strong>${t.labels[idxSmallAt]}의 비대칭성</strong> 자기 자신의 FGSM에는 ${fmtPct(selfAttack, 2)}로 강하지만, ${t.labels[idxLenetStd]}가 만든 공격에는 ${fmtPct(incomingFromLenetStd, 2)}까지 취약해진다.`;

  document.getElementById("heatmap-callout-3").textContent =
    "자기 gradient에 강한 것이 실제 강건성이 아니라 gradient masking의 신호일 수 있다 — obfuscated gradients 논의와 연결된다.";
}

/* ============================================================
   Slide 5 — pivot / pipeline architecture
   ============================================================ */
function renderPivot() {
  const ex = DATA.firewall.examples;
  const pipeline = [
    { icon: "🖼️", title: "입력 이미지", desc: "원본 또는 공격받은 MNIST 이미지" },
    { icon: "🧠", title: "분류기 예측", desc: "SmallCNN 분류기가 1차 예측 수행" },
    { icon: "🔁", title: "Autoencoder 정화", desc: "Convolutional autoencoder가 입력을 복원, 고주파 노이즈 감쇠" },
    { icon: "📊", title: "재구성 오차 탐지", desc: `임계값 ${fmtNum(ex.threshold, 5)} (clean 검증 95th percentile)` },
    { icon: "🚦", title: "판정", desc: "원본 통과 · 정화 후 통과 · 위험 거부 중 하나" },
  ];
  document.getElementById("pipeline-flow").innerHTML = pipeline.map((node, i) => `
    <div class="pipeline-node">
      <div class="node-icon">${node.icon}</div>
      <div class="node-title">${node.title}</div>
      <div class="node-desc">${node.desc}</div>
    </div>
    ${i < pipeline.length - 1 ? '<div class="pipeline-arrow">→</div>' : ""}
  `).join("");

  const cards = [
    { title: "ACCEPT_ORIGINAL", badge: "good", rule: `재구성 오차 &lt; 임계값(${fmtNum(ex.threshold, 5)}) → 원본을 그대로 분류` },
    { title: "ACCEPT_PURIFIED", badge: "warning", rule: `오차 ≥ 임계값 이고 정화 후 신뢰도 ≥ ${fmtNum(ex.minConfidence, 2)} → 정화된 이미지로 분류` },
    { title: "REJECT_SUSPICIOUS", badge: "critical", rule: `오차 ≥ 임계값 이고 정화 후 신뢰도 &lt; ${fmtNum(ex.minConfidence, 2)} → 판단 거부, 인간 검토` },
  ];
  document.getElementById("decision-grid").innerHTML = cards.map((c) => `
    <div class="decision-card">
      <span class="badge ${c.badge}">${c.title}</span>
      <div class="dc-rule" style="margin-top:8px;">${c.rule}</div>
    </div>
  `).join("");
}

/* ============================================================
   Slide 6 — live demo
   ============================================================ */
function initLiveDemo() {
  const ex = DATA.firewall.examples;
  document.getElementById("demo-sub").textContent =
    `예시 이미지는 firewall_examples.pt에 저장된 실제 MNIST 샘플입니다 (model=${ex.model}, seed=${ex.seed}, ε=${ex.epsilon.toFixed(2)}).`;

  const conditions = Object.keys(ex.conditions);
  document.getElementById("demo-condition-buttons").innerHTML = conditions.map((c) => `
    <button class="toggle-btn" data-cond="${c}">${ex.conditions[c].label}</button>
  `).join("");
  document.querySelectorAll("#demo-condition-buttons .toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => setDemoCondition(btn.dataset.cond));
  });

  buildStageCards();
  setDemoCondition(conditions[0]);

  document.getElementById("demo-play").addEventListener("click", playDemoStages);
  document.getElementById("demo-autocycle").addEventListener("click", toggleAutoCycle);
}

function buildStageCards() {
  const stageTitles = ["원본 (Original)", "공격 입력 (Attack/Input)", "정화 (Purifier)", "탐지 · 판정 (Detector)"];
  document.getElementById("demo-stages").innerHTML = stageTitles.map((title, i) => {
    if (i < 3) {
      return `
        <div class="p-stage" data-stage="${i}">
          <div class="p-stage-head"><span class="p-stage-title">${title}</span><span class="stage-num">${i + 1}</span></div>
          <div class="p-stage-img"><img id="demo-img-${i}" alt="${title}"></div>
          <div class="p-stage-kv" id="demo-kv-${i}"></div>
        </div>
      `;
    }
    return `
      <div class="p-stage" data-stage="3">
        <div class="p-stage-head"><span class="p-stage-title">${title}</span><span class="stage-num">4</span></div>
        <div class="gauge-track"><div class="gauge-fill" id="demo-gauge-fill"></div><div class="gauge-marker" id="demo-gauge-marker"></div></div>
        <div class="p-stage-kv" id="demo-kv-3" style="margin-top:20px;"></div>
      </div>
    `;
  }).join("");
}

function currentSample() {
  const ex = DATA.firewall.examples;
  return ex.conditions[demoCondition].samples[demoSampleIndex];
}

function setDemoCondition(cond) {
  demoCondition = cond;
  demoSampleIndex = 0;
  document.querySelectorAll("#demo-condition-buttons .toggle-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.cond === cond);
  });
  buildSampleThumbs();
  renderDemoSample();
  resetStages();
}

function buildSampleThumbs() {
  const ex = DATA.firewall.examples;
  const samples = ex.conditions[demoCondition].samples;
  const wrap = document.getElementById("demo-sample-thumbs");
  wrap.innerHTML = samples.map((s, i) => `
    <button class="sample-thumb${i === demoSampleIndex ? " active" : ""}" data-idx="${i}"><img src="${s.images.input}" alt="sample ${i}"></button>
  `).join("");
  wrap.querySelectorAll(".sample-thumb").forEach((btn) => {
    btn.addEventListener("click", () => {
      demoSampleIndex = Number(btn.dataset.idx);
      wrap.querySelectorAll(".sample-thumb").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderDemoSample();
      resetStages();
    });
  });
}

function renderDemoSample() {
  const s = currentSample();
  document.getElementById("demo-img-0").src = s.images.original;
  document.getElementById("demo-img-1").src = s.images.input;
  document.getElementById("demo-img-2").src = s.images.purified;
  document.getElementById("demo-kv-0").innerHTML = kv("정답", s.label) + kv("Clean 예측", s.cleanPrediction);
  document.getElementById("demo-kv-1").innerHTML =
    kv("입력 예측", s.inputPrediction) + kv("L∞ 변화", fmtNum(s.perturbationLinf, 3)) + kv("예측 변화", s.attackChangedPrediction ? "예" : "아니오");
  document.getElementById("demo-kv-2").innerHTML =
    kv("정화 후 예측", s.purifiedPrediction) + kv("평균 절대 변화", fmtNum(s.purificationMeanAbs, 4));
  document.getElementById("demo-kv-3").innerHTML =
    kv("재구성 오차", fmtNum(s.score, 5)) + kv("임계값", fmtNum(s.threshold, 5)) + kv("탐지 여부", s.detected ? "예" : "아니오") +
    kv("판정", s.decisionLabel) + kv("최종 예측", s.finalPrediction === null ? "거부" : s.finalPrediction);

  setGauge(document.getElementById("demo-gauge-fill"), document.getElementById("demo-gauge-marker"), s.score, s.threshold);

  const badgeClass = s.decision === "ACCEPT_ORIGINAL" ? "good" : (s.decision === "ACCEPT_PURIFIED" ? "warning" : "critical");
  document.getElementById("demo-outcome").innerHTML = `
    <div class="ob-main"><span class="badge ${badgeClass}">${s.decisionLabel}</span><span>최종 예측: ${s.finalPrediction === null ? "거부 (인간 검토)" : s.finalPrediction}</span></div>
    <div class="ob-detail">${s.correctAfterFirewall ? "정답과 일치" : "정답과 불일치 또는 거부"}</div>
  `;

  renderDemoLog(-1);
}

function resetStages() {
  window.clearTimeout(demoStageTimer);
  document.querySelectorAll("#demo-stages .p-stage").forEach((el, i) => {
    el.classList.remove("active");
    el.classList.toggle("revealed", i === 0);
  });
  renderDemoLog(0);
}

function playDemoStages() {
  window.clearTimeout(demoStageTimer);
  const stages = document.querySelectorAll("#demo-stages .p-stage");
  let step = 0;
  function tick() {
    stages.forEach((el, i) => {
      el.classList.toggle("active", i === step);
      el.classList.toggle("revealed", i <= step);
    });
    renderDemoLog(step);
    step += 1;
    if (step < stages.length) demoStageTimer = window.setTimeout(tick, 700);
  }
  tick();
}

function renderDemoLog(step) {
  const s = currentSample();
  const lines = [`[0] condition=${s.condition} sample=${s.index} true_label=${s.label}`];
  if (step >= 0) lines.push(`[1] clean_prediction=${s.cleanPrediction}`);
  if (step >= 1) lines.push(`[2] input_prediction=${s.inputPrediction} l_inf=${fmtNum(s.perturbationLinf, 4)}`);
  if (step >= 2) lines.push(`[3] purified_prediction=${s.purifiedPrediction} mean_abs_update=${fmtNum(s.purificationMeanAbs, 4)}`);
  if (step >= 3) {
    lines.push(`[4] reconstruction_error=${fmtNum(s.score, 6)} threshold=${fmtNum(s.threshold, 6)} detected=${s.detected}`);
    lines.push(`[5] decision=${s.decision} final=${s.finalPrediction === null ? "reject" : s.finalPrediction}`);
  }
  document.getElementById("demo-log").textContent = lines.join("\n");
}

function toggleAutoCycle() {
  demoAutoRunning = !demoAutoRunning;
  const btn = document.getElementById("demo-autocycle");
  btn.textContent = demoAutoRunning ? "⏸ 자동 순환 중지" : "자동 순환";
  if (demoAutoRunning) {
    autoCycleStep();
  } else {
    window.clearTimeout(demoAutoTimer);
  }
}

function autoCycleStep() {
  const ex = DATA.firewall.examples;
  const samples = ex.conditions[demoCondition].samples;
  demoSampleIndex = (demoSampleIndex + 1) % samples.length;
  document.querySelectorAll("#demo-sample-thumbs .sample-thumb").forEach((b, i) => b.classList.toggle("active", i === demoSampleIndex));
  renderDemoSample();
  resetStages();
  playDemoStages();
  demoAutoTimer = window.setTimeout(autoCycleStep, 3600);
}

/* ============================================================
   Slide 7 — recovery results
   ============================================================ */
function renderRecoverySlide() {
  const models = ["smallcnn_standard", "smallcnn_fgsm_at"].filter((m) => DATA.firewall.results[m]);
  document.getElementById("recovery-model-buttons").innerHTML = models.map((m) => `
    <button class="toggle-btn" data-model="${m}">${modelShortLabel(m)}</button>
  `).join("");
  document.querySelectorAll("#recovery-model-buttons .toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => setRecoveryModel(btn.dataset.model));
  });
  setRecoveryModel(models.includes(recoveryModel) ? recoveryModel : models[0]);
}

function setRecoveryModel(model) {
  recoveryModel = model;
  document.querySelectorAll("#recovery-model-buttons .toggle-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.model === model);
  });
  document.getElementById("recovery-sub").textContent =
    `${modelLabel(model)} 기준, 방화벽 적용 전(원본 정확도)과 최종 안전 정확도(정화+거부 반영)를 비교한다.`;

  const results = DATA.firewall.results[model];
  const conditions = ["Clean", "FGSM", "PGD"];
  const seriesA = { label: "적용 전", color: "#5b6576", values: {} };
  const seriesB = { label: "최종 안전 정확도", color: "#24c479", values: {} };
  conditions.forEach((c) => {
    seriesA.values[c] = results[c].original_accuracy;
    seriesB.values[c] = results[c].final_safe_accuracy;
  });
  renderGroupedBar("chart-recovery", {
    categories: conditions.map((c) => ({ key: c, label: CONDITION_LABELS[c] })),
    seriesA,
    seriesB,
    height: 300,
  });
  renderLegend("recovery-legend", [
    { label: "적용 전", color: "#5b6576" },
    { label: "최종 안전 정확도", color: "#24c479" },
  ]);

  renderScoreHistogram("chart-score-hist", model);
  renderLegend("hist-legend", [
    { label: "정상 입력", color: "#24c479" },
    { label: "FGSM 공격", color: "#ec835a" },
    { label: "PGD 공격", color: "#ef5b5b" },
  ]);

  const detection = DATA.firewall.detection[model];
  const kpis = [
    { label: "PGD: 적용 전 → 최종", value: `${fmtPct(results.PGD.original_accuracy, 2)} → ${fmtPct(results.PGD.final_safe_accuracy, 2)}` },
    { label: "FGSM: 적용 전 → 최종", value: `${fmtPct(results.FGSM.original_accuracy, 2)} → ${fmtPct(results.FGSM.final_safe_accuracy, 2)}` },
    { label: "탐지 AUC (FGSM / PGD)", value: `${fmtNum(detection.FGSM.auc, 3)} / ${fmtNum(detection.PGD.auc, 3)}` },
    { label: "TPR@FPR5% (전체 공격)", value: fmtPct(detection.ALL_ATTACKS.tprAtFpr5, 2) },
  ];
  document.getElementById("recovery-kpis").innerHTML = kpis.map((k) => `
    <div class="kpi"><div class="kpi-label">${k.label}</div><div class="kpi-value" style="font-size:24px;">${k.value}</div></div>
  `).join("");
}

/* ============================================================
   Slide 8 — conclusion
   ============================================================ */
function renderConclusion() {
  const smallcnnModels = DATA.models.filter((m) => m.architecture === "smallcnn").map((m) => m.training);
  const limitations = [
    `모든 방화벽 결과는 seed ${DATA.firewall.examples.seed} 단일 실험 — 여러 seed에 걸친 일반화는 미확인`,
    `평가 대상은 SmallCNN 계열(${smallcnnModels.join(", ")})만 포함, LeNet 계열은 미평가`,
    "Adaptive attack (방어 구조를 알고 우회하는 공격)에 대한 보장 없음",
    "MNIST 기반 결과이므로 고해상도 · 실제 도메인에 직접 적용은 어려움",
    "Transfer attack에 대한 방화벽 성능은 별도로 평가하지 않음",
  ];
  document.getElementById("limitations-list").innerHTML = limitations.map((t) => `<li>${t}</li>`).join("");
  document.getElementById("references-list").innerHTML = DATA.references.map((r) => `[${r.id}] ${r.text}`).join("<br>");
}

/* ============================================================
   Navigation
   ============================================================ */
function buildNavDots() {
  const dots = document.getElementById("nav-dots");
  const names = Array.from(document.querySelectorAll(".slide")).map((el) => el.dataset.name);
  dots.innerHTML = names.map((name, i) => `<button class="nav-dot" data-idx="${i}" title="${name}"></button>`).join("");
  dots.querySelectorAll(".nav-dot").forEach((btn) => {
    btn.addEventListener("click", () => goToSlide(Number(btn.dataset.idx)));
  });
}

function goToSlide(index) {
  const slideCount = document.querySelectorAll(".slide").length;
  const clamped = Math.max(0, Math.min(slideCount - 1, index));
  currentSlide = clamped;
  document.querySelectorAll(".slide").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.index) === clamped);
  });
  document.querySelectorAll(".nav-dot").forEach((el, i) => el.classList.toggle("active", i === clamped));
  document.getElementById("nav-counter").textContent = `${clamped + 1} / ${slideCount}`;
  document.getElementById("topbar-fill").style.width = `${((clamped + 1) / slideCount) * 100}%`;
  const activeEl = document.querySelector(`.slide[data-index="${clamped}"]`);
  document.getElementById("nav-slidename").textContent = activeEl ? activeEl.dataset.name : "";
  hideTooltip();
}

function nextSlide() { goToSlide(currentSlide + 1); }
function prevSlide() { goToSlide(currentSlide - 1); }

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen().catch(() => {});
  }
}

function toggleHelp() {
  document.getElementById("help-overlay").classList.toggle("open");
}
function closeHelp() {
  document.getElementById("help-overlay").classList.remove("open");
}

function setupNav() {
  document.getElementById("nav-prev").addEventListener("click", prevSlide);
  document.getElementById("nav-next").addEventListener("click", nextSlide);
  document.getElementById("deck").addEventListener("click", (evt) => {
    if (evt.target.closest("button, input, .panel, a")) return;
    nextSlide();
  });
  document.addEventListener("keydown", (evt) => {
    if (evt.key === "?") { toggleHelp(); return; }
    if (evt.key === "Escape") { closeHelp(); return; }
    if (document.getElementById("help-overlay").classList.contains("open")) return;
    if (evt.key === "ArrowRight" || evt.key === " ") { evt.preventDefault(); nextSlide(); }
    else if (evt.key === "ArrowLeft") { evt.preventDefault(); prevSlide(); }
    else if (evt.key.toLowerCase() === "f") { toggleFullscreen(); }
  });
  document.getElementById("help-overlay").addEventListener("click", closeHelp);
  if (!document.documentElement.requestFullscreen) {
    document.body.classList.add("no-fullscreen-api");
  } else {
    document.getElementById("nav-fullscreen").addEventListener("click", toggleFullscreen);
  }
}

/* ============================================================
   Bootstrap
   ============================================================ */
function init() {
  buildNavDots();
  setupNav();
  renderTitle();
  renderSetup();
  renderFgsmWorks();
  renderPgdInstability();
  renderTransferHeatmap();
  renderPivot();
  initLiveDemo();
  renderRecoverySlide();
  renderConclusion();
  goToSlide(0);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
"""
