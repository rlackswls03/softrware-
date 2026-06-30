"""Build an interactive browser simulation from saved Firewall artifacts."""

from __future__ import annotations

import base64
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from adversarial_mnist.firewall import ACCEPT_ORIGINAL, ACCEPT_PURIFIED, REJECT_SUSPICIOUS
from adversarial_mnist.utils import ensure_dir

CONDITION_ORDER: tuple[str, ...] = ("Clean", "FGSM", "PGD")
IMAGE_SIZE = 168


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a dictionary artifact at {path}.")
    return loaded


def _require_file(path: str | Path) -> Path:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Required simulation artifact is missing: {target}")
    return target


def _tensor_image_to_data_uri(tensor: torch.Tensor) -> str:
    image_array = tensor.detach().cpu().squeeze().clamp(0.0, 1.0).numpy()
    if image_array.ndim != 2:
        raise ValueError("Expected a single grayscale image tensor.")
    pixels = np.rint(image_array * 255.0).astype(np.uint8)
    resampling = getattr(Image, "Resampling", Image).NEAREST
    image = Image.fromarray(pixels).resize((IMAGE_SIZE, IMAGE_SIZE), resample=resampling)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _final_prediction(decision: str, input_prediction: int, purified_prediction: int) -> int | None:
    if decision == ACCEPT_ORIGINAL:
        return input_prediction
    if decision == ACCEPT_PURIFIED:
        return purified_prediction
    if decision == REJECT_SUSPICIOUS:
        return None
    raise ValueError(f"Unknown Firewall decision: {decision}")


def _decision_label(decision: str) -> str:
    labels = {
        ACCEPT_ORIGINAL: "원본 통과",
        ACCEPT_PURIFIED: "정화 후 통과",
        REJECT_SUSPICIOUS: "위험 입력 거부",
    }
    return labels.get(decision, decision)


def _condition_label(condition: str) -> str:
    labels = {
        "Clean": "정상 입력",
        "FGSM": "FGSM 공격",
        "PGD": "PGD 공격",
    }
    return labels.get(condition, condition)


def _as_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _as_json_value(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def _records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append({key: _as_json_value(value) for key, value in row.items()})
    return records


def _condition_sample_count(condition_data: dict[str, Any]) -> int:
    required = [
        "original",
        "input",
        "purified",
        "labels",
        "clean_predictions",
        "input_predictions",
        "purified_predictions",
        "scores",
        "decisions",
    ]
    for key in required:
        if key not in condition_data:
            raise ValueError(f"Firewall examples are missing condition field '{key}'.")
    tensor_lengths = [
        int(condition_data[key].shape[0])
        for key in required
        if isinstance(condition_data[key], torch.Tensor)
    ]
    list_lengths = [
        len(condition_data[key]) for key in required if isinstance(condition_data[key], list)
    ]
    return min([*tensor_lengths, *list_lengths])


def _build_sample(condition: str, condition_data: dict[str, Any], index: int, threshold: float) -> dict[str, Any]:
    original = condition_data["original"][index]
    input_image = condition_data["input"][index]
    purified = condition_data["purified"][index]
    label = int(condition_data["labels"][index].item())
    clean_prediction = int(condition_data["clean_predictions"][index].item())
    input_prediction = int(condition_data["input_predictions"][index].item())
    purified_prediction = int(condition_data["purified_predictions"][index].item())
    score = float(condition_data["scores"][index].item())
    decision = str(condition_data["decisions"][index])
    final_prediction = _final_prediction(decision, input_prediction, purified_prediction)
    correct_after_firewall = final_prediction == label if final_prediction is not None else False
    attack_changed_prediction = input_prediction != clean_prediction
    perturbation_linf = float((input_image - original).abs().max().item())
    purification_l1 = float((purified - input_image).abs().mean().item())
    return {
        "index": index,
        "condition": condition,
        "conditionLabel": _condition_label(condition),
        "images": {
            "original": _tensor_image_to_data_uri(original),
            "input": _tensor_image_to_data_uri(input_image),
            "purified": _tensor_image_to_data_uri(purified),
        },
        "label": label,
        "cleanPrediction": clean_prediction,
        "inputPrediction": input_prediction,
        "purifiedPrediction": purified_prediction,
        "finalPrediction": final_prediction,
        "score": score,
        "threshold": threshold,
        "detected": score > threshold,
        "decision": decision,
        "decisionLabel": _decision_label(decision),
        "correctAfterFirewall": correct_after_firewall,
        "attackChangedPrediction": attack_changed_prediction,
        "perturbationLinf": perturbation_linf,
        "purificationMeanAbs": purification_l1,
    }


def _build_examples_payload(examples: dict[str, Any]) -> dict[str, Any]:
    if "conditions" not in examples or not isinstance(examples["conditions"], dict):
        raise ValueError("Firewall examples must contain a 'conditions' dictionary.")
    threshold = float(examples.get("threshold", 0.0))
    conditions_payload: dict[str, Any] = {}
    for condition in CONDITION_ORDER:
        condition_data = examples["conditions"].get(condition)
        if not isinstance(condition_data, dict):
            continue
        count = _condition_sample_count(condition_data)
        conditions_payload[condition] = {
            "label": _condition_label(condition),
            "samples": [
                _build_sample(condition, condition_data, index, threshold) for index in range(count)
            ],
        }
    if not conditions_payload:
        raise ValueError("No supported Clean/FGSM/PGD conditions were found in firewall examples.")
    return {
        "model": str(examples.get("model", "")),
        "seed": int(examples.get("seed", 0)),
        "epsilon": float(examples.get("epsilon", 0.0)),
        "threshold": threshold,
        "minConfidence": float(examples.get("min_confidence", 0.0)),
        "conditions": conditions_payload,
    }


def _build_metrics_payload(metrics_path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(metrics_path)
    required = {
        "model",
        "seed",
        "condition",
        "original_accuracy",
        "purified_accuracy",
        "detection_rate",
        "reject_rate",
        "final_safe_accuracy",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{metrics_path} is missing required columns: {sorted(missing)}")
    return _records_from_frame(frame)


def _simulation_html(data: dict[str, Any]) -> str:
    json_data = json.dumps(data, ensure_ascii=False, allow_nan=False)
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adversarial Firewall Simulation</title>
  <style>
    :root {
      --ink: #1e252f;
      --muted: #64707d;
      --line: #d6dde6;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --blue: #315f92;
      --green: #2f7a55;
      --amber: #9c6b22;
      --red: #a34242;
      --stage: #eef3f8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, "Malgun Gothic", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.5;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-end;
      padding: 22px 28px;
      color: #ffffff;
      background: #17202c;
    }
    h1 {
      margin: 0 0 6px;
      font-size: clamp(26px, 4vw, 42px);
      letter-spacing: 0;
    }
    header p {
      max-width: 900px;
      margin: 0;
      color: #d5dce5;
      font-size: 15px;
    }
    .run-info {
      min-width: 230px;
      text-align: right;
      color: #d5dce5;
      font-size: 13px;
    }
    main {
      max-width: 1380px;
      margin: 0 auto;
      padding: 18px;
    }
    .top-grid {
      display: grid;
      grid-template-columns: minmax(260px, 330px) 1fr;
      gap: 16px;
      align-items: stretch;
    }
    section, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    h2, h3 {
      margin: 0 0 12px;
      letter-spacing: 0;
    }
    h2 { font-size: 20px; }
    h3 { font-size: 16px; }
    label {
      display: block;
      margin: 12px 0 6px;
      font-weight: 700;
      font-size: 13px;
      color: var(--muted);
    }
    select, input[type="range"] {
      width: 100%;
    }
    select {
      min-height: 38px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
    }
    .button-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 14px;
    }
    button {
      min-height: 38px;
      border: 1px solid #254c76;
      border-radius: 6px;
      background: var(--blue);
      color: #ffffff;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      border-color: var(--line);
      color: var(--ink);
      background: #eef2f6;
    }
    button:focus-visible, select:focus-visible, input:focus-visible {
      outline: 3px solid #8ab4e8;
      outline-offset: 2px;
    }
    .sample-value {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .timeline {
      display: grid;
      grid-template-columns: repeat(4, minmax(170px, 1fr));
      gap: 12px;
    }
    .stage {
      position: relative;
      min-height: 386px;
      border: 2px solid var(--line);
      border-radius: 8px;
      background: var(--stage);
      padding: 12px;
      transition: border-color 180ms ease, transform 180ms ease, box-shadow 180ms ease;
    }
    .stage.active {
      border-color: var(--blue);
      box-shadow: 0 8px 18px rgba(49, 95, 146, 0.18);
      transform: translateY(-2px);
    }
    .stage.done {
      border-color: #9eb8a9;
    }
    .stage-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      margin-bottom: 10px;
      font-weight: 800;
    }
    .stage-number {
      display: inline-grid;
      width: 28px;
      height: 28px;
      place-items: center;
      flex: 0 0 auto;
      border-radius: 50%;
      color: #ffffff;
      background: #52616f;
      font-size: 13px;
    }
    .stage.active .stage-number { background: var(--blue); }
    .image-shell {
      display: grid;
      place-items: center;
      width: 100%;
      min-height: 190px;
      margin-bottom: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
    }
    .digit {
      width: 168px;
      height: 168px;
      image-rendering: pixelated;
    }
    .kv {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 6px 10px;
      align-items: baseline;
      font-size: 13px;
    }
    .kv span:nth-child(odd) { color: var(--muted); }
    .kv strong {
      font-size: 14px;
      text-align: right;
    }
    .progress-track {
      position: relative;
      height: 9px;
      margin: 16px 0 2px;
      border-radius: 999px;
      background: #e3e8ee;
      overflow: hidden;
    }
    .progress-fill {
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, #315f92, #2f7a55);
      transition: width 220ms ease;
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .status {
      border: 1px solid var(--line);
      border-left: 5px solid var(--blue);
      border-radius: 6px;
      padding: 12px;
      background: #fbfcfe;
    }
    .status strong {
      display: block;
      margin-top: 5px;
      font-size: 22px;
    }
    .status span {
      color: var(--muted);
      font-size: 13px;
    }
    .decision-accept { color: var(--green); }
    .decision-purify { color: var(--amber); }
    .decision-reject { color: var(--red); }
    .metrics-grid {
      display: grid;
      grid-template-columns: minmax(250px, 360px) 1fr;
      gap: 16px;
      margin-top: 16px;
    }
    .metric-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(120px, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .metric strong {
      display: block;
      margin-top: 4px;
      font-size: 20px;
    }
    .log {
      min-height: 126px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #111827;
      color: #d7dee8;
      font-family: Consolas, "Courier New", monospace;
      font-size: 13px;
      white-space: pre-wrap;
    }
    .gauge {
      position: relative;
      height: 22px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #e9edf2;
      overflow: hidden;
    }
    .gauge-fill {
      height: 100%;
      width: 0%;
      background: #9c6b22;
      transition: width 180ms ease;
    }
    .gauge-marker {
      position: absolute;
      top: -2px;
      bottom: -2px;
      width: 3px;
      background: #a34242;
      left: 50%;
    }
    .note {
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 980px) {
      header {
        display: block;
      }
      .run-info {
        margin-top: 12px;
        text-align: left;
      }
      .top-grid, .metrics-grid {
        grid-template-columns: 1fr;
      }
      .timeline {
        grid-template-columns: repeat(2, minmax(160px, 1fr));
      }
      .status-grid {
        grid-template-columns: repeat(2, minmax(140px, 1fr));
      }
    }
    @media (max-width: 560px) {
      main { padding: 10px; }
      .timeline, .status-grid, .metric-list {
        grid-template-columns: 1fr;
      }
      .stage { min-height: auto; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Adversarial Firewall Simulation</h1>
      <p>저장된 MNIST 예시를 사용해 원본 입력, 공격 입력, autoencoder 정화, reconstruction-error 탐지, 최종 판정을 한 화면에서 재생합니다.</p>
    </div>
    <div class="run-info">
      <div>Model examples: <strong id="meta-model"></strong></div>
      <div>Seed <strong id="meta-seed"></strong>, epsilon <strong id="meta-epsilon"></strong></div>
      <div>Threshold <strong id="meta-threshold"></strong></div>
    </div>
  </header>
  <main>
    <div class="top-grid">
      <section>
        <h2>Simulation Control</h2>
        <label for="condition-select">Attack condition</label>
        <select id="condition-select"></select>
        <label for="sample-range">Sample</label>
        <input id="sample-range" type="range" min="0" max="0" value="0">
        <div class="sample-value">
          <span id="sample-label">sample 1 / 1</span>
          <span id="condition-label"></span>
        </div>
        <label for="metrics-model-select">Full-test metric model</label>
        <select id="metrics-model-select"></select>
        <div class="button-row">
          <button id="run-button" type="button">Run</button>
          <button id="auto-button" type="button" class="secondary">Auto Play</button>
          <button id="prev-button" type="button" class="secondary">Previous</button>
          <button id="next-button" type="button" class="secondary">Next</button>
        </div>
      </section>
      <section>
        <h2>Live Pipeline</h2>
        <div class="timeline">
          <article class="stage" data-stage="0">
            <div class="stage-title"><span>Original</span><span class="stage-number">1</span></div>
            <div class="image-shell"><img id="img-original" class="digit" alt="original MNIST image"></div>
            <div class="kv">
              <span>True label</span><strong id="original-label"></strong>
              <span>Clean prediction</span><strong id="clean-prediction"></strong>
            </div>
          </article>
          <article class="stage" data-stage="1">
            <div class="stage-title"><span>Attack/Input</span><span class="stage-number">2</span></div>
            <div class="image-shell"><img id="img-input" class="digit" alt="input MNIST image"></div>
            <div class="kv">
              <span>Input prediction</span><strong id="input-prediction"></strong>
              <span>L-inf change</span><strong id="linf-change"></strong>
              <span>Changed pred</span><strong id="changed-prediction"></strong>
            </div>
          </article>
          <article class="stage" data-stage="2">
            <div class="stage-title"><span>Purifier</span><span class="stage-number">3</span></div>
            <div class="image-shell"><img id="img-purified" class="digit" alt="purified MNIST image"></div>
            <div class="kv">
              <span>Purified prediction</span><strong id="purified-prediction"></strong>
              <span>Mean abs update</span><strong id="purify-change"></strong>
            </div>
          </article>
          <article class="stage" data-stage="3">
            <div class="stage-title"><span>Detector/Decision</span><span class="stage-number">4</span></div>
            <div class="gauge" aria-label="reconstruction score gauge">
              <div id="gauge-fill" class="gauge-fill"></div>
              <div id="gauge-marker" class="gauge-marker"></div>
            </div>
            <div class="kv" style="margin-top: 16px;">
              <span>Recon score</span><strong id="score-value"></strong>
              <span>Threshold</span><strong id="threshold-value"></strong>
              <span>Detected</span><strong id="detected-value"></strong>
              <span>Decision</span><strong id="decision-value"></strong>
              <span>Final prediction</span><strong id="final-prediction"></strong>
            </div>
          </article>
        </div>
        <div class="progress-track"><div id="progress-fill" class="progress-fill"></div></div>
      </section>
    </div>

    <section style="margin-top: 16px;">
      <h2>Sample Outcome</h2>
      <div class="status-grid">
        <div class="status"><span>Scenario</span><strong id="status-scenario"></strong></div>
        <div class="status"><span>Attack effect</span><strong id="status-attack"></strong></div>
        <div class="status"><span>Firewall decision</span><strong id="status-decision"></strong></div>
        <div class="status"><span>Final correctness</span><strong id="status-correct"></strong></div>
      </div>
    </section>

    <div class="metrics-grid">
      <section>
        <h2>Full-Test Metrics</h2>
        <div id="metric-list" class="metric-list"></div>
        <p class="note">이 지표는 저장된 `firewall_results.csv`에서 읽은 값이며, 위 샘플 이미지는 `firewall_examples.pt`의 대표 예시입니다.</p>
      </section>
      <section>
        <h2>Step Log</h2>
        <div id="log-output" class="log"></div>
      </section>
    </div>
  </main>
  <script>
    const SIM_DATA = __SIM_DATA__;
    const CONDITION_ORDER = ["Clean", "FGSM", "PGD"];
    const stageDelayMs = 650;
    let autoTimer = null;

    const els = {
      conditionSelect: document.getElementById("condition-select"),
      sampleRange: document.getElementById("sample-range"),
      sampleLabel: document.getElementById("sample-label"),
      conditionLabel: document.getElementById("condition-label"),
      metricsModelSelect: document.getElementById("metrics-model-select"),
      runButton: document.getElementById("run-button"),
      autoButton: document.getElementById("auto-button"),
      prevButton: document.getElementById("prev-button"),
      nextButton: document.getElementById("next-button"),
      stages: Array.from(document.querySelectorAll(".stage")),
      progressFill: document.getElementById("progress-fill"),
      imgOriginal: document.getElementById("img-original"),
      imgInput: document.getElementById("img-input"),
      imgPurified: document.getElementById("img-purified"),
      metaModel: document.getElementById("meta-model"),
      metaSeed: document.getElementById("meta-seed"),
      metaEpsilon: document.getElementById("meta-epsilon"),
      metaThreshold: document.getElementById("meta-threshold"),
      originalLabel: document.getElementById("original-label"),
      cleanPrediction: document.getElementById("clean-prediction"),
      inputPrediction: document.getElementById("input-prediction"),
      linfChange: document.getElementById("linf-change"),
      changedPrediction: document.getElementById("changed-prediction"),
      purifiedPrediction: document.getElementById("purified-prediction"),
      purifyChange: document.getElementById("purify-change"),
      gaugeFill: document.getElementById("gauge-fill"),
      gaugeMarker: document.getElementById("gauge-marker"),
      scoreValue: document.getElementById("score-value"),
      thresholdValue: document.getElementById("threshold-value"),
      detectedValue: document.getElementById("detected-value"),
      decisionValue: document.getElementById("decision-value"),
      finalPrediction: document.getElementById("final-prediction"),
      statusScenario: document.getElementById("status-scenario"),
      statusAttack: document.getElementById("status-attack"),
      statusDecision: document.getElementById("status-decision"),
      statusCorrect: document.getElementById("status-correct"),
      metricList: document.getElementById("metric-list"),
      logOutput: document.getElementById("log-output"),
    };

    function pct(value) {
      if (value === null || Number.isNaN(Number(value))) return "NaN";
      return `${(Number(value) * 100).toFixed(2)}%`;
    }

    function fixed(value, digits = 4) {
      if (value === null || Number.isNaN(Number(value))) return "NaN";
      return Number(value).toFixed(digits);
    }

    function currentCondition() {
      return els.conditionSelect.value;
    }

    function currentSamples() {
      return SIM_DATA.examples.conditions[currentCondition()].samples;
    }

    function currentSample() {
      return currentSamples()[Number(els.sampleRange.value)];
    }

    function conditionMetrics(model, condition) {
      return SIM_DATA.metrics.find((row) => row.model === model && row.condition === condition) || null;
    }

    function decisionClass(decision) {
      if (decision === "ACCEPT_PURIFIED") return "decision-purify";
      if (decision === "REJECT_SUSPICIOUS") return "decision-reject";
      return "decision-accept";
    }

    function setStage(stageIndex) {
      els.stages.forEach((stage, index) => {
        stage.classList.toggle("active", index === stageIndex);
        stage.classList.toggle("done", index < stageIndex);
      });
      els.progressFill.style.width = `${((stageIndex + 1) / els.stages.length) * 100}%`;
    }

    function renderMetrics() {
      const model = els.metricsModelSelect.value;
      const row = conditionMetrics(model, currentCondition());
      const items = row ? [
        ["Original acc", pct(row.original_accuracy)],
        ["Purified acc", pct(row.purified_accuracy)],
        ["Detection rate", pct(row.detection_rate)],
        ["Reject rate", pct(row.reject_rate)],
        ["Final safe acc", pct(row.final_safe_accuracy)],
        ["Samples", String(row.evaluated_samples)],
      ] : [["No metrics", "missing"]];
      els.metricList.innerHTML = items.map(([label, value]) => `
        <div class="metric"><span>${label}</span><strong>${value}</strong></div>
      `).join("");
    }

    function renderStaticSample() {
      const condition = currentCondition();
      const samples = currentSamples();
      const index = Math.min(Number(els.sampleRange.value), samples.length - 1);
      els.sampleRange.value = String(index);
      const sample = samples[index];
      els.sampleRange.max = String(samples.length - 1);
      els.sampleLabel.textContent = `sample ${index + 1} / ${samples.length}`;
      els.conditionLabel.textContent = sample.conditionLabel;

      els.imgOriginal.src = sample.images.original;
      els.imgInput.src = sample.images.input;
      els.imgPurified.src = sample.images.purified;
      els.originalLabel.textContent = sample.label;
      els.cleanPrediction.textContent = sample.cleanPrediction;
      els.inputPrediction.textContent = sample.inputPrediction;
      els.linfChange.textContent = fixed(sample.perturbationLinf);
      els.changedPrediction.textContent = sample.attackChangedPrediction ? "yes" : "no";
      els.purifiedPrediction.textContent = sample.purifiedPrediction;
      els.purifyChange.textContent = fixed(sample.purificationMeanAbs);
      els.scoreValue.textContent = fixed(sample.score, 6);
      els.thresholdValue.textContent = fixed(sample.threshold, 6);
      els.detectedValue.textContent = sample.detected ? "yes" : "no";
      els.decisionValue.textContent = sample.decisionLabel;
      els.decisionValue.className = decisionClass(sample.decision);
      els.finalPrediction.textContent = sample.finalPrediction === null ? "reject" : sample.finalPrediction;

      const scaleMax = Math.max(sample.threshold * 1.8, sample.score, 0.000001);
      const gaugeWidth = Math.min(100, (sample.score / scaleMax) * 100);
      const markerLeft = Math.min(100, (sample.threshold / scaleMax) * 100);
      els.gaugeFill.style.width = `${gaugeWidth}%`;
      els.gaugeMarker.style.left = `${markerLeft}%`;
      els.gaugeFill.style.background = sample.detected ? "#9c6b22" : "#2f7a55";

      els.statusScenario.textContent = sample.conditionLabel;
      els.statusAttack.textContent = condition === "Clean"
        ? "no attack"
        : (sample.attackChangedPrediction ? "prediction changed" : "prediction held");
      els.statusDecision.textContent = sample.decisionLabel;
      els.statusDecision.className = decisionClass(sample.decision);
      els.statusCorrect.textContent = sample.correctAfterFirewall ? "correct" : "not correct";
      els.statusCorrect.style.color = sample.correctAfterFirewall ? "var(--green)" : "var(--red)";
      renderMetrics();
      renderLog(0);
    }

    function renderLog(stageIndex) {
      const sample = currentSample();
      const lines = [
        `[0] condition=${sample.condition}, sample=${sample.index}, true_label=${sample.label}`,
        `[1] clean_prediction=${sample.cleanPrediction}`,
      ];
      if (stageIndex >= 1) {
        lines.push(`[2] input_prediction=${sample.inputPrediction}, l_inf_delta=${fixed(sample.perturbationLinf)}`);
      }
      if (stageIndex >= 2) {
        lines.push(`[3] purified_prediction=${sample.purifiedPrediction}, mean_abs_update=${fixed(sample.purificationMeanAbs)}`);
      }
      if (stageIndex >= 3) {
        lines.push(`[4] score=${fixed(sample.score, 6)}, threshold=${fixed(sample.threshold, 6)}, detected=${sample.detected}`);
        lines.push(`[5] decision=${sample.decision}, final=${sample.finalPrediction === null ? "reject" : sample.finalPrediction}`);
      }
      els.logOutput.textContent = lines.join("\\n");
    }

    function runSimulation() {
      window.clearTimeout(autoTimer);
      let stage = 0;
      setStage(stage);
      renderLog(stage);
      function tick() {
        stage += 1;
        if (stage >= els.stages.length) return;
        setStage(stage);
        renderLog(stage);
        autoTimer = window.setTimeout(tick, stageDelayMs);
      }
      autoTimer = window.setTimeout(tick, stageDelayMs);
    }

    function nextSample(delta = 1) {
      const samples = currentSamples();
      const next = (Number(els.sampleRange.value) + delta + samples.length) % samples.length;
      els.sampleRange.value = String(next);
      renderStaticSample();
      runSimulation();
    }

    function autoPlay() {
      nextSample(1);
      autoTimer = window.setTimeout(autoPlay, stageDelayMs * 5);
    }

    function populateControls() {
      const conditionOptions = CONDITION_ORDER
        .filter((condition) => SIM_DATA.examples.conditions[condition])
        .map((condition) => {
          const label = SIM_DATA.examples.conditions[condition].label;
          return `<option value="${condition}">${label} (${condition})</option>`;
        }).join("");
      els.conditionSelect.innerHTML = conditionOptions;

      const models = Array.from(new Set(SIM_DATA.metrics.map((row) => row.model)));
      els.metricsModelSelect.innerHTML = models.map((model) => {
        const selected = model === SIM_DATA.examples.model ? " selected" : "";
        return `<option value="${model}"${selected}>${model}</option>`;
      }).join("");

      els.metaModel.textContent = SIM_DATA.examples.model;
      els.metaSeed.textContent = SIM_DATA.examples.seed;
      els.metaEpsilon.textContent = fixed(SIM_DATA.examples.epsilon, 2);
      els.metaThreshold.textContent = fixed(SIM_DATA.examples.threshold, 6);
    }

    els.conditionSelect.addEventListener("change", () => {
      els.sampleRange.value = "0";
      renderStaticSample();
      runSimulation();
    });
    els.sampleRange.addEventListener("input", () => {
      renderStaticSample();
      setStage(0);
    });
    els.metricsModelSelect.addEventListener("change", renderMetrics);
    els.runButton.addEventListener("click", runSimulation);
    els.prevButton.addEventListener("click", () => nextSample(-1));
    els.nextButton.addEventListener("click", () => nextSample(1));
    els.autoButton.addEventListener("click", autoPlay);

    populateControls();
    renderStaticSample();
    runSimulation();
  </script>
</body>
</html>
""".replace("__SIM_DATA__", json_data)


def build_simulation(
    config: dict[str, Any],
    examples_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate a standalone interactive HTML simulation from saved results."""
    raw_dir = Path(config["paths"]["raw_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    examples_file = _require_file(examples_path or raw_dir / "firewall_examples.pt")
    metrics_file = _require_file(metrics_path or raw_dir / "firewall_results.csv")
    output_root = ensure_dir(output_dir or results_dir / "simulation")

    examples = _build_examples_payload(_torch_load(examples_file))
    metrics = _build_metrics_payload(metrics_file)
    payload = {
        "examples": examples,
        "metrics": metrics,
        "sourceFiles": {
            "examples": str(examples_file),
            "metrics": str(metrics_file),
        },
    }
    output_path = output_root / "index.html"
    output_path.write_text(_simulation_html(payload), encoding="utf-8")
    return output_path
