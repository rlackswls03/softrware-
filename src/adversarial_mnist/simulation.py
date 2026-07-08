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


def _simulation_html_v2(data: dict[str, Any]) -> str:
    """Render a more cinematic browser simulation from saved Firewall samples."""
    json_data = json.dumps(data, ensure_ascii=False, allow_nan=False)
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adversarial Firewall Simulation</title>
  <style>
    :root {
      --ink: #18212d;
      --muted: #6a7481;
      --line: #d7dde5;
      --paper: #ffffff;
      --bg: #eef2f6;
      --navy: #172233;
      --blue: #2f5d7c;
      --red: #b64b4b;
      --amber: #d88c2d;
      --green: #2f7a55;
      --purple: #7f4c8a;
      --track: #e4e9ef;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, "Malgun Gothic", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(180deg, #111827 0, #111827 280px, var(--bg) 280px),
        var(--bg);
      line-height: 1.45;
    }
    header {
      max-width: 1440px;
      margin: 0 auto;
      padding: 26px 28px 18px;
      color: #f8fafc;
    }
    header h1 {
      margin: 0 0 8px;
      font-size: clamp(30px, 4.5vw, 54px);
      letter-spacing: 0;
    }
    header p {
      max-width: 980px;
      margin: 0;
      color: #ccd5e1;
      font-size: 16px;
    }
    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 0 22px 28px;
    }
    .shell {
      display: grid;
      grid-template-columns: 310px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }
    .controls {
      position: sticky;
      top: 14px;
      padding: 16px;
    }
    h2, h3 {
      margin: 0;
      letter-spacing: 0;
    }
    h2 { font-size: 20px; }
    h3 { font-size: 15px; }
    .control-block {
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }
    label {
      display: block;
      margin: 0 0 7px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
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
    input[type="range"] {
      accent-color: var(--blue);
    }
    button {
      min-height: 38px;
      border: 1px solid #244966;
      border-radius: 6px;
      color: #ffffff;
      background: var(--blue);
      font-weight: 800;
      cursor: pointer;
    }
    button.secondary {
      border-color: var(--line);
      color: var(--ink);
      background: #f0f4f8;
    }
    button.warn {
      border-color: #873939;
      background: var(--red);
    }
    button:focus-visible, select:focus-visible, input:focus-visible {
      outline: 3px solid #8ab4e8;
      outline-offset: 2px;
    }
    .button-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .meta {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #f8fafc;
    }
    .meta span {
      display: block;
      color: var(--muted);
      font-size: 11px;
    }
    .meta strong {
      display: block;
      margin-top: 3px;
      font-size: 16px;
    }
    .sample-readout {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .arena {
      overflow: hidden;
    }
    .arena-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      padding: 18px 18px 0;
    }
    .arena-head p {
      max-width: 780px;
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
    }
    .decision-pill {
      min-width: 190px;
      border-radius: 999px;
      padding: 9px 14px;
      color: #ffffff;
      background: var(--blue);
      text-align: center;
      font-weight: 900;
    }
    .decision-pill.accept { background: var(--green); }
    .decision-pill.purify { background: var(--amber); }
    .decision-pill.reject { background: var(--red); }
    .stage-rail {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 16px 18px 0;
    }
    .beat {
      position: relative;
      min-height: 78px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px;
      background: #f8fafc;
      transition: border-color 180ms ease, background 180ms ease, transform 180ms ease;
    }
    .beat::before {
      content: attr(data-step);
      display: inline-grid;
      width: 26px;
      height: 26px;
      place-items: center;
      margin-right: 7px;
      border-radius: 50%;
      color: #ffffff;
      background: #667485;
      font-size: 12px;
      font-weight: 900;
    }
    .beat strong {
      font-size: 14px;
    }
    .beat span {
      display: block;
      margin-top: 7px;
      color: var(--muted);
      font-size: 12px;
    }
    .beat.active {
      border-color: var(--blue);
      background: #eef6fb;
      transform: translateY(-2px);
    }
    .beat.active::before {
      background: var(--blue);
    }
    .beat.done {
      border-color: #9fc1ae;
    }
    .stage-bar {
      height: 8px;
      margin: 14px 18px 0;
      border-radius: 999px;
      background: var(--track);
      overflow: hidden;
    }
    .stage-fill {
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, var(--blue), var(--amber), var(--green));
      transition: width 220ms ease;
    }
    .scene {
      display: grid;
      grid-template-columns: 1.1fr 0.72fr;
      gap: 16px;
      padding: 16px 18px 18px;
    }
    .image-flow {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .image-card {
      position: relative;
      min-height: 396px;
      border: 2px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #f8fafc;
      transition: opacity 180ms ease, border-color 180ms ease, transform 180ms ease;
    }
    .image-card.pending {
      opacity: 0.45;
      filter: grayscale(0.7);
    }
    .image-card.active {
      border-color: var(--blue);
      transform: translateY(-2px);
      box-shadow: 0 10px 24px rgba(47, 93, 124, 0.16);
    }
    .image-card.attack.active {
      border-color: var(--red);
      box-shadow: 0 10px 24px rgba(182, 75, 75, 0.16);
    }
    .image-card.purifier.active {
      border-color: var(--green);
      box-shadow: 0 10px 24px rgba(47, 122, 85, 0.16);
    }
    .image-card header {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      margin: 0 0 12px;
      padding: 0;
      color: var(--ink);
    }
    .image-card h3 {
      font-size: 16px;
    }
    .tag {
      border-radius: 999px;
      padding: 4px 8px;
      color: #ffffff;
      background: #64748b;
      font-size: 12px;
      font-weight: 800;
    }
    .tag.blue { background: var(--blue); }
    .tag.red { background: var(--red); }
    .tag.green { background: var(--green); }
    .digit-wrap {
      display: grid;
      place-items: center;
      min-height: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(45deg, #f5f7fa 25%, transparent 25%),
        linear-gradient(-45deg, #f5f7fa 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #f5f7fa 75%),
        linear-gradient(-45deg, transparent 75%, #f5f7fa 75%),
        #ffffff;
      background-position: 0 0, 0 10px, 10px -10px, -10px 0;
      background-size: 20px 20px;
    }
    .digit {
      width: 168px;
      height: 168px;
      image-rendering: pixelated;
    }
    .image-card.attack .digit.active {
      animation: shake 420ms ease;
    }
    @keyframes shake {
      0% { transform: translateX(0); }
      25% { transform: translateX(-3px); }
      50% { transform: translateX(3px); }
      75% { transform: translateX(-2px); }
      100% { transform: translateX(0); }
    }
    .stat-list {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 7px 10px;
      margin-top: 12px;
      font-size: 13px;
    }
    .stat-list span:nth-child(odd) {
      color: var(--muted);
    }
    .stat-list strong {
      text-align: right;
    }
    .right-stack {
      display: grid;
      gap: 12px;
    }
    .live-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
      background: #fbfcfe;
    }
    .live-title {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
      font-weight: 900;
    }
    .gauge {
      position: relative;
      height: 24px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #e7ebf0;
      overflow: hidden;
    }
    .gauge-fill {
      width: 0%;
      height: 100%;
      background: var(--green);
      transition: width 220ms ease, background 220ms ease;
    }
    .gauge-marker {
      position: absolute;
      top: -2px;
      bottom: -2px;
      width: 3px;
      left: 0%;
      background: #111827;
    }
    .score-row {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .mini {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #ffffff;
    }
    .mini span {
      display: block;
      color: var(--muted);
      font-size: 11px;
    }
    .mini strong {
      display: block;
      margin-top: 4px;
      font-size: 16px;
    }
    .outcome {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .outcome .mini strong {
      font-size: 18px;
    }
    .console {
      min-height: 172px;
      padding: 12px;
      border-radius: 7px;
      background: #101827;
      color: #d8e1ec;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      white-space: pre-wrap;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .metric strong {
      color: var(--blue);
    }
    .note {
      margin: 11px 0 0;
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 1160px) {
      .shell, .scene {
        grid-template-columns: 1fr;
      }
      .controls {
        position: static;
      }
    }
    @media (max-width: 820px) {
      main { padding: 0 10px 18px; }
      .arena-head, .image-flow, .stage-rail, .metrics {
        grid-template-columns: 1fr;
      }
      .arena-head {
        display: block;
      }
      .decision-pill {
        margin-top: 12px;
      }
      .image-card {
        min-height: auto;
      }
      .stage-rail, .scene {
        padding-left: 12px;
        padding-right: 12px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>공격과 방어가 지나가는 과정을 눈으로 보기</h1>
    <p>
      저장된 MNIST 예시를 사용해 원본 입력, FGSM/PGD 공격 입력, autoencoder 정화,
      reconstruction-error 탐지, 최종 accept/purify/reject 판정을 단계별로 재생합니다.
      새 학습이나 새 공격을 실행하지 않고 기존 실험 산출물을 시뮬레이션처럼 보여줍니다.
    </p>
  </header>
  <main>
    <div class="shell">
      <aside class="panel controls">
        <h2>Simulation Control</h2>
        <div class="control-block">
          <label for="condition-select">공격 조건</label>
          <select id="condition-select"></select>
        </div>
        <div class="control-block">
          <label for="sample-range">샘플 선택</label>
          <input id="sample-range" type="range" min="0" max="0" value="0">
          <div class="sample-readout">
            <span id="sample-label">sample 1 / 1</span>
            <span id="condition-label"></span>
          </div>
        </div>
        <div class="control-block">
          <label for="model-select">full-test metric 모델</label>
          <select id="model-select"></select>
        </div>
        <div class="control-block button-grid">
          <button id="play-button" type="button">Play</button>
          <button id="pause-button" type="button" class="secondary">Pause</button>
          <button id="prev-button" type="button" class="secondary">Prev</button>
          <button id="next-button" type="button" class="secondary">Next</button>
          <button id="step-button" type="button" class="secondary">Step</button>
          <button id="auto-button" type="button" class="warn">Auto</button>
        </div>
        <div class="control-block">
          <label for="stage-scrub">단계 직접 보기</label>
          <input id="stage-scrub" type="range" min="0" max="3" value="0">
        </div>
        <div class="control-block meta-grid">
          <div class="meta"><span>examples model</span><strong id="meta-model"></strong></div>
          <div class="meta"><span>seed</span><strong id="meta-seed"></strong></div>
          <div class="meta"><span>epsilon</span><strong id="meta-epsilon"></strong></div>
          <div class="meta"><span>threshold</span><strong id="meta-threshold"></strong></div>
        </div>
      </aside>

      <section class="panel arena">
        <div class="arena-head">
          <div>
            <h2 id="scene-title">Adversarial Firewall replay</h2>
            <p id="scene-subtitle"></p>
          </div>
          <div id="decision-pill" class="decision-pill">대기 중</div>
        </div>

        <div class="stage-rail">
          <div class="beat" data-step="1" data-beat="0"><strong>원본 입력</strong><span>clean prediction 확인</span></div>
          <div class="beat" data-step="2" data-beat="1"><strong>공격 입력</strong><span>FGSM/PGD 교란 적용</span></div>
          <div class="beat" data-step="3" data-beat="2"><strong>정화</strong><span>autoencoder 재구성</span></div>
          <div class="beat" data-step="4" data-beat="3"><strong>판정</strong><span>score와 threshold 비교</span></div>
        </div>
        <div class="stage-bar"><div id="stage-fill" class="stage-fill"></div></div>

        <div class="scene">
          <div class="image-flow">
            <article id="card-original" class="image-card">
              <header><h3>Original</h3><span class="tag blue">clean</span></header>
              <div class="digit-wrap"><img id="img-original" class="digit" alt="original image"></div>
              <div class="stat-list">
                <span>true label</span><strong id="true-label"></strong>
                <span>clean pred</span><strong id="clean-pred"></strong>
              </div>
            </article>
            <article id="card-input" class="image-card attack pending">
              <header><h3>Attack/Input</h3><span class="tag red" id="attack-tag">attack</span></header>
              <div class="digit-wrap"><img id="img-input" class="digit" alt="attacked input image"></div>
              <div class="stat-list">
                <span>input pred</span><strong id="input-pred"></strong>
                <span>L-inf delta</span><strong id="linf"></strong>
                <span>prediction changed</span><strong id="changed"></strong>
              </div>
            </article>
            <article id="card-purified" class="image-card purifier pending">
              <header><h3>Purified</h3><span class="tag green">autoencoder</span></header>
              <div class="digit-wrap"><img id="img-purified" class="digit" alt="purified image"></div>
              <div class="stat-list">
                <span>purified pred</span><strong id="purified-pred"></strong>
                <span>mean abs update</span><strong id="purify-delta"></strong>
              </div>
            </article>
          </div>

          <div class="right-stack">
            <section class="live-card">
              <div class="live-title">
                <span>Detector score</span>
                <span id="detected-label"></span>
              </div>
              <div class="gauge">
                <div id="gauge-fill" class="gauge-fill"></div>
                <div id="gauge-marker" class="gauge-marker"></div>
              </div>
              <div class="score-row">
                <div class="mini"><span>score</span><strong id="score"></strong></div>
                <div class="mini"><span>threshold</span><strong id="threshold"></strong></div>
                <div class="mini"><span>detected</span><strong id="detected"></strong></div>
              </div>
            </section>

            <section class="live-card">
              <div class="live-title"><span>Sample outcome</span><span id="outcome-state"></span></div>
              <div class="outcome">
                <div class="mini"><span>decision</span><strong id="decision"></strong></div>
                <div class="mini"><span>final pred</span><strong id="final-pred"></strong></div>
                <div class="mini"><span>attack effect</span><strong id="attack-effect"></strong></div>
                <div class="mini"><span>final correctness</span><strong id="correctness"></strong></div>
              </div>
            </section>

            <section class="live-card">
              <div class="live-title"><span>Full-test metrics</span><span id="metric-model-name"></span></div>
              <div id="metrics" class="metrics"></div>
              <p class="note">
                이 값은 전체 테스트 CSV에서 읽은 평균 결과이고, 위 이미지는 저장된 대표 샘플입니다.
              </p>
            </section>

            <section class="live-card">
              <div class="live-title"><span>Event log</span><span id="stage-name"></span></div>
              <div id="console" class="console"></div>
            </section>
          </div>
        </div>
      </section>
    </div>
  </main>
  <script>
    const SIM_DATA = __SIM_DATA__;
    const CONDITION_ORDER = ["Clean", "FGSM", "PGD"];
    const STAGES = [
      "원본 입력",
      "공격 입력",
      "Autoencoder 정화",
      "탐지와 최종 판정",
    ];
    const STAGE_NOTES = [
      "모델이 공격 없는 이미지를 어떻게 분류하는지 확인합니다.",
      "선택한 공격 조건의 입력이 들어오고 예측 변화와 L-inf 교란 크기를 봅니다.",
      "정화기가 공격 입력을 재구성해 분류기에 넣을 후보 이미지를 만듭니다.",
      "재구성 오류가 threshold를 넘는지 보고 원본 통과, 정화 후 통과, 거부 중 하나를 선택합니다.",
    ];

    const els = {
      conditionSelect: document.getElementById("condition-select"),
      sampleRange: document.getElementById("sample-range"),
      sampleLabel: document.getElementById("sample-label"),
      conditionLabel: document.getElementById("condition-label"),
      modelSelect: document.getElementById("model-select"),
      playButton: document.getElementById("play-button"),
      pauseButton: document.getElementById("pause-button"),
      prevButton: document.getElementById("prev-button"),
      nextButton: document.getElementById("next-button"),
      stepButton: document.getElementById("step-button"),
      autoButton: document.getElementById("auto-button"),
      stageScrub: document.getElementById("stage-scrub"),
      beats: Array.from(document.querySelectorAll(".beat")),
      stageFill: document.getElementById("stage-fill"),
      sceneTitle: document.getElementById("scene-title"),
      sceneSubtitle: document.getElementById("scene-subtitle"),
      decisionPill: document.getElementById("decision-pill"),
      metaModel: document.getElementById("meta-model"),
      metaSeed: document.getElementById("meta-seed"),
      metaEpsilon: document.getElementById("meta-epsilon"),
      metaThreshold: document.getElementById("meta-threshold"),
      cardOriginal: document.getElementById("card-original"),
      cardInput: document.getElementById("card-input"),
      cardPurified: document.getElementById("card-purified"),
      imgOriginal: document.getElementById("img-original"),
      imgInput: document.getElementById("img-input"),
      imgPurified: document.getElementById("img-purified"),
      attackTag: document.getElementById("attack-tag"),
      trueLabel: document.getElementById("true-label"),
      cleanPred: document.getElementById("clean-pred"),
      inputPred: document.getElementById("input-pred"),
      linf: document.getElementById("linf"),
      changed: document.getElementById("changed"),
      purifiedPred: document.getElementById("purified-pred"),
      purifyDelta: document.getElementById("purify-delta"),
      detectedLabel: document.getElementById("detected-label"),
      gaugeFill: document.getElementById("gauge-fill"),
      gaugeMarker: document.getElementById("gauge-marker"),
      score: document.getElementById("score"),
      threshold: document.getElementById("threshold"),
      detected: document.getElementById("detected"),
      outcomeState: document.getElementById("outcome-state"),
      decision: document.getElementById("decision"),
      finalPred: document.getElementById("final-pred"),
      attackEffect: document.getElementById("attack-effect"),
      correctness: document.getElementById("correctness"),
      metricModelName: document.getElementById("metric-model-name"),
      metrics: document.getElementById("metrics"),
      stageName: document.getElementById("stage-name"),
      console: document.getElementById("console"),
    };

    let stage = 0;
    let playTimer = null;
    let autoTimer = null;

    function pct(value) {
      if (value === null || Number.isNaN(Number(value))) return "NaN";
      return `${(Number(value) * 100).toFixed(2)}%`;
    }

    function fixed(value, digits = 4) {
      if (value === null || Number.isNaN(Number(value))) return "NaN";
      return Number(value).toFixed(digits);
    }

    function condition() {
      return els.conditionSelect.value;
    }

    function samples() {
      return SIM_DATA.examples.conditions[condition()].samples;
    }

    function sample() {
      return samples()[Number(els.sampleRange.value)];
    }

    function decisionClass(value) {
      if (value === "ACCEPT_PURIFIED") return "purify";
      if (value === "REJECT_SUSPICIOUS") return "reject";
      return "accept";
    }

    function setTextColor(element, kind) {
      const colors = {
        accept: "var(--green)",
        purify: "var(--amber)",
        reject: "var(--red)",
        good: "var(--green)",
        bad: "var(--red)",
        neutral: "var(--blue)",
      };
      element.style.color = colors[kind] || "var(--ink)";
    }

    function metricRow(model, currentCondition) {
      return SIM_DATA.metrics.find((row) => row.model === model && row.condition === currentCondition) || null;
    }

    function renderMetrics() {
      const model = els.modelSelect.value;
      const row = metricRow(model, condition());
      els.metricModelName.textContent = model;
      if (!row) {
        els.metrics.innerHTML = "<div class='mini metric'><span>missing</span><strong>no row</strong></div>";
        return;
      }
      const items = [
        ["original acc", pct(row.original_accuracy)],
        ["purified acc", pct(row.purified_accuracy)],
        ["detection", pct(row.detection_rate)],
        ["reject", pct(row.reject_rate)],
        ["final safe", pct(row.final_safe_accuracy)],
        ["samples", String(row.evaluated_samples)],
      ];
      els.metrics.innerHTML = items.map(([label, value]) => (
        `<div class="mini metric"><span>${label}</span><strong>${value}</strong></div>`
      )).join("");
    }

    function renderSample() {
      const current = sample();
      const all = samples();
      const index = Number(els.sampleRange.value);
      els.sampleRange.max = String(all.length - 1);
      els.sampleLabel.textContent = `sample ${index + 1} / ${all.length}`;
      els.conditionLabel.textContent = current.conditionLabel;
      els.attackTag.textContent = current.condition;

      els.imgOriginal.src = current.images.original;
      els.imgInput.src = current.images.input;
      els.imgPurified.src = current.images.purified;

      els.trueLabel.textContent = current.label;
      els.cleanPred.textContent = current.cleanPrediction;
      els.inputPred.textContent = current.inputPrediction;
      els.linf.textContent = fixed(current.perturbationLinf);
      els.changed.textContent = current.attackChangedPrediction ? "yes" : "no";
      setTextColor(els.changed, current.attackChangedPrediction ? "bad" : "good");
      els.purifiedPred.textContent = current.purifiedPrediction;
      els.purifyDelta.textContent = fixed(current.purificationMeanAbs);

      const scaleMax = Math.max(current.score, current.threshold * 1.75, 0.000001);
      els.gaugeFill.style.width = `${Math.min(100, (current.score / scaleMax) * 100)}%`;
      els.gaugeMarker.style.left = `${Math.min(100, (current.threshold / scaleMax) * 100)}%`;
      els.gaugeFill.style.background = current.detected ? "var(--amber)" : "var(--green)";
      els.score.textContent = fixed(current.score, 6);
      els.threshold.textContent = fixed(current.threshold, 6);
      els.detected.textContent = current.detected ? "yes" : "no";
      els.detectedLabel.textContent = current.detected ? "threshold exceeded" : "below threshold";
      setTextColor(els.detected, current.detected ? "bad" : "good");

      const decisionKind = decisionClass(current.decision);
      els.decision.textContent = current.decisionLabel;
      setTextColor(els.decision, decisionKind);
      els.finalPred.textContent = current.finalPrediction === null ? "reject" : current.finalPrediction;
      els.attackEffect.textContent = current.condition === "Clean"
        ? "no attack"
        : (current.attackChangedPrediction ? "prediction changed" : "prediction held");
      setTextColor(els.attackEffect, current.attackChangedPrediction ? "bad" : "good");
      els.correctness.textContent = current.correctAfterFirewall ? "correct" : "not correct";
      setTextColor(els.correctness, current.correctAfterFirewall ? "good" : "bad");
      els.outcomeState.textContent = current.correctAfterFirewall ? "safe sample" : "unsafe sample";

      els.decisionPill.textContent = current.decisionLabel;
      els.decisionPill.className = `decision-pill ${decisionKind}`;
      renderMetrics();
      renderStage(stage);
    }

    function renderStage(nextStage) {
      stage = Math.max(0, Math.min(3, nextStage));
      els.stageScrub.value = String(stage);
      els.stageFill.style.width = `${((stage + 1) / 4) * 100}%`;
      els.sceneTitle.textContent = STAGES[stage];
      els.sceneSubtitle.textContent = STAGE_NOTES[stage];
      els.stageName.textContent = `stage ${stage + 1}`;

      els.beats.forEach((beat, index) => {
        beat.classList.toggle("active", index === stage);
        beat.classList.toggle("done", index < stage);
      });
      els.cardOriginal.classList.toggle("active", stage === 0);
      els.cardInput.classList.toggle("active", stage === 1);
      els.cardPurified.classList.toggle("active", stage === 2);
      els.cardInput.classList.toggle("pending", stage < 1);
      els.cardPurified.classList.toggle("pending", stage < 2);
      els.imgInput.classList.toggle("active", stage === 1);

      const current = sample();
      const lines = [
        `[stage 1] original label=${current.label}, clean_pred=${current.cleanPrediction}`,
      ];
      if (stage >= 1) {
        lines.push(
          `[stage 2] ${current.condition} input_pred=${current.inputPrediction}, `
          + `l_inf=${fixed(current.perturbationLinf)}`
        );
      }
      if (stage >= 2) {
        lines.push(
          `[stage 3] purified_pred=${current.purifiedPrediction}, `
          + `mean_abs_update=${fixed(current.purificationMeanAbs)}`
        );
      }
      if (stage >= 3) {
        lines.push(
          `[stage 4] score=${fixed(current.score, 6)}, threshold=${fixed(current.threshold, 6)}, `
          + `detected=${current.detected}`
        );
        lines.push(
          `[decision] ${current.decision}, final=`
          + `${current.finalPrediction === null ? "reject" : current.finalPrediction}`
        );
      }
      els.console.textContent = lines.join("\\n");
    }

    function stopTimers() {
      window.clearTimeout(playTimer);
      window.clearTimeout(autoTimer);
      playTimer = null;
      autoTimer = null;
    }

    function play() {
      stopTimers();
      renderStage(0);
      function advance() {
        if (stage >= 3) return;
        renderStage(stage + 1);
        playTimer = window.setTimeout(advance, 760);
      }
      playTimer = window.setTimeout(advance, 760);
    }

    function nextSample(delta) {
      const all = samples();
      const next = (Number(els.sampleRange.value) + delta + all.length) % all.length;
      els.sampleRange.value = String(next);
      stage = 0;
      renderSample();
      play();
    }

    function auto() {
      nextSample(1);
      autoTimer = window.setTimeout(auto, 4200);
    }

    function populateControls() {
      els.conditionSelect.innerHTML = CONDITION_ORDER
        .filter((key) => SIM_DATA.examples.conditions[key])
        .map((key) => {
          const label = SIM_DATA.examples.conditions[key].label;
          return `<option value="${key}">${label} (${key})</option>`;
        })
        .join("");
      const models = Array.from(new Set(SIM_DATA.metrics.map((row) => row.model)));
      els.modelSelect.innerHTML = models.map((model) => {
        const selected = model === SIM_DATA.examples.model ? " selected" : "";
        return `<option value="${model}"${selected}>${model}</option>`;
      }).join("");
      els.metaModel.textContent = SIM_DATA.examples.model;
      els.metaSeed.textContent = SIM_DATA.examples.seed;
      els.metaEpsilon.textContent = fixed(SIM_DATA.examples.epsilon, 2);
      els.metaThreshold.textContent = fixed(SIM_DATA.examples.threshold, 6);
      els.sampleRange.max = String(samples().length - 1);
    }

    els.conditionSelect.addEventListener("change", () => {
      stopTimers();
      els.sampleRange.value = "0";
      stage = 0;
      renderSample();
      play();
    });
    els.sampleRange.addEventListener("input", () => {
      stopTimers();
      stage = 0;
      renderSample();
    });
    els.modelSelect.addEventListener("change", renderMetrics);
    els.playButton.addEventListener("click", play);
    els.pauseButton.addEventListener("click", stopTimers);
    els.prevButton.addEventListener("click", () => nextSample(-1));
    els.nextButton.addEventListener("click", () => nextSample(1));
    els.stepButton.addEventListener("click", () => {
      stopTimers();
      renderStage((stage + 1) % 4);
    });
    els.autoButton.addEventListener("click", () => {
      stopTimers();
      auto();
    });
    els.stageScrub.addEventListener("input", () => {
      stopTimers();
      renderStage(Number(els.stageScrub.value));
    });

    populateControls();
    renderSample();
    play();
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
    output_path.write_text(_simulation_html_v2(payload), encoding="utf-8")
    return output_path
