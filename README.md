# MNIST FGSM 적대적 훈련 및 전이성 분석

## 1. 프로젝트 개요

이 저장소는 Goodfellow et al.의 FGSM과 adversarial training을 기반으로 MNIST 이미지 분류 모델의 적대적 공격 취약성과 방어 성능을 분석한다. 입력 이미지는 기본적으로 `[0, 1]` 픽셀 공간을 유지하며, epsilon도 같은 공간에서 해석한다. 기본 실험에서는 mean/std normalization을 사용하지 않는다.

## 2. 연구 질문

“FGSM 적대적 훈련으로 향상된 강건성이 다른 구조의 모델에서 생성된 전이 공격과 더 강한 반복 공격인 PGD에도 유지되는가?”

비교 대상은 `LeNet`과 `SmallCNN`, 학습 방식은 `standard`와 `fgsm_at`이다. 최종 모델은 `lenet_standard`, `smallcnn_standard`, `lenet_fgsm_at`, `smallcnn_fgsm_at` 네 개다.

## 3. 설치 방법

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

`py -3.10`이 없으면 설치된 Python 3.10 이상을 사용한다. PyTorch는 Python 버전과 플랫폼에 맞는 wheel이 필요하다.

### macOS/Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 4. 가상환경 생성

가상환경은 필수는 아니지만 권장한다. CUDA 전용 PyTorch wheel을 강제로 고정하지 않았으므로, GPU 환경에서는 PyTorch 공식 설치 안내에 맞춰 먼저 `torch`와 `torchvision`을 설치해도 된다.

## 5. 의존성 설치

최소 의존성은 `requirements.txt`와 `pyproject.toml`에 정의되어 있다.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 6. quick 실행

`--quick`은 CPU에서도 전체 연결을 확인하기 위한 모드다. 학습 subset, 평가 subset, epoch, epsilon 목록, PGD step을 줄인다. 성능 결론을 내리면 안 된다.

```bash
python -m scripts.run_pipeline --quick
```

## 7. full 실행

```bash
python -m scripts.run_pipeline --seeds 42 123 2026
```

기본 full 설정은 `configs/default.json`을 사용한다. 결과는 `results/`, checkpoint는 `checkpoints/`에 저장된다.

## 8. 개별 스크립트 실행

```bash
python -m scripts.train_models --config configs/default.json
python -m scripts.evaluate_robustness --config configs/default.json
python -m scripts.run_transferability --config configs/default.json
python -m scripts.evaluate_pgd --config configs/default.json
python -m scripts.evaluate_pgd_strong --config configs/default.json --seeds 42 123 2026 --test-subset 2000 --steps 20 --restarts 5 --force
python -m scripts.plot_results --config configs/default.json
python -m scripts.build_visual_report --config configs/default.json
python -m scripts.build_simulation --config configs/default.json
python -m scripts.smoke_test
```

이미 checkpoint가 있으면 `scripts.train_models`는 기본적으로 덮어쓰지 않는다. 다시 학습하려면 `--force`를 사용한다.

## 9. 결과 파일 설명

- `results/environment.json`: Python, PyTorch, TorchVision, NumPy, OS, 장치, Git commit hash.
- `results/run_config.json`: 실제 실행에 사용된 config.
- `results/raw/training_history.csv`: epoch별 학습 및 validation 기록.
- `results/raw/clean_accuracy.csv`: 모델별 clean accuracy.
- `results/raw/fgsm_robustness.csv`: epsilon별 FGSM white-box 평가.
- `results/raw/transferability_long.csv`: source-target FGSM 전이성 long-form 결과.
- `results/raw/pgd_whitebox.csv`: PGD L-infinity white-box 평가.
- `results/raw/pgd20_restart5_whitebox.csv`: 고정 test subset 2,000개에서 수행한 PGD-20 restart 5회 white-box 평가.
- `results/aggregated/*.csv`: 여러 seed 평균과 표준편차.
- `results/figures/*.png`: robustness curve, epsilon별 transferability heatmap, clean/robust 비교 bar chart, clean accuracy retention, PGD white-box bar chart, 공격 예시.
- `results/figures/figure_index.md`: 생성된 그래프를 한 번에 훑어볼 수 있는 Markdown 인덱스.
- `results/simulation/index.html`: 저장된 Firewall 예시를 단계별로 재생하는 인터랙티브 시뮬레이터.
- `results/summary.md`: CSV에서 자동 생성한 요약 보고서.

그래프만 다시 만들고 싶으면 기존 CSV를 유지한 채 다음 명령을 실행한다.

```bash
python -m scripts.plot_results --config configs/default.json
```

## 9-0. 시각 보고서 생성

이미 학습된 checkpoint와 저장된 CSV/PNG를 사용해 발표용 시각 스토리보드를 만들 수 있다. 새 학습을 수행하지 않고 기존 결과를 읽어서 HTML과 추가 PNG를 생성한다.

```bash
python -m scripts.build_visual_report --config configs/default.json
```

주요 생성 파일:

- `results/visual_report/index.html`: 브라우저에서 열어보는 전체 시각 보고서.
- `results/figures/project_pipeline_flow.png`: 모델 학습, 공격 평가, 한계 진단, Firewall 확장 흐름도.
- `results/figures/training_progress.png`: 사전 학습된 CNN들의 epoch별 학습 경과.
- `results/figures/autoencoder_training_progress.png`: Firewall autoencoder purifier 학습 경과.
- `results/figures/result_storyboard.png`: FGSM 강건성, PGD 취약성, Firewall final safe accuracy 핵심 요약.

## 9-1. 인터랙티브 시뮬레이션 생성

`firewall_examples.pt`와 `firewall_results.csv`를 사용해 원본 입력, 공격 입력, autoencoder 정화, reconstruction-error 탐지, 최종 판정을 단계별로 재생하는 HTML을 만든다. 새 학습이나 새 평가를 수행하지 않는다.

```bash
python -m scripts.build_simulation --config configs/default.json
```

생성 파일:

- `results/simulation/index.html`: 브라우저에서 여는 standalone 시뮬레이터.

시뮬레이터에는 Clean/FGSM/PGD 조건 선택, 샘플 선택, 단계별 재생, reconstruction score와 threshold 게이지, full-test metric 비교가 포함된다. 발표에서는 정적 그래프를 먼저 보여준 뒤 이 파일로 실제 샘플 흐름을 재생하면 된다.

## 9-2. Adversarial Firewall 확장

기존 FGSM/PGD/전이성 분석은 Part 1로 유지한다. Part 2에서는 `smallcnn_standard`와
`smallcnn_fgsm_at`을 대상으로 **Adversarial Firewall: 적대적 입력 탐지·정화·거부 기반 방어 파이프라인**을 추가했다.

이 확장의 목적은 새로운 공격을 더 붙이는 것이 아니라, FGSM adversarial training이 PGD에서 무너진 조건을 바탕으로 입력 단계 방어를 구현하는 것이다.

Firewall 흐름:

```text
input image
-> classifier prediction
-> convolutional autoencoder purification
-> reconstruction error detection
-> accept original / accept purified / reject suspicious
```

Quick 검증:

```bash
python -m scripts.run_firewall_pipeline --quick --force
```

Firewall quick 모드는 test subset 1,000개, PGD 3 step, autoencoder 10 epoch를 사용한다. 성능 결론이 아니라 탐지·정화·거부 파이프라인과 시각화 산출물 연결 검증용이다.

기존 full checkpoint를 사용한 seed 42 평가:

```bash
python -m scripts.run_firewall_pipeline --force --seeds 42
```

CPU 환경에서 full test 10,000개와 PGD 10 step 평가는 오래 걸릴 수 있다. 중간 규모 검증은 다음처럼 줄여서 실행할 수 있다.

```bash
python -m scripts.run_firewall_pipeline --force --seeds 42 --test-subset 2000 --pgd-steps 3
```

개별 실행:

```bash
python -m scripts.train_autoencoder --config configs/default.json --seeds 42 --force
python -m scripts.evaluate_firewall --config configs/default.json --seeds 42 --force
python -m scripts.plot_firewall_results --config configs/default.json
```

Firewall 결과 파일:

- `checkpoints/autoencoder_seed{seed}.pt`: clean MNIST로 학습한 autoencoder purifier.
- `results/raw/autoencoder_history.csv`: autoencoder train/validation reconstruction loss.
- `results/raw/firewall_detection_scores.csv`: 샘플별 reconstruction error, threshold, decision, prediction.
- `results/raw/firewall_results.csv`: 조건별 original/purified/final safe accuracy, detection rate, reject rate.
- `results/aggregated/firewall_detection_summary.csv`: detector AUC와 TPR@FPR 5%.
- `results/aggregated/firewall_results_summary.csv`: 여러 seed 평균/표준편차용 요약.
- `results/figures/firewall_score_distribution.png`: clean/FGSM/PGD reconstruction error 분포.
- `results/figures/firewall_detection_roc_curve.png`: reconstruction error detector ROC curve.
- `results/figures/firewall_accuracy_recovery.png`: 정화 전후 및 최종 safe accuracy 비교.
- `results/figures/firewall_decision_breakdown.png`: 원본 통과/정화 후 통과/거부 비율.
- `results/figures/original_attacked_purified_examples.png`: 원본, 공격 입력, 정화 이미지 예시.

주의: 이 방어는 adaptive attack에 대한 보장을 제공하지 않는 prototype이다. quick 결과는 연결 검증용이며 성능 결론으로 사용하면 안 된다.

## 10. 지표 정의

- Clean accuracy: 공격 없는 테스트 정확도.
- Robust accuracy: 공격 이미지 테스트 정확도.
- Attack success rate: 전체 평가 샘플 중 공격 이미지에서 target 모델이 오분류한 비율.
- Conditional transfer success rate: source와 target이 원본 이미지를 모두 맞힌 샘플만 분모로 두고, source 모델로 만든 공격 이미지가 target 모델을 오분류하게 만든 비율.
- Clean accuracy retention: 같은 구조의 일반 학습 모델 대비 방어 모델의 clean accuracy 비율.

```text
clean_accuracy_retention =
    defended_model_clean_accuracy / standard_counterpart_clean_accuracy * 100
```

군 적용 참고 자료에서 제시한 작전운용성능 참고 목표치 90%는 `reference_robust_accuracy_target = 0.90`으로 기록한다. 이는 공식적·보편적 기준으로 단정하지 않고, epsilon `0.25`에서의 참고 비교값으로만 사용한다.

## 11. 전이성 행렬 읽는 방법

행은 공격을 생성한 source model, 열은 공격 이미지를 평가한 target model이다. 순서는 항상 다음과 같다.

1. `lenet_standard`
2. `smallcnn_standard`
3. `lenet_fgsm_at`
4. `smallcnn_fgsm_at`

대각선은 white-box FGSM이고, 비대각선은 transfer attack이다. `transferability_long.csv`는 각 source-target 조합별 `robust_accuracy`, `attack_success_rate`, `conditional_transfer_success_rate`를 함께 저장한다. 조건부 전이 성공률의 분모가 0이면 `NaN`으로 기록하고 경고를 남긴다.

## 12. 재현성 및 한계

코드는 `random`, `numpy`, `torch`, CUDA seed, DataLoader generator, worker seed를 설정한다. deterministic 옵션도 config에 포함되어 있다. Git 저장소가 아니면 commit hash는 `null`이 될 수 있다.

MNIST는 작고 정형화된 데이터셋이므로 실제 환경의 복잡한 이미지 분포를 대표하지 않는다. quick 모드는 연결 검증용이므로 어떠한 성능 결론도 내리면 안 된다. 테스트셋은 최종 평가에만 사용한다.

## 13. 2인 팀 역할 분담 예시

- 팀원 A: 모델 학습, checkpoint 관리, FGSM adversarial training 구현 검토.
- 팀원 B: 평가 스크립트, 전이성 행렬, PGD 평가, 시각화와 보고서 작성.

두 팀원 모두 실험 결과 조작 금지, 테스트 데이터 누수 금지, epsilon 해석 일관성을 공동 검토한다.

## 14. 향후 연구

이번 구현 범위에서 제외한 항목은 향후 연구로만 다룬다.

- CIFAR-10 등 더 복잡한 데이터셋.
- 직접 촬영 데이터.
- 군사 자산 이미지.
- 물리적 적대적 패치.
- 객체탐지 모델.
- BIM, DeepFool 등 추가 공격.
- 탐지 및 정화 방어.
- 새로운 공격 알고리즘 개발.

## 15. 문제 해결

### CUDA 없음

코드는 `CUDA -> MPS -> CPU` 순서로 장치를 자동 선택한다. CUDA가 없어도 CPU로 실행되지만 full 실험은 오래 걸릴 수 있다.

### MNIST 다운로드 실패

인터넷이 막혀 있으면 `torchvision.datasets.MNIST(download=True)`가 실패한다. 네트워크가 가능한 환경에서 `data/MNIST/`를 미리 내려받거나, 다운로드가 가능한 상태에서 다시 실행한다.

### 메모리 부족

`configs/default.json`의 `batch_size`, `test_batch_size`, `num_workers`를 낮춘다. 먼저 `--quick`으로 연결을 확인한다.

### Windows num_workers 오류

Windows에서 multiprocessing 문제가 나면 config의 `num_workers`를 `0`으로 설정한다. 이 저장소는 OS에 맞춰 안전한 기본값을 선택하며, 모든 CLI는 `main guard`를 사용한다.

## 검증 명령

```bash
ruff check .
pytest -q
python -m scripts.smoke_test
python -m scripts.run_pipeline --quick
```
