# 논문 골격: FGSM 적대적 훈련의 일반화 한계와 Adversarial Firewall 방어 파이프라인

## 1. 서론

- MNIST 분류 모델을 대상으로 적대적 예제 취약성과 방어 성능을 재현 가능하게 분석한다.
- 단순 FGSM 재현을 넘어 두 모델 구조와 두 학습 방법을 교차 비교한다.
- 방어 모델의 white-box 강건성과 모델 간 전이 공격 강건성을 구분한다.

## 2. 연구 배경

### 적대적 예제

- 사람이 보기에는 원본과 유사하지만 모델의 예측을 바꾸는 입력 교란이다.
- 본 연구는 `[0, 1]` 픽셀 공간에서 L-infinity 제약을 사용한다.

### FGSM

- 손실 함수의 입력 gradient 부호를 이용하는 1-step untargeted 공격이다.
- 수식: `x_adv = clamp(x + epsilon * sign(gradient_x loss(model(x), y)), 0, 1)`.

### adversarial training

- 학습 중 현재 모델이 생성한 적대적 예제를 함께 사용해 강건성을 높이는 방법이다.
- 본 연구에서는 clean loss와 adversarial loss를 각각 `0.5`로 가중한다.

### 군 AI 적용 동기

- 군 AI 시스템은 환경 변화, 센서 잡음, 의도적 교란에 대한 안정성이 중요하다.
- 본 연구는 MNIST라는 제한된 실험 환경에서 강건성 평가 절차와 지표를 명확히 하는 데 초점을 둔다.

## 3. 연구 질문

“FGSM 적대적 훈련으로 향상된 강건성이 다른 구조의 모델에서 생성된 전이 공격과 더 강한 반복 공격인 PGD에도 유지되는가?”

## 4. 연구 방법

- 데이터셋: MNIST.
- 모델: LeNet, SmallCNN.
- 학습 방식: standard training, FGSM adversarial training.
- 최종 모델: `lenet_standard`, `smallcnn_standard`, `lenet_fgsm_at`, `smallcnn_fgsm_at`.
- 공격: FGSM, PGD L-infinity.

### Part 2: Adversarial Firewall 확장

- 기존 FGSM/PGD/전이성 결과는 모델 자체 방어의 한계를 보이는 Part 1로 둔다.
- Part 2에서는 `smallcnn_standard`와 `smallcnn_fgsm_at`을 대상으로 입력 단계 방어 파이프라인을 구현한다.
- 제안 구조:
  - convolutional autoencoder purifier/reformer.
  - reconstruction error 기반 adversarial input detector.
  - 정화 후 재분류.
  - `ACCEPT_ORIGINAL`, `ACCEPT_PURIFIED`, `REJECT_SUSPICIOUS` reject option policy.
- 핵심 연구 질문:
  - FGSM adversarial training이 PGD에서 일반화되지 않는 조건에서, 입력 탐지·정화·거부 정책이 오분류 위험을 얼마나 줄일 수 있는가?
  - clean 입력의 오탐률을 제한하면서 공격 입력을 탐지할 수 있는가?
  - autoencoder 정화가 FGSM/PGD 입력에서 분류 정확도를 회복시키는가?

## 5. 실험 환경

| 항목 | 값 |
|---|---|
| Python | 3.14.0 |
| PyTorch | 2.12.1+cpu로 기본 결과 생성, PGD-20 restart 5 full 재평가는 2.12.1+cu126 |
| TorchVision | 0.27.1 |
| 장치 | 기본 학습/평가 및 firewall: CPU, PGD-20 restart 5 full 재평가: CUDA GTX 1660 Ti |
| seed | 42, 123, 2026 |

## 6. 평가 지표

- Clean accuracy.
- Robust accuracy.
- Attack success rate.
- Conditional transfer success rate.
- Clean accuracy retention.
- 군 적용 참고 자료[7]에서 제시한 작전운용성능 참고 목표치 90%와 epsilon `0.25` robust accuracy 비교.

## 7. 실험 결과

| 모델 | Clean accuracy | FGSM epsilon=0.25 robust accuracy | Clean accuracy retention |
|---|---:|---:|---:|
| lenet_standard | 98.35% | 2.81% | N/A |
| smallcnn_standard | 99.25% | 28.70% | N/A |
| lenet_fgsm_at | 97.52% | 87.57% | 99.16% |
| smallcnn_fgsm_at | 99.19% | 96.65% | 99.94% |

추가 PGD 평가 결과는 다음과 같다.

| 모델 | PGD-10 epsilon=0.25 robust accuracy | PGD-20 restart 5 full-test robust accuracy |
|---|---:|---:|
| lenet_standard | 0.85% | 0.51% |
| smallcnn_standard | 0.94% | 0.39% |
| lenet_fgsm_at | 15.39% | 12.33% |
| smallcnn_fgsm_at | 10.07% | 5.54% |

## 8. 전이성 분석

- 행은 source model, 열은 target model이다.
- 순서: `lenet_standard`, `smallcnn_standard`, `lenet_fgsm_at`, `smallcnn_fgsm_at`.
- 대각선은 white-box FGSM, 비대각선은 transfer attack이다.

| Source / Target | lenet_standard | smallcnn_standard | lenet_fgsm_at | smallcnn_fgsm_at |
|---|---:|---:|---:|---:|
| lenet_standard | 97.14% | 31.17% | 28.85% | 45.46% |
| smallcnn_standard | 21.17% | 71.09% | 19.20% | 28.57% |
| lenet_fgsm_at | 38.10% | 30.70% | 10.85% | 29.06% |
| smallcnn_fgsm_at | 6.48% | 22.98% | 12.50% | 2.72% |

### Adversarial Firewall 결과

Firewall 평가는 `smallcnn_standard`, `smallcnn_fgsm_at`에 대해 seed 42, 123, 2026과 full test 10,000개 기준으로 수행하였다.

| 모델 | 조건 | Original accuracy | Purified accuracy | Final safe accuracy |
|---|---|---:|---:|---:|
| smallcnn_standard | Clean | 99.25% | 98.77% | 99.07% |
| smallcnn_standard | FGSM | 28.47% | 75.11% | 83.73% |
| smallcnn_standard | PGD | 0.99% | 84.84% | 89.15% |
| smallcnn_fgsm_at | Clean | 99.19% | 98.58% | 98.90% |
| smallcnn_fgsm_at | FGSM | 96.65% | 93.61% | 96.58% |
| smallcnn_fgsm_at | PGD | 10.00% | 89.72% | 93.62% |

Reconstruction error detector는 본 MNIST 및 non-adaptive FGSM/PGD 조건에서 3-seed 평균 AUC가 사실상 1.0이며, TPR@FPR 5%가 100%로 기록되었다. 이는 현재 공격 설정에서 clean 입력과 공격 입력의 reconstruction error 분포가 거의 분리되었음을 의미하지만, adaptive attack에 대한 보장을 의미하지 않는다.

## 9. 논의

- FGSM 적대적 훈련이 white-box FGSM에는 강건성을 높이는지 분석한다.
- 동일 방어가 전이 공격에도 유지되는지 source-target 방향성을 구분해 해석한다.
- FGSM 적대적 훈련이 PGD에도 일반화되는지 추가 검증한다.
- 정상 정확도와 강건 정확도의 trade-off를 정량화한다.
- Adversarial Firewall 결과에서는 단순 정화 성능뿐 아니라 탐지율, 오탐률, 거부율, final safe accuracy를 함께 해석한다.
- autoencoder purifier가 PGD 정확도를 완전히 회복하지 못하더라도, reject option이 고위험 입력의 무리한 자동 판단을 줄이는지 분석한다.

## 10. 군 적용 시사점

- 작전 환경의 AI는 clean 성능뿐 아니라 입력 교란 하 성능을 함께 보고해야 한다.
- 본 실험의 90% 값은 군 적용 참고 자료[7]에서 제시한 작전운용성능 참고 목표치로만 사용하며, 공식적·보편적 기준으로 단정하지 않는다.
- MNIST 결과를 실제 군사 자산 이미지나 물리 환경으로 직접 일반화하지 않는다.

## 11. 한계

- MNIST는 실제 시각 인식 환경보다 단순하다.
- FGSM와 PGD만 사용한다.
- Adversarial Firewall은 adaptive attack에 대한 보장을 제공하지 않는 prototype이다.
- reconstruction error detector와 autoencoder purifier는 공격자가 방어 구조를 알고 최적화하는 경우 우회될 수 있다.
- 물리 패치, 직접 촬영 데이터, 객체탐지는 다루지 않는다.
- quick 모드 결과는 성능 결론에 사용할 수 없다.

## 12. 향후 연구

- CIFAR-10 확장.
- 직접 촬영 데이터.
- 군사 자산 이미지.
- 물리적 적대적 패치.
- 객체탐지 모델.
- BIM, DeepFool 등 추가 공격.
- adaptive attack에 대한 Adversarial Firewall 강건성 평가.
- transfer attack에 대한 Firewall 평가.
- Feature squeezing, prediction disagreement 등 다중 detector ensemble 비교.
- 새로운 공격 알고리즘 개발.

## 13. 결론

- Part 1에서는 FGSM adversarial training이 동일한 FGSM 공격에는 높은 강건성을 보였지만, PGD와 모델 간 전이 공격에는 안정적으로 일반화되지 않음을 확인하였다.
- Part 2에서는 이 한계를 보완하기 위해 Adversarial Firewall을 구현하고, 입력 탐지·정화·거부 정책이 공격 상황의 자동 오분류 위험을 줄일 수 있음을 3-seed full test 결과로 확인하였다.
- 최종적으로 본 프로젝트는 단일 모델 방어가 아니라 모델 수준 방어와 입력 단계 방어를 결합한 다층 방어 구조의 필요성을 보여준다.

## 참고문헌

[7] 이승민, 「인공지능(AI) 적대적 공격 및 적대적 공격에 대한 방어 기술 동향과 군 적용 발전방안」, 국방논단.
