# 토큰별 기울기 차이 복원: 원인 검증용 pilot

## 현재 질문과 한계

세 학생 seed에서 ToMe-KD의 native-best validation WGA가 KD보다 낮았다.
저장된 logits를 이용한 공통 median offset 보정은 그 차이를 대부분 복구하지 못했고,
병합 on의 AUC는 off보다 낮았다. **시험한 공통 offset만으로 충분하지 않았다**는 뜻이다.
다른 보정법까지 배제하거나, 기울기 손실이 원인이라고 입증한 결과는 아니다.

다음 질문: **같은 병합 순전파에서, 병합 전 토큰별로 달라야 할 기울기를 추가하면
같은 크기의 공통 기울기나 섞인 기울기보다 학습 결과가 개선되는가?**

## 모델·데이터·학습

- Waterbirds 기존 split, Teacher = 고정된 DeiT-S seed 0 best.pt.
- Student = ImageNet pretrained DeiT-Tiny, 기존과 동일한 초기화/배치/증강/최적화 설정.
- 주 경로는 3번째 block에서 이미지 토큰 196→98, CLS 보호. 병합 선택·평균·proportional attention 동일.
- 네 조건을 각각 초기 가중치에서 100 epoch 학습. seed 0 pilot부터 진행.
- 개입 강도 lambda = 1로 고정. teacher 학습이나 테스트 셋 평가는 추가하지 않는다.

## 개입 정의

병합 직전 feature를 `x`, 그 지점에서 분리한 전체 토큰 참조 경로의 CE+KD 손실 기울기를 `g`라 한다.
참조 경로는 **같은 학생의 suffix**를 FP32로 계산한다. Teacher는 기존과 같이 고정된 정답 확률을 제공한다.
참조 경로는 `x.detach()`에서 시작하며, 참조 손실은 suffix 파라미터에 직접 업데이트를 주지 않는다.
참조 dropout RNG도 별도로 보존·복구하므로 주 경로의 난수 진행을 바꾸지 않는다.

이 실험은 이전 병합이 없는 단일 병합이므로 입력 토큰 크기가 모두 1이다.
같은 병합 그룹 안의 평균 기울기를 `P(g)`라 하면,

```
r = g - P(g)
backward gradient at x = ordinary ToMe gradient + lambda * r
```

`r`의 그룹별 합은 0이다. 평균 병합을 거쳐 오는 동일한 그룹 내 공통 기울기로 표현할 수 없는 성분이다.
추가 항의 순전파 값은 정확히 0이며, 동일한 가중치에서 logits와 병합 결과는 그대로다.
개입으로 파라미터가 업데이트된 뒤의 예측과 병합 선택은 달라질 수 있다.

| 조건 | 병합 직전 추가하는 기울기 |
|---|---|
| standard | 없음. 참조 경로도 계산하는 비용 대조 기준선 |
| residual | 토큰별 차이 `r` |
| common | 병합 참여 토큰에 그룹 공통 `P(g)`를 추가. 이미지별 norm을 `r`과 맞춤 |
| shuffled | 같은 병합 그룹 안에서 `r` 벡터를 비영(非零) 순환 이동. norm·그룹별 합 보존 |

네 조건 모두 전체 토큰 참조 경로와 후보 벡터를 계산한다. 전체 계산비용이 기존 ToMe보다 증가한다.
효율화 방법의 성능 주장이 아니라 **추가 신호의 내용과 토큰별 대응이 중요한지 보는 대조 실험**이다.
shuffled는 singleton 이외 토큰이 자기 벡터를 받지 않게 한다. 두 토큰 그룹에서는 서로 교환한다.
CLS와 singleton 토큰에는 세 개입 모두 직접 추가 신호가 없다.
common norm이 0인 예외는 0 벡터로 처리하고 `common_zero_norm_fraction`으로 기록한다.

## 판정 기준

- 1차 지표: 기존과 같은 best validation-WGA, 동률이면 이른 epoch.
- 보조 지표: 동일한 100 epoch의 WGA, 전체/4개 그룹 정확도, 실제 계산 시간.
- residual이 standard뿐 아니라 common/shuffled보다 나은지 본다.
- seed 0만으로 결론내리지 않는다. 가능성이 보이면 설정을 그대로 유지해 seed 1/2를 반복한다.
- 공통 신호도 같은 효과이면 토큰별 차이의 특별한 역할을 주장할 수 없다.
- shuffled도 같은 효과이면 올바른 토큰별 대응의 역할을 주장할 수 없다.
- 개선되지 않아도 이 참조 신호/강도/개입 위치의 결과일 뿐, 모든 역전파 병목을 부정하지 않는다.
- `residual_energy_fraction`은 병합 참여 토큰의 참조 기울기 에너지 중 `r`의 비율(이미지별 계산 후 평균)이다.
  이 값이 크다는 사실 자체는 성능 손해의 증거가 아니다.
- 기존 ToMe 학습과 standard의 차이가 크면 새 개입 효과 해석 전에 재현 상태부터 점검한다.
- 새 방법의 novelty는 확정하지 않았다. 본 실험은 기전 가설을 직접 시험하는 단계다.

## 실행

```bash
git pull --ff-only
OMP_NUM_THREADS=4 nohup bash scripts/run_gradient_pilot.sh 0 \
  > runs/gradient_pilot_seed0.log 2>&1 < /dev/null &
tail -f runs/gradient_pilot_seed0.log
```

`[ALL DONE]` 후:

```bash
cat runs/gradient_rescue_seed0/summary.csv
```

네 조건은 순차 실행한다. 오류가 나면 후속 작업을 멈춘다. 같은 명령 재실행 시 기존 run은 resume한다.
학습 데이터·기존 코드·추가 구현 hash·설정·Teacher가 checkpoint에 기록된다.
구현 검증은 `tests/test_gradient_rescue.py`에 있다. CPU에서 순전파/기울기 분리, 제어 신호의 norm과
그룹 합, 참조 경로의 파라미터·난수 상태 보존, dropout을 포함한 기준선 학습 업데이트 일치를 확인한다.
서버 CUDA 실행은 실제 pilot에서 확인한다.
