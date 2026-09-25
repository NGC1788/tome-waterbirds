# Waterbirds: ToMe 학습 신호 진단, 1차 기준선

현재 상태: 기준선 실행 코드 작성 및 로컬 CPU 테스트 완료. 실제 데이터 학습은 미실행.
원격 서버의 GPU/드라이버/데이터 경로를 확인한 후 시작한다.

## 이번 단계

1. 공식 Waterbirds 데이터와 GPU 사용 상태를 점검한다.
2. ImageNet pretrained DeiT-S teacher를 30 epoch 학습한다.
3. 같은 teacher를 고정하고 pretrained DeiT-Tiny의 CE, KD, ToMe-KD를 각각 100 epoch 학습한다.
4. 기준선의 학습/validation 그룹별 결과를 먼저 검토한다.
5. 설정을 고정한 뒤 학생 seed 0, 1, 2를 비교한다. 첫 teacher는 seed 0으로 공유한다.
6. 최종 평가 시 같은 checkpoint에서 병합 on/off를 평가한다.

단일 시드는 탐색 결과다. 결과가 가설을 지지하지 않아도 기록한다.
기울기 복원·공통 신호·셔플 대조군은 다음 단계이며, 이 실행 묶음에는 아직 없다.
FG/BG 분석용 segmentation도 기준선에는 필요하지 않다.

## 고정한 설계

- 표준 Waterbirds 95% split. group = 2*y + place, 그룹명을 명시한다.
- 교사는 전체 입력을 보고, ToMe는 학생의 3번째 block attention 뒤 / MLP 앞에서 한 번만 적용한다.
- 이미지 토큰 196→98, CLS 유지. 크기 가중 평균과 proportional attention을 사용한다.
- 공식 ToMe matching 코드를 고정 revision으로 보존했다. 단일 병합 설정은 원논문 기본 스케줄과 다른 진단용 변형이다.
- 모든 조건에서 동일한 attention kernel, 학생 초기 가중치, 샘플 순서, 이미지별 augmentation seed를 사용한다.
- AdamW, 실제 학습률 5e-5(배치 스케일링 없음), batch 32 × accumulation 4, warmup 5 epoch, cosine decay.
- CE label smoothing 0, KD temperature 2, alpha 0.5. 기존 친구의 실험을 그대로 재현하는 설정은 아니다.
- checkpoint 선택은 validation-WGA 최대, 동률이면 먼저 나온 epoch. test는 학습 중 읽지 않는다.
- 추가 epoch/tuning이 필요하면 모든 비교 조건에 같은 규칙을 적용하고 새 run 이름으로 기록한다.
- WGA<50%만으로 오류를 단정하지 않는다. 평균/클래스 macro/그룹 macro/WGA를 구별한다.
- 학습 시간에는 teacher forward가 포함된다. 별도 평가 시간은 epoch_wall_seconds와 구별한다.
- 데이터 파일/메타데이터/코드/교사 checkpoint hash와 실행 환경을 기록한다.
- CPU에서 중단 후 재개 일치를 확인했다. CUDA scatter 연산 등에 따른 비결정성은 남을 수 있다.

## 서버 환경

공개 저장소에서 코드를 받는다. GitHub 로그인이나 sudo는 필요하지 않다.

```bash
git clone https://github.com/NGC1788/tome-waterbirds.git
cd tome-waterbirds
git rev-parse HEAD
```

Python 3.12와 torch 2.8.0 / torchvision 0.23.0 / timm 1.0.20으로 로컬 검증했다.
서버에서는 NVIDIA driver와 CUDA wheel 호환성을 먼저 확인한다. 기존 conda base 환경을 변경하지 않는다.

```bash
nvidia-smi
df -h .
python --version
```

환경 확인 후 별도 환경에서 requirements.txt를 설치한다. torch의 CUDA wheel 선택은 서버 driver에 맞춘다.
아래 명령의 `python`은 해당 별도 환경의 Python을 뜻한다.

## 데이터 준비

이미 공식 데이터가 있다면 재다운로드할 필요가 없다. `DATA`는 metadata.csv가 있는 디렉터리다.

```bash
python prepare_data.py --destination data
DATA="$PWD/data/waterbird_complete95_forest2water2"
python run.py preflight --data "$DATA" --out runs/preflight.json
```

preflight는 split별 그룹 수, 이미지 존재/디코딩, 정확히 같은 파일의 split 간 중복을 검사한다.
압축을 다르게 한 근접 중복이나 콘텐츠 수준의 중복 전체를 보증하지는 않는다.

## 첫 실행: teacher

```bash
python run.py train --data "$DATA" --out runs/teacher_seed0 --method teacher --seed 0
python report.py --runs runs --out runs/report
```

완료 후 train_eval/group별 validation 곡선을 검토한다. first epoch 예산/메모리도 로그에서 확인한다.

## 학생 기준선

```bash
python run.py train --data "$DATA" --out runs/ce_seed0 --method ce --seed 0
python run.py train --data "$DATA" --out runs/kd_seed0 --method kd --seed 0 --teacher runs/teacher_seed0/best.pt
python run.py train --data "$DATA" --out runs/tome_kd_seed0 --method tome_kd --seed 0 --teacher runs/teacher_seed0/best.pt
```

같은 설정으로 seed 1, 2와 각각 다른 output 경로를 사용한다. 자동으로 모든 장시간 작업을 시작하지 않는다.
중단 후 같은 명령에 `--resume`을 붙인다. 데이터/코드/설정이 바뀌면 기존 run 재개를 거부한다.

seed 0의 CE/KD/ToMe-KD가 모두 완료된 뒤 Linux 서버에서 다음 명령을 실행하면
seed 1, 2의 학생 6개를 순차 학습하고 각 seed의 validation 병합 on/off 평가까지 수행한다.
Teacher는 기존 seed 0의 best.pt로 고정한다. 이는 **하나의 고정 Teacher에 대한 학생 seed 반복**이며,
Teacher 재학습이나 다른 데이터 split까지 포함하는 반복은 아니다.

```bash
git pull --ff-only
OMP_NUM_THREADS=4 nohup bash scripts/repeat_students.sh \
  > runs/repeat_seed12.log 2>&1 < /dev/null &
tail -f runs/repeat_seed12.log
```

실행 전 seed 0과 설정/학습 코드/torch/timm/Teacher가 일치하는지 확인한다.
실패하면 후속 작업도 중단한다. 같은 스크립트 재실행 시 기존 run은 `--resume`으로 처리한다.
`[ALL DONE]` 메시지가 모든 작업의 완료를 뜻한다. `Ctrl+C`는 tail 로그 보기만 종료한다.

## 최종 test 평가

최종 test 전에, 완료된 seed 0 기준선은 **validation에서만** 병합 on/off를 비교한다.
먼저 동일한 100 epoch checkpoint를 비교하고, 각 조건의 native validation-WGA로 선택한
best checkpoint 결과를 보조적으로 확인한다. 두 조건에서 같은 checkpoint의 가중치는 고정한다.

```bash
git pull --ff-only
OMP_NUM_THREADS=4 .venv/bin/python scripts/validate_merge.py \
  --data data/waterbird_complete95_forest2water2 \
  --runs runs --seed 0 --out runs/merge_probe_seed0
```

- KD 학습 / ToMe-KD 학습 × validation 병합 off / on의 2×2 비교다.
- summary.csv는 정확도, WGA, 4개 그룹 정확도를 기록한다.
- results.json과 prediction 파일은 같은 이미지의 정오답 전환 수와 KL(off || on, T=1)을 보존한다.
- ToMe 학습 모델의 병합을 끄는 것 자체도 학습 때와 다른 조건이다. 회복 여부만으로
  기울기 손실을 증명하거나, 두 종류의 손실을 정확하게 분해했다고 주장하지 않는다.
- 아직 단일 seed이며, seed 1/2 반복과 기울기에 직접 개입하는 대조 실험은 별도 단계다.
- 이 스크립트 추가는 기존 학습 코드와 checkpoint를 변경하지 않는다.

세 seed의 병합 probe가 끝나면, 저장된 validation logits만으로 공통 클래스 점수 이동을 진단할 수 있다.

```bash
git pull --ff-only
.venv/bin/python scripts/probe_logit_shift.py --runs runs
```

`margin = waterbird logit - landbird logit`으로 정의한다. `median_delta`는 병합 on minus off이며,
음수이면 중앙값 기준 육지새 방향으로 이동한 것이다. AUC는 판단 임계값에 무관한 클래스 순위 구분력을 본다.
`aligned_WGA`는 다른 4개 validation fold에서 추정한 **하나의 공통 median offset**을 나머지 fold에 적용한
예측들을 합쳐 계산한다. fold는 4개 그룹별로 나누며, offset fitting에는 정답이나 WGA 최적화를 사용하지 않는다.
전체 정확도·그룹별 변화·배경별 AUC·fold별 offset은 `runs/logit_shift_probe/results.json`에 기록한다.

이는 원인 분석용 대조 실험이다. 보정용 full-token 예측을 요구하며, 이미 checkpoint 선택에 사용한 validation을
재사용하므로 새로운 방법의 독립적인 성능 검증으로 주장하지 않는다. 일정한 offset으로 회복돼도 정보 보존이나
기울기 손실의 부재가 증명되지는 않는다. AUC와 그룹별 결과를 함께 읽고 test는 계속 보류한다.

학습과 설정 선택을 마친 뒤 실행한다. 학생은 같은 checkpoint로 병합 on/off 두 가지를 모두 평가한다.

```bash
python run.py evaluate --data "$DATA" --checkpoint runs/kd_seed0/best.pt --out runs/kd_seed0/evaluation
python run.py evaluate --data "$DATA" --checkpoint runs/tome_kd_seed0/best.pt --out runs/tome_kd_seed0/evaluation
```

test_results.json의 비율은 0~1이다. report의 *_percent 열은 0~100이다.
선택된 epoch, 4개 그룹 정확도/표본 수, WGA, 일반 정확도, 실행 시간, GPU peak를 함께 보고한다.
현재 테스트 숫자나 기대 성능은 제공하지 않는다.

## 구현 검증

다음 단계의 학습 개입 실험은 [토큰별 기울기 차이 복원 pilot](experiments/README.md)에 정의했다.
네 조건의 주 경로는 같은 ToMe를 사용하고, 추가 역전파 신호만 다르게 준다.
기존 학습 코드와 checkpoint는 유지하며, 이 단계는 추가 계산을 쓰는 원인 검증용이다.

```bash
python -m pytest tests -q
```

테스트는 합성 이미지/작은 모델을 사용하며, 연구 성능 결과로 사용하지 않는다.
기본 timm과 forward 및 parameter gradient 일치, ToMe의 CLS·토큰 mass·역전파,
그룹 평가, 동일 augmentation, 불균일 microbatch 누적, 교사 불변, epoch 단위 resume를 검사한다.

## 출처

- [Waterbirds / group DRO](https://github.com/kohpangwei/group_DRO#waterbirds)
- [ToMe](https://github.com/facebookresearch/ToMe), revision af95e4b1befa172dadccd8c81e223b10090f9579
- [timm](https://github.com/huggingface/pytorch-image-models), 1.0.20

vendor/tome/merge.py는 변경하지 않은 공식 코드이며 CC-BY-NC-4.0 라이선스가 적용된다.
models.py의 ToMe 연결 부분도 해당 공식 구현에 기반한다. 라이선스 전문을 함께 포함했다.
