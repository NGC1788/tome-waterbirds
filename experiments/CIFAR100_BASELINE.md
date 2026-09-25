# CIFAR-100 첫 기준선: CE와 Full KD

목표는 **증류의 이득이 재현되는 실험 환경 확보**다. 새 방법이나 novelty 실험이 아니다.
첫 실행은 seed 0의 CE/KD 두 조건만 수행한다. 결과가 나오기 전에 효과나 시간을 예측하지 않는다.

## 고정 설정

| 항목 | 설정 |
|---|---|
| 데이터 | CIFAR-100, train 50,000 / test 10,000, 32×32 |
| Teacher | 공식 ResNet32×4, 공개 epoch-240 checkpoint, 고정/eval/no_grad |
| Student | ResNet8×4, random initialization |
| CE | CE만 사용 |
| Full KD | 0.1 CE + 0.9 T² KL(teacher ∥ student), T=4 |
| 학습 | 240 epochs, batch 64, FP32, TF32/AMP 끔 |
| 최적화 | SGD, lr .05, momentum .9, weight decay .0005 |
| LR | 151/181/211 epoch 시작에 각각 ×0.1 |
| 증강 | padding 4의 RandomCrop(32), HorizontalFlip; 교사·학생에 같은 텐서 |
| 정규화 | mean (.5071,.4867,.4408), std (.2675,.2565,.2761) |
| 평가 지표 | 마지막 epoch의 test top-1, CE, 실제 학습 시간, 교사 처리 이미지 수 |

[공식 CE 설정](https://github.com/megvii-research/mdistiller/blob/a08d46f10d6102bd6e3f258ca5ac880b020ea259/configs/cifar100/vanilla.yaml),
[공식 KD 설정](https://github.com/megvii-research/mdistiller/blob/a08d46f10d6102bd6e3f258ca5ac880b020ea259/configs/cifar100/kd.yaml).
모델 소스도 이 commit으로 고정한다. 두 조건은 동일한 초기 가중치·epoch별 데이터 난수를 사용한다.
각 epoch의 첫 증강 batch와 전체 label 순서 hash를 기록해 비교한다. 라이브러리/장비 간 완전한 수치 일치를 주장하지 않는다.

## 평가 원칙

공개 교사는 train 전체에서 학습됐다고 취급한다. train 일부만 학생 validation으로 빼면
그 이미지가 교사에게는 이미 노출되어 있어 전체 증류 과정의 독립 validation이 아니다.
이번 기준선은 train 전체를 사용하고 **240 epoch 종료점을 미리 고정**한다.
훈련 과정에서 test를 읽거나 best-test checkpoint를 고르지 않는다.
두 학습이 끝난 뒤 queue가 `finish`를 호출해 교사와 두 학생의 test를 함께 평가한다.
공식 구현의 epoch별 test/best 선택과는 다르므로 발표 수치의 정확한 재현을 보장하지 않는다.
전처리 점검의 teacher accuracy는 train 첫 1,024장에 대한 진단값이며 validation/test가 아니다.

한 seed의 KD−CE 차이는 예비 결과다. 결과를 보고 hyperparameter를 바꿔 같은 test에 반복 맞추지 않는다.
후속 방법 탐색은 별도 개발용 split과 그 split을 제외하고 학습한 교사 등 독립적인 검증 설계를 먼저 정한다.
기준선 반복은 설정을 유지한 seed 1/2로 진행하고, 그 뒤 paired 차이를 보고 판단한다.

## 서버 실행

기존 `.venv`(torch 2.8 / torchvision .23)로 실행하며 추가 패키지 설치는 필요 없다.
CIFAR-100 파일과 공식 teacher archive(약 364 MB)를 처음 실행에서 내려받는다.
교사 checkpoint는 확인한 SHA-256을 고정해 검사하고, 안전한 tensor state_dict로 변환한다.
다운로드/출처 정보는 `data/cifar100/teacher_provenance.json`에 저장한다.

```bash
git pull --ff-only
mkdir -p runs/cifar100_baseline
OMP_NUM_THREADS=4 nohup bash scripts/run_cifar_baseline.sh 0 \
  > runs/cifar100_baseline/queue_seed0.log 2>&1 < /dev/null &
```

진행 확인 (Ctrl+C는 확인만 종료하며 nohup 학습은 계속됨):

```bash
tail -f runs/cifar100_baseline/queue_seed0.log
```

통합 queue 로그에 두 조건의 epoch 진행과 오류가 함께 기록된다.
조건별 `ce_seed0.log`, `kd_seed0.log`에도 각각 기록하며 오류가 나면 후속 학습을 중단한다.
중단 후 같은 queue 명령으로 재개한다. config/코드/데이터/환경이 바뀌면 재개를 거부한다.
다른 GPU 작업은 종료하지 않는다. 공용 GPU 사용 상황에 따라 시간은 달라지며 단일 순차 실행만으로 속도 우위를 결론내리지 않는다.

완료 후 공유할 파일:

```bash
cat runs/cifar100_baseline/report_seed0/summary.csv
```

`history.json`은 epoch별 train 성능·시간을, `training_complete.json`은 완료와 checkpoint hash를 기록한다.
`student_training_seconds`에는 데이터 로딩과 교사/학생 순전파·역전파가 포함된다.
다운로드·사전학습 teacher 생성·preflight·최종 test는 포함되지 않는다. 이 시간을 전체 프로젝트 비용이라고 부르지 않는다.
`wall_seconds`는 체크포인트 저장을 포함한 실행 시간이다. 재개 시 직전 저장 시간 한 번과 중단된 epoch의 비용은 누락될 수 있다.
teacher 사전학습 비용은 이미 공개된 자원을 사용하므로 측정할 수 없고, 최종 효율 연구에서 별도 명시한다.

로컬 CPU 점검은 체크포인트 호환성과 실행 경로를 확인할 뿐 서버의 실제 성능 결과가 아니다.
