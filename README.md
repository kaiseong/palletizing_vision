# palletizing_vision

RB-Y1 M v1.2와 헤드 고정 RealSense D435를 사용해 단일 택배 박스를
인식·파지하고, 팔레트 인터락 적재를 개발하는 Python 3.12 코드베이스입니다.

현재 구현은 세 영역으로 나뉩니다.

```text
Palletizing/
├── Box_picking/   # 테이블 위 상자 pose 추정, mobile 정렬, 양팔 grasp/lift
├── Box_placing/   # 팔레트 slot-1 정렬/배치 MVP
└── Common/        # D435 recording, calibration, geometry, serialization 공통 코드
```

## Box Picking

기본 실행은 기존 자동 파지 시퀀스를 바로 시작합니다. 기존
`--auto-grab`과 `--allow-nominal-registration` 승인 플래그는 picking
operator 명령에서 제거됐고, nominal registration 경고는 시작 시 계속 출력됩니다.

```bash
cd Box_picking
python box_picking.py
```

원격 또는 디스플레이 없는 세션에서는 창만 끕니다.

```bash
python box_picking.py --headless
```

`--config`, `--calibration`, warmup/frame/window 옵션, robot address/power,
`--output-mp4`, `--log-jsonl` 같은 진단 옵션은 유지됩니다. MP4/JSONL은
명시적으로 경로를 준 경우에만 생성되며, 기존 파일 덮어쓰기는 거부됩니다.

## Box Placing

팔레트 slot-1 정렬/배치 개발 명령은 기존 subcommand와 safety flag를 유지합니다.

```bash
cd Box_placing
python pallet.py live --headless --ensure-slot1-ready --auto-palletize-slot1
```

실제 mobile/arm 동작을 켤 때 필요한 placing 쪽 승인 플래그와 옵션은
`python pallet.py live --help`를 기준으로 확인합니다.

## Recording

새 RGB-D 데이터 수집은 공통 recorder를 사용합니다.

```bash
cd /home/kgs/workspace/Palletizing
python Common/record.py --session-name box_0 --duration-sec 10
```

자세한 녹화 절차는 [`RECORDING_GUIDE.md`](RECORDING_GUIDE.md)를 따릅니다.
녹화 데이터와 생성 영상은 Git에 포함하지 않습니다.

## Safety

이 구조 마이그레이션은 하드웨어 없는 테스트와 replay로 검증됐습니다. 물리 RB-Y1
또는 D435 실기 검증을 수행했다는 의미가 아닙니다. 실제 로봇 실행 전에는 현재
캘리브레이션, 작업공간, stream ownership, fail-closed 조건을 다시 확인해야 합니다.
