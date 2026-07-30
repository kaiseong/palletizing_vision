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

## 코드를 어디서 고치나

두 진입점이 각각 전체 시퀀스를 담고 있습니다. 모션이 잘못되면 여기부터 봅니다.

| 파일 | 함수 | 흐름 |
|---|---|---|
| `Box_picking/box_picking.py` | `pick_box` | 설정 로드 → 파지 자동화 생성 → `resolve_live_view_plan` → `watch_and_grab` |
| `Box_placing/box_pallet.py` | `place_box` | `resolve_live_plan` → `assemble_live_stack` → `initial_run_state` → `align_and_place` |

`align_and_place` 안에서 매 프레임은 `observe_pallet_frame` → `decide_base_motion`
→ `advance_placement` → 텔레메트리 → 오버레이 순서입니다. 고칠 곳은 이렇습니다.

- **자세를 바꾼다** → `pallet.slots.<N>` 설정 (아래 "슬롯 추가하기")
- **주행 판정을 바꾼다** → `pallet_runtime.decide_base_motion`
- **앉히기·손빼기를 바꾼다** → `pallet_runtime.advance_placement`
- **파지 동작을 바꾼다** → `parcel_pose_picking.auto_grab`

Containment(Ctrl-C 두 번, DANGER), 스트림 만료, 텔레메트리, 오버레이는
라이브러리에 남겨 뒀습니다. 흐름이 아니라 보장이기 때문입니다.

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

기본 실행은 인식 전용이며 로봇에 연결하지 않습니다.

```bash
cd Box_placing
python box_pallet.py live --headless
```

`--execute`를 주면 준비 자세 확인, 비전 정렬, 시연 자세로 내려놓기, 손 빼기까지
로봇에서 실행합니다. `--slot N`으로 팔레트 슬롯을 고릅니다. 아직 시연되지 않은
슬롯은 이름과 함께 거부되며 로봇은 움직이지 않습니다.

```bash
python box_pallet.py live --headless --execute --slot 1
```

커미셔닝 정책은 CLI가 아니라 설정에 있습니다. `grip_interlock.
fixed_ready_geometry_only_commissioning_enabled`, `placement.enabled`,
`placement.vision_geometry_release_enabled`가 켜져 있어야 `--execute`가 통과합니다.

### 슬롯 추가하기

`Box_placing/configs/placing_config.json`의 `pallet.slots`가 슬롯별 시연값을
담는 **유일한 위치**입니다. 슬롯 1만 채워져 있고 2·5·6은 전 항목 `null`입니다.

슬롯 하나에 필요한 값은 여섯 개입니다.

| 키 | 내용 | 단위 |
|---|---|---|
| `hole_reference` | 시연한 중앙 홀 중심·yaw·표준편차 | m, deg |
| `offset_right_far_m` | 팔레트 코너 기준 개구부 오프셋 | m |
| `long_axis` | 긴 변이 놓인 이미지 축 (`u_right` / `v_far`) | — |
| `ready_pose_rad` | 적재 준비 자세 (torso 6, 양팔 7, head 2) | **rad** |
| `place_pose_deg` | 박스를 앉히는 자세 (torso 6, 양팔 7) | **deg** |
| `retreat_pose_deg` | 손을 빼는 자세 (torso 6, 양팔 7) | **deg** |

값을 채우지 않은 슬롯을 지정하면 로봇에 닿기 전에 거부되고, 메시지가 **무엇을
어디에 어떤 형태로** 넣어야 하는지 알려줍니다.

```
$ python box_pallet.py live --headless --execute --slot 5
pallet: error: slot 5 has no demonstrated hole_reference; set
pallet.slots.5.hole_reference in the placing config to the demonstrated centre
hole, {"center_base_xy_m": [x, y], "yaw_base_deg": deg, ...}
```

준비 자세만 예외로 코드 상수(`READY_*_RAD`)와 설정 두 곳이 일치해야 합니다.
승인되지 않은 자세가 로봇에 가는 것을 막는 이중 잠금입니다.

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
