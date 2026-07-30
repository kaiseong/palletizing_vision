# D435 재녹화 가이드

이 문서는 RB-Y1 로봇 PC의 Python 3.12 환경에서 box-picking 및
palletizing용 RGB-D 데이터를 녹화하는 절차를 정리한다. 녹화 진입점은
공통 recorder인 `Common/record.py`다.

기존 `recordings/box_*` 데이터는 `1280x720` RGB 좌표계에 정합된
`float32 meter` Depth다. 시각화와 난이도 높은 장면 점검에는 계속 쓸 수
있지만, 최종 기하 추정과 정확도 검증에는 아래 형식으로 다시 녹화한다.

## 고정 조건

- 카메라: Intel RealSense D435, USB 3
- Python: 3.12
- Depth: raw Z16 `640x480 @ 30 FPS`
- RGB: raw BGR8 `640x480 @ 30 FPS`
- 박스: 닫힌 단일 박스군. 8개 실측 범위 `395-401x252-256x156-164 mm`,
  대표 prior `400x253x160 mm`
- 카메라, head, torso, 책상 높이는 모든 세션에서 고정
- 정지 세션은 박스와 로봇을 완전히 멈춘 뒤 녹화 시작
- raw Depth와 raw RGB가 기준 데이터이며, RGB-to-Depth 정합 영상은 디버그용

`-4.2 mm` front-glass 값은 기계 도면 기준점이므로 Depth나 TF에 더하지 않는다.

## 1. 실행 환경 확인

```bash
cd /home/kgs/workspace/Palletizing

python3.12 --version
python3.12 -c 'import cv2; print("opencv", cv2.__version__)'
python3.12 -c 'import pyrealsense2 as rs; print("realsense import ok")'
```

프로젝트 설치가 필요하면 Jetson에 이미 설치된 OpenCV를 우선 사용한다.

```bash
python3.12 -m pip install -e .
```

## 2. 빈 책상 녹화

박스, 팔, 손, 공구를 모두 치우고 책상만 보이는 상태에서 10초 녹화한다.

```bash
python3.12 Common/record.py \
  --output recordings/codex_640x480 \
  --session-name empty_table --duration-sec 10 \
  --robot-state-json Common/configs/rby1m_v1_2_fixed_pose.json
```

기본 config가 `640x480 @ 30 FPS`를 적용하고, `--output`이 저장 위치를
지정한다.

기본 config의 책상 ROI `[120, 190, 420, 360]`는 현재 고정된 카메라와
책상 배치에서 검증한 값이다. 카메라, head, torso 또는 책상을 움직였다면
빈 책상 영상에서 ROI부터 다시 확인한다.

책상 평면 캘리브레이션:

```bash
PYTHONPATH=Common/src:Box_picking/src python3.12 -m parcel_pose_picking.cli calibrate-plane \
  --session recordings/codex_640x480/empty_table \
  --config Box_picking/configs/d435_rby1_nominal.json \
  --robot-state-json Common/configs/rby1m_v1_2_fixed_pose.json \
  --output out/calibrations/table_plane.json
```

품질 기준을 통과하지 못하면 calibration 파일은 생성되지 않는다. 이 경우
카메라나 책상이 움직이지 않았는지 확인하고, 필요하면 책상 ROI를 지정한다.

## 3. 정지 박스 녹화 명령

한 pose마다 박스를 완전히 멈춘 뒤 10초 녹화한다.

```bash
python3.12 Common/record.py \
  --output recordings/codex_640x480 \
  --session-name box_0 --duration-sec 10 \
  --robot-state-json Common/configs/rby1m_v1_2_fixed_pose.json
```

다른 pose는 `--session-name`만 바꿔 같은 명령을 반복한다. 필요할 때만
`--annotation '{"pose_label":"center_yaw_000"}'`을 추가한다.
표기 각도는 측정 Ground Truth가 아니라 작업자가 배치한 대략적인 라벨이다.

## 4. 필수 녹화 세트

### P0: 반드시 필요

| 구분 | 권장 세션 이름 | 목적 |
| --- | --- | --- |
| 빈 책상 | `empty_table` | table plane calibration |
| 중앙 0도 | `static_center_yaw_000` | 0도 기준 |
| 중앙 약 30도 | `static_center_yaw_p030` | 일반 0도 분류 구간 |
| 중앙 약 44도 | `static_center_yaw_p044` | 45도 경계 바로 아래 |
| 중앙 약 46도 | `static_center_yaw_p046` | 45도 경계 바로 위 |
| 중앙 약 60도 | `static_center_yaw_p060` | 일반 90도 분류 구간 |
| 중앙 약 90도 | `static_center_yaw_p090` | 90도 기준 |
| 왼쪽+회전 | `static_left_yaw_p030` | XY와 yaw 복합 변화 |
| 오른쪽+회전 | `static_right_yaw_p060` | XY와 yaw 복합 변화 |
| 카메라 가까이 | `static_near_yaw_p044` | 거리 변화와 경계각 |
| 카메라 멀리 | `static_far_yaw_p046` | 거리 변화와 경계각 |
| 왼쪽 edge crop | `crop_left_center_visible` | 한 edge 누락 |
| 오른쪽 edge crop | `crop_right_center_visible` | 한 edge 누락 |
| 가까운 edge crop | `crop_near_center_visible` | 한 edge 누락 |
| 먼 edge crop | `crop_far_center_visible` | 한 edge 누락 |

Crop 세션에서도 박스 중심은 영상 안에 남겨둔다. 중심까지 영상 밖이면 해당
방향의 중심 좌표는 물리적으로 underconstrained이므로 정답을 강제하지 않는다.

### 반복성 확인

동일한 중앙 pose에서 박스를 건드리지 않고 세션을 세 번 따로 녹화한다.

- `repeat_center_00`
- `repeat_center_01`
- `repeat_center_02`

### P1: 안전성과 강건성 검증

- 테이프 반사가 강한 방향과 약한 방향
- 송장/라벨이 보이는 방향
- 조명 밝기 변화
- 팔이나 손이 박스 높이 근처에 있으나 박스와 분리된 장면
- 박스 일부를 팔이나 손이 가리는 장면
- 같은 높이의 다른 물체가 있는 장면

이 장면들은 오차가 큰 pose를 억지로 출력하는지보다, confidence와 reason을
통해 안전하게 abstain하는지 확인하기 위한 데이터다.

### 선택: 연속 움직임

- `move_xy_optional`
- `rotation_optional`
- `move_and_rotation_optional`

연속 움직임 데이터는 stress test에는 유용하지만 초기 정확도 평가나
stationary burst에는 사용하지 않는다. 초기 visual servo는 stop-and-observe
방식으로 검증한다.

## 5. 녹화 직후 무결성 검사

각 세션이 끝날 때마다 manifest, checksum, RGB/Depth frame 쌍을 검사한다.

```bash
PYTHONPATH=Common/src:Box_picking/src python3.12 -m parcel_pose_picking.cli replay \
  --session recordings/codex_640x480/static_center_yaw_000
```

캘리브레이션 후 pose 결과 생성:

```bash
PYTHONPATH=Common/src:Box_picking/src python3.12 -m parcel_pose_picking.cli replay \
  --session recordings/codex_640x480/static_center_yaw_000 \
  --calibration out/calibrations/table_plane.json \
  --config Box_picking/configs/d435_rby1_nominal.json \
  --burst-size 5 \
  --burst-min-valid 3 \
  --output-jsonl out/results/static_center_yaw_000.jsonl
```

## 6. Base 좌표 출력 조건

녹화 당시의 고정 `T_base_from_head`를 알고 있으면 JSON 객체로 저장하고
`record --robot-state-json PATH`를 추가한다. 이 정보가 없거나 독립 검증되지
않았으면 Depth/table-plane 좌표 결과만 사용하며 `*_base` 출력은 유효 처리하지
않는다.

현재 확정된 RB-Y1 M v1.2 고정 자세(torso
`[0,55,-59.988,6.532,0,0] deg`, head `[0,49.846] deg`)의 SDK FK 결과는
`Common/configs/rby1m_v1_2_fixed_pose.json`에 있다. 이 FK를 사용해도 nominal
head-camera mount가 독립 검증되기 전까지 base pose는
`nominal_unverified`로 취급한다.

카메라, 마운트, head/torso 자세, 책상 높이 또는 stream profile이 바뀌면
`empty_table`부터 다시 녹화하고 재보정한다.
