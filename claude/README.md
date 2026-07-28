# box_orient — 택배 박스 orientation (0°/90°) 추정

RB-Y1 팔레타이징용. **색에 의존하지 않는 depth 기반**으로 단일 박스의 yaw(장축 방향)를
추정하고, 대칭성을 이용해 **0° 기준 / 90° 기준 + 편차(deviation)** 로 분류합니다.

- 카메라: D435(권장, 집기 0.6~0.7m + 놓기 ~1m 커버) 또는 D405. **코드는 카메라 무관**,
  런타임에 intrinsic을 읽음. 바뀌는 건 extrinsic 캘리브레이션뿐.
- 박스: 실측 공칭 **400×253×160mm** (라벨 400×250×150, 8개 평균). CLI `--box`로 변경.
- 입력: RGB-D (depth는 color에 align, 미터 단위).

## 왜 이렇게 접근했나

기존 `box_codex/box_pose`는 **노란색 HSV 마스크**로 박스를 격리한 뒤 평면 fit을 했습니다.
새 박스는 색이 달라 그 방식이 깨집니다. 그래서:

1. depth를 3D로 deproject
2. 받침 평면(책상/팔레트)을 RANSAC로 찾음 — 법선은 카메라를 향하게 정렬
3. **평면 위 "박스 후보"(clearance ~ box_height+margin)를 넉넉히 모은 뒤, 그 안에서
   top면 평면을 다시 RANSAC** → 기울어진 평면 fit에도 top면 전체를 정확히 격리
   (색 무관, 이게 옛 방식 대비 핵심 개선)
4. top면 점들을 평면 basis에 투영 → `minAreaRect` → 장축/단축 + 실측 크기
5. 장축 yaw를 (지정 시) 로봇 T5 프레임에서 계산 → `[-45,135)`로 wrap → 0/90 분류 + 편차

사선 뷰에서 depth 임계값 분할이 깨지는 문제를 "평면 기준 높이"로 우회하고, 평면 기울기가
top면을 얇게 슬라이스하는 문제를 "2단계 top면 재fit"으로 해결합니다.

## 실행

오프라인(녹화 리플레이, 이 개발 PC에서 numpy+cv2만 있으면 됨):

```bash
python run.py --recording recordings/box_sweep --yaw-frame ref \
    --z-min 0.4 --z-max 0.9 --step 3 --save-overlay out/ --summary
```
(기본값: `--camera d435`, `--box 400x253x160`, `--yaw-frame base` — 생략 가능)

라이브(로봇 위, pyrealsense2 필요):

```bash
python run.py --live --yaw-frame base --smooth 5     # 기본: d435, 400x253x160, base
```

주요 옵션: `--z-min/--z-max`(작업거리 근처로 ROI를 좁히면 클러터에 강해짐),
`--yaw-frame camera|ref`(ref는 T5 프레임), `--smooth N`(yaw temporal 평균),
`--save-overlay DIR`(디버그 오버레이).

## 출력 (프레임별 JSON)

```json
{
  "ok": true, "reference": 90, "deviation_deg": -0.2, "yaw_deg": 89.8,
  "yaw_frame": "camera", "long_len_m": 0.517, "short_len_m": 0.335,
  "aspect": 1.54, "measured_height_m": 0.121, "seg_mode": "table_relative",
  "n_top_points": 122126, "confidence": 0.94, "reasons": [],
  "center_camera_m": [...], "long_axis_camera": [...]
}
```

- `reference`: 0 또는 90. `deviation_deg`: 그 기준으로부터의 부호 있는 편차([-45,45)).
- `--smooth N>1`이면 `"smoothed"` 필드에 시간 평균된 ref/dev/yaw 추가.

## 0°/90° 규약

장축 yaw를 mod-180으로 구한 뒤 `[-45,135)`로 wrap:
- `-45 ≤ yaw < 45` → **reference 0**, deviation = yaw
- `45 ≤ yaw < 135` → **reference 90**, deviation = yaw − 90

`ref_zero`(기본 = 참조 프레임 x축)가 0° 방향. 경계(45°)는 `OrientConfig.boundary_deg`.

## 검증 결과 (D435IF, 박스 400×253×160 실측)

실제 팔레타이징 셋업(헤드 D435IF, 박스 top ~0.6m·책상 ~0.75m, `z 0.3~1.0`) 녹화로:

- **정적 8자세(box_0~7)**: 전부 **ok 100%**, 크기 400~413 × 260~274mm, 8자세 모두 구분
  (0°, +22°, ref90 −38°, +44°(경계) …). 정지 시 dev 안정 ±1°.
- **연속 회전(box_rotation, 244f)**: yaw가 끊김 없이 추적, **0↔90 전환이 45°/135° 경계에서
  정확**. conf 0.78~0.97.
- **빈 책상(box_empty, 68f)**: 오검출 **0/68** (size 검증으로 거름).
- 오버레이: 초록 사각형·장축이 닫힌 박스 top에 정확히 정렬(conf 0.95).

이전 D405 데이터(다른 박스 505×335×195)에서도 회전 감지·크롭 강건성 확인.

## D435로 스왑 시 체크리스트

알고리즘/코드는 그대로. **intrinsic**은 카메라에서 자동으로 읽힘. **extrinsic만 재캘리브**.

✅ **D435 마운트 반영 완료**: `link_head_2 → cam = [0.049, −0.0115, 0.057, −90, 0, −90]`
(ZYX, 카메라-평면 기준) + Depth Start Point `−4.2mm`를 광학축 따라 보정.
`is_calibrated('d435') == True`. 검증: T5 광학축이 D405와 동일(25° 아래, head pitch와 일치).

1. (참고) `box_orient/extrinsics.py`의 `HEAD2_TO_D435_XYZ_RPY_ZYX_DEG` / `DEPTH_START_POINT_M`.
2. 관측 posture의 head_1 pitch를 `camera_to_t5_static(..., head1_pitch_rad=...)`로 전달.
   로봇 위에서는 `camera_to_t5_from_fk(camera, t5_from_head2)`로 FK 실시간 반영.
3. 캘리브 전까지 `--yaw-frame ref`의 T5 yaw는 bias가 있을 수 있음. **geometry/segmentation은
   extrinsic과 무관**하므로 `--yaw-frame camera`로 먼저 검증 가능.

## 튜닝 노브 (`OrientConfig`)

- `z_min/z_max`: 작업거리 근처로 ROI 제한(고정 거리이므로 강력).
- `normal_hint` + `normal_hint_tol_deg`: 고정 카메라의 예상 테이블 법선(카메라 프레임)을 주면
  벽/바닥 평면을 배제. extrinsic이 있으면 자동 계산 가능.
- `top_margin_m / top_inlier_m / band_m`: top면 높이 게이팅(밴드를 box_height에 중심 맞춤).
- `size_tol_m`, `min_aspect`: 크기/종횡비 검증(confidence·ok에 반영).

## 데이터 녹화 (self-contained)

`recording.py`가 이 레포에 포함됨(box-perception-recording-v2 포맷, `run.py`가 직접 소비).
**Jetson에서** pyrealsense2 환경으로 실행 — 정지조건(`--duration-sec`/`--max-frames`) 필수,
세션 이름은 매번 새로:

```bash
python recording.py --session-name box_r01  --duration-sec 3    # 자세별 짧은 클립
python recording.py --session-name box_sweep --duration-sec 15  # 천천히 연속 회전(권장)
python recording.py --session-name empty     --duration-sec 3   # 빈 책상
```

기본값(1280×720·30fps·depth→color align·rgb npy·depth npz·emitter ON) 그대로 두면 됨.

**녹화본을 분석 PC로**: 코드는 git으로, **녹화 데이터는 크므로(클립당 수백MB~GB) git 말고
scp/rsync 권장** — git 히스토리 비대화 방지. 그래서 `recordings/`는 `.gitignore`에 있음.
```bash
scp -r recordings/box_* <분석PC>:~/.../Palletizing/claude/recordings/
```

## 구조

```
box_orient/
  geometry.py     intrinsic, deproject, RANSAC 평면
  segment.py      2단계 top면 분할 (색 무관)
  orientation.py  yaw 추정 + 0/90 규약 + BoxOrientation
  extrinsics.py   camera->T5 (D405/D435 반영됨)
  sources.py      녹화 리플레이 + 라이브 D435
  viz.py          디버그 오버레이
run.py            CLI
recording.py      D435 녹화 (Jetson)
```
