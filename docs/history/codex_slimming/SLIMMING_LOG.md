# 코드 슬림화 작업 내역 (Slimming Log)

규칙은 [`CODE_SLIMMING.md`](CODE_SLIMMING.md)에 있다. 이 파일은 **누적 기록**이다.
다음 세션은 "미완료 / 다음 후보" 절부터 읽으면 이어서 작업할 수 있다.

---

## 회차 1 — 2026-07-30 (미사용 심볼 제거)

목표: 성공률·비전 인식 동작을 바꾸지 않고 참조 0건 심볼을 제거.

### 결과 요약

| 항목 | 값 |
|---|---|
| `src/parcel_pose` 줄 수 | 27,224 → **26,885** (−339) |
| 제거한 심볼 | 별칭 11개, 함수 9개, 클래스 1개, 패키지 re-export 17개 |
| 동작 변화 | **없음** (replay 수치 비트 단위 동일) |
| 테스트 | 67 passed (변화 없음) |

### 동일성 증거

슬림화 전/후 `pallet.py replay` 요약 JSON 비교. 네 지표 전부 동일.

| 세션 | acceptance | valid_frame_count | yaw_base_mean_deg | yaw_base_std_deg |
|---|---|---|---|---|
| `pallet_slot1` | SAME (True) | SAME (28) | SAME (−89.11545270690885) | SAME (0.026863460625005893) |
| `pallet_demo` | SAME (True) | SAME (34) | SAME (−85.32430316430538) | SAME (1.2653917105405268) |

`ruff check` clean, `compileall` clean, `live_view.py --help` / `record.py --help` OK,
`import parcel_pose{,.cli,.pallet_cli,.pallet_runtime,.realtime,.auto_grab}` OK.

### 제거 목록 (A급 — 참조 0건)

확인 범위: `src/parcel_pose/**`, `tests/**`, `live_view.py`, `pallet.py`, `record.py`,
`/home/kgs/workspace/.omx/verification/*.py` (32개). 정의 라인과 `__all__` 항목은
참조로 세지 않았다.

**순수 별칭 (`A = B`) 11개**

| 파일 | 제거한 이름 | 가리켰던 실체 |
|---|---|---|
| `angles.py` | `normalize_line_deg` | `normalize_line_angle_deg` |
| `angles.py` | `normalize_signed_line_deg` | `normalize_signed_line_angle_deg` |
| `angles.py` | `line_difference_deg` | `line_angle_difference_deg` |
| `angles.py` | `classify_reference_deg` | `classify_canonical_angle_deg` |
| `plane.py` | `top_plane_from_table` | `offset_plane` |
| `plane.py` | `point_plane_signed_distances` | `signed_distances` |
| `projection.py` | `ray_plane_intersections` | `intersect_rays_with_plane` |
| `projection.py` | `points_to_plane_xy` | `project_points_to_plane` |
| `projection.py` | `plane_xy_to_points` | `unproject_plane_points` |
| `rectangle_fit.py` | `fit_rectangle` | `fit_fixed_rectangle` |
| `pallet_evaluation.py` | `replay_pallet_session` | `evaluate_pallet_session` |
| `realsense_adapter.py` | `D435Adapter` | `RealSenseAdapter` |
| `visualization.py` | `draw_estimate` | `draw_pose_overlay` |

**함수·클래스 10개**

| 파일 | 심볼 | 줄 수 | 비고 |
|---|---|---|---|
| `projection.py` | `deproject_depth` | 24 | 전체 프레임 deprojection. 실제 경로는 `DepthPlaneProjector`가 슬랩 픽셀만 처리 |
| `projection.py` | `pixel_rays` | 24 | `DepthPlaneProjector`가 ray를 캐시하므로 미사용 |
| `projection.py` | `project_depth_to_plane` | 28 | `DepthPlaneProjector`의 함수형 래퍼 |
| `rectangle_fit.py` | `_line_distance` | 2 | private, 호출부 0 |
| `estimator.py` | `estimate_pose` | 20 | `ParcelPoseEstimator`의 함수형 래퍼 |
| `calibration.py` | `fit_empty_table_plane` | 13 | `fit_empty_table_plane_result`가 실제 경로 |
| `output.py` | `pose_estimate_to_json` | 2 | `dumps_strict`가 실제 경로 |
| `recording.py` | `write_session` | 11 | `SessionWriter`가 실제 경로 |
| `visualization.py` | `colorize_depth` | 11 | 오버레이 전용, 호출부 0 |
| `visualization.py` | `draw_pose_overlay` + `_as_bgr` + `_cv2` | 70 | 별칭 제거로 연쇄 사멸. `visualization.py` 122 → 32줄 |
| `burst.py` | `PoseBurstAggregator` | 17 | `aggregate_pose_burst`만 사용됨 |

**패키지 re-export**

`__init__.py` 46 → 8줄. `__all__`의 17개 이름을 쓰는 소비자가 하나도 없었다
(모든 `from parcel_pose import X`는 **모듈** 임포트였다). 부수 효과로
`import parcel_pose`가 더 이상 `.angles`/`.models`를 즉시 끌어오지 않는다.

### 제거하려다 되돌린 것 (중요)

`grabbing.py`의 `move_arms_to_mobile_ready_pose`, `move_to_camera_calibration_posture`
를 제거했다가 **복원**했다. 정적 참조는 0이지만 `auto_grab.py:236`, `auto_grab.py:513`이
`getattr(grabbing, "이름", None)`으로 **문자열 동적 호출**한다.

> 교훈: 참조 계수에서 문자열 리터럴을 빼고 세면 동적 호출을 놓친다.
> A급 판정 시 `grep -n '"심볼"'`을 반드시 함께 본다. 이후 재고조사 스크립트는
> `quoted` 수를 별도 컬럼으로 출력하도록 고쳤다.

### 의도적으로 보존한 것

| 대상 | 근거 |
|---|---|
| `pallet_models.SlotAlignmentObservation` | `.omx/verification` 하네스 1개가 임포트. 우리 소유가 아니다 |
| `PlacementDescentPlan.valid`, `.rejection_reason` | 생성자가 `valid=False`를 거부해 `if not plan.valid:`는 도달 불가(B급)지만, `verify_pallet_control_cartesian_placement_stream.py:344`가 이 필드를 **키워드 인자로 생성**한다. 제거하면 하네스가 깨진다 |
| `_finite` / `_positive` / `_nonnegative` / `_rotation_error_rad` 중복 | 완전 중복이나 총 **19줄**뿐. 공용 모듈로 옮기면 모듈 간 임포트 의존만 늘고 읽을 로직은 줄지 않는다 |
| `_linear_quantile_pair`, `_densest_fixed_window` | 주석에 Jetson 지연 근거가 있는 성능 특화 코드 |
| `PalletServoConfig`의 미배선 필드 7개 | `max_linear_acceleration_mps2`, `max_angular_acceleration_radps2`, `filter_window`, `yaw_jump_threshold_rad`, `jump_reseed_frames`, `timeout_s`, `wheel_feedback_stale_after_s`. `from_root_config`가 읽지 않아 사실상 상수지만, 하네스가 `PalletServoConfig(...)`를 키워드로 생성한다 |
| `PalletControlConfig.held_top_peak_to_peak_m` | 파지 인터록(`pallet_control.py:2634`)에서 실제로 쓰이나 `from_root_config`가 읽지 않는다. 안전 한계를 상수화/배선하는 것은 동작 변경이므로 슬림화 범위 밖 |

---

## 남은 방대함의 실체 (측정값)

제거 후 26,885줄 중 **도달 불가는 0줄**이다. 즉 "안 쓰는 코드"는 위 회차로
거의 소진됐다. 남은 크기는 기능 범위와 모듈 구조에서 온다.

| 구획 | 줄 수 | 모듈 |
|---|---|---|
| `pallet.py` 경로 전용 | 17,704 | pallet_runtime, pallet_control, pallet_geometry, pallet_acquisition, pallet_place, pallet_models, pallet_evaluation, pallet_servo, pallet_visualization, pallet_ready, pallet_control_feedback, pallet_cli |
| `live_view.py` / `record.py` 경로 전용 | 5,132 | cli, realtime, evaluation, burst, session, recording, calibration, **auto_grab, grabbing** |
| 공유 | 4,042 | models, estimator, rectangle_fit, projection, plane, angles, transforms, output, visualization, realsense_adapter, mobile_servo |

단일 모듈 상위: `pallet_control.py` 3,864 / `pallet_runtime.py` 3,251 /
`pallet_geometry.py` 1,847 / `pallet_acquisition.py` 1,827 / `mobile_servo.py` 1,330 /
`pallet_place.py` 1,331 / `pallet_models.py` 1,322.

---

## 미완료 / 다음 후보

우선순위 순. 각 항목은 **사람의 결정이 필요한 이유**를 적었다.

### 1. box-pick 자동 파지 경로 (1,620줄) — 보존 확정 (2026-07-30)

`auto_grab.py`(772) + `grabbing.py`(848)은 `live_view.py --auto-grab` 에서만
도달하므로 최대 감축 후보였다. **운영자가 계속 사용한다고 확인했으므로 제거하지
않는다.** 이후 세션은 이 경로를 "미사용"으로 재분류하지 말 것. 근거: ADR 0002의
box-pick 제어 경계 계약, `.omx/verification/verify_boxpick_headless.py`,
그리고 `grabbing`이 `importlib.import_module(".grabbing")` + `getattr` 문자열로
동적 호출된다는 사실.

### 2. `pallet_control.py` / `pallet_runtime.py` 분할 — 리팩터링(슬림화 아님)

**1단계 완료(회차 2 참조): `evaluate_grip_and_clearance_dwell` 추출 + 20 테스트.**
남은 단계는 커버리지가 더 쌓인 뒤에 진행한다.

- 2단계: 설정·DTO 계층(약 1,000줄)을 별 파일로 이동. 순수 이동이라 위험 낮음
- 3단계: 측정 상태·FK 계층, Cartesian 타깃 생성 계층
- 4단계(마지막): 스트림 소유·펌프 계층. 스레드가 얽혀 가장 위험

`pallet.py replay`는 `pallet_control.py`를 실행하지 않으므로, 이 모듈의 변경은
replay로 검증되지 않는다. 단위 테스트와 실기 확인이 유일한 증거다.
**동일성 검증이 무의미해지므로 슬림화 커밋과 섞지 말 것.**

### 3. 값이 하나뿐인 설정 필드 정리 — 동작 변경 위험

`ready_pose != ReadyPose()` → raise, `stiffness != ARM_STIFFNESS` → raise,
`filter_window != 3` → raise, `clearance_evidence_fresh_after_s != 0.30` → raise,
`PalletServoConfig.hard_upper_bounds`의 다수 항목은 상한 = 기본값.
설정 파일이 읽히지만 다른 값을 넣으면 무조건 거부된다. 같은 상수가 코드와 JSON에
이중 관리된다. 정리하면 9KB 설정 파일이 크게 줄지만 하네스 생성 계약과 얽힌다.

### 4. `SEATED` / `LOWERING` 개명 — 문자열 계약 파급

실제 동작(무하강, 접촉 검출 없음)과 이름이 어긋난다. `LOWERING_MODE` /
`RELEASE_MODE` 문자열이 `pallet_runtime`과 `ArmStreamMode`에 걸쳐 있어 파급이 크다.
현재는 ADR 0008에 의미만 기록해 둔 상태.

### 5. `assert`로 된 안전 불변식 42개 — 동작 변경

`python -O`에서 소멸한다. 특히 `pallet_servo.py`의
`assert self._zero_latched_at_s is not None` 다음 줄이 `None + float`가 된다.
명시적 `raise`로 바꾸는 것은 동작 변경이므로 별도 작업.

---

## 재고조사 재현 방법

다음 세션이 같은 판정을 재현할 수 있도록 스캐너를 남긴다.
`quoted` 컬럼이 0이 아니면 **문자열 동적 호출을 의심**해야 한다.

```bash
cd Palletizing/Codex && python3 - <<'PY'
import ast, pathlib, re, collections
root = pathlib.Path("src/parcel_pose")
files = sorted(root.glob("*.py"))
extra = [pathlib.Path(p) for p in ("live_view.py","pallet.py","record.py")] \
        + sorted(pathlib.Path("tests").glob("*.py"))
harness = sorted(pathlib.Path("/home/kgs/workspace/.omx/verification").glob("*.py"))
corpus = {p: p.read_text(encoding="utf-8") for p in files + extra}
allsrc = "\n".join(corpus.values())
hsrc = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in harness)
symbols = collections.defaultdict(list)
for p in files:
    for node in ast.parse(corpus[p]).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols[node.name].append((p, node.lineno, type(node).__name__))
        elif isinstance(node, ast.Assign) and len(node.targets)==1 \
             and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Name):
            symbols[node.targets[0].id].append((p, node.lineno, "alias"))
for name, places in sorted(symbols.items()):
    if name.startswith("__"): continue
    pat = r"\b"+re.escape(name)+r"\b"
    code = len(re.findall(pat, allsrc)) - len(places)
    quoted = len(re.findall(r"['\"]"+re.escape(name)+r"['\"]", allsrc))
    if code - quoted <= 0 and len(re.findall(pat, hsrc)) == 0:
        for p, ln, kind in places:
            print(f"{kind:11s} quoted={quoted:<2d} {p}:{ln} {name}")
PY
```

---

## 회차 2 — 2026-07-30 (파지 인터록 추출 + 커버리지)

목표: `pallet_control.py` 분할의 **1단계만**. 가장 안전 민감한 판정 로직을 순수
함수로 떼내 테스트 가능하게 만든다. 분할의 나머지는 하지 않는다.

### 왜 이것부터인가

`pallet_control.py`는 `pallet.py replay`가 **한 줄도 실행하지 않는다**(로봇 SDK
필요). 즉 이 모듈에 대해서는 replay 동일성 검증이 아무 정보를 주지 않는다.
커버리지 없이 구조를 바꾸면 조용히 다르게 동작하는 상태를 잡을 수 없고, 그
버그는 화면이 아니라 팔이 잘못 움직이는 형태로 나타난다.
따라서 **먼저 테스트를 만들 수 있게 만드는 것**이 1단계다.

`evaluate_grip_and_clearance_dwell`을 고른 이유: 383줄로 클래스 최대 메서드이고,
`self.config` / `self._state_history` / `self._ready_joint_errors`만 읽는 순수 판정이며,
2026-07-29 실기에서 로봇을 못 움직이게 만든 `arm_tracking_error`가 나오는 지점이다.

### 변경

| 항목 | 값 |
|---|---|
| `RBY1PalletController.evaluate_grip_and_clearance_dwell` | 383줄 → **63줄** |
| 신규 모듈 함수 `evaluate_grip_continuity` | 351줄 (같은 파일, 클래스 밖) |
| 신규 테스트 `tests/test_grip_continuity.py` | 20 케이스 |
| 테스트 총합 | 67 → **87 passed** |

메서드에 남은 것은 권한 검사(3개), 락 하에서의 스냅샷(`now_s`,
`cartesian_arm_motion`, `states`), 순수 함수 호출, 결과 게시(`self._grip_result`)뿐이다.

새 함수 시그니처:

```python
def evaluate_grip_continuity(
    config: PalletControlConfig,
    scene_window: Sequence[Any],
    *,
    now_s: float,
    cartesian_arm_motion: bool,
    states: Sequence[MeasuredRobotState],
    ready_joint_errors: Callable[[MeasuredRobotState], tuple[np.ndarray, bool]],
) -> GripContinuityResult
```

락도, 컨트롤러 상태도, 전송도 없다. SDK·스트림·스레드 없이 호출 가능하다.

### 동작 보존 증거

판정 로직을 **문자 단위로 이동**했다(dedent와 `self.config.`→`config.`,
`self._ready_joint_errors(`→`ready_joint_errors(` 치환만). 추출 전 파일을
`/tmp/pallet_control_pre_extract.py`로 보관해 AST로 대조했다.

```
before: method 383 lines
after : method 63 lines + pure function 351 lines
decision core identical: True (319 statements)
```

replay 동일성도 재확인(회차 1 기준선 대비):

| 세션 | acceptance | valid_frame_count | yaw_base_mean_deg | yaw_base_std_deg |
|---|---|---|---|---|
| `pallet_slot1` | SAME | SAME | SAME | SAME |
| `pallet_demo` | SAME | SAME | SAME | SAME |

단, 위 replay는 `pallet_control.py`를 실행하지 않으므로 **이 변경에 대한 증거가
아니다.** 이 변경의 증거는 위 AST 대조와 신규 20개 테스트다.

### 신규 테스트가 고정한 계약

클리어런스 하한(보수적 경계 사용), 상태 샘플 수, dwell 시간, scene 신선도,
프레임 단조성, `stack_top_source` 화이트리스트, `held_box_pose_source` 일치,
scene 샘플 수, EEF 간격 안정성, FK 결손, 사유 중복 제거.
그리고 **Cartesian arm 모드에서 ready-joint 추적 검사를 건너뛴다**는 계약을
양방향으로 고정했다(`cartesian_arm_motion=True`면 30° 오차도 통과,
`False`면 `arm_tracking_error`). 이것이 커밋 `7e581b3`의 수정 내용이며,
이제 테스트가 회귀를 막는다.

컨트롤러 진입점의 권한 계약도 함께 고정했다: 명시적 승인 인자 없이 호출 거부,
설정 플래그 없이 거부, 불리언 아닌 인자 거부, 결과가 `self._grip_result`에 게시됨.

### 작업 중 실수와 방어

첫 시도에서 `find("        now_s = self._clock()")`를 파일 전체에서 찾아
`_require_cartesian_entry_locked`의 동일 문장에 먼저 걸려 **잘못된 구간을 덮었다.**
백업 패치에서 파일을 복원한 뒤, 앵커 탐색을 AST로 얻은 메서드 범위 안으로
한정하고 "정확히 1회 일치"를 단정하도록 고쳐 재적용했다. 두 번째 시도에서는
`with self._condition:`이 메서드 안에 2회 있어 스크립트가 **파일을 쓰기 전에
중단**했고, 그 가드가 손상을 막았다.

> 교훈: 대용량 파일을 스크립트로 편집할 때 앵커는 AST로 얻은 노드 범위 안에서만
> 찾고, 일치 횟수를 단정한 뒤 쓴다. 파일 전체 문자열 탐색은 금지.

### 여기서 멈춘 이유

운영자와 합의한 범위가 1단계까지다. 다음 단계(설정·DTO 계층 분리, 스트림 소유·
펌프 계층 분리)는 `pallet_control.py`에 대한 테스트가 더 쌓인 뒤에 한다.
특히 펌프 계층은 스레드가 얽혀 있어 마지막에 해야 한다.
