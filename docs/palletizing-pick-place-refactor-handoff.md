# Palletizing pick/place refactor — software handoff

이 문서는 `.omx/plans/prd-palletizing-pick-place-refactor.md`와 test spec의 Phase 7 handoff다. 현재 결론은 **software-only complete**이며, RB-Y1/D435 실기 성공이나 hardware commissioning 완료를 뜻하지 않는다.

## 1. 수정 위치와 책임

| 바꾸려는 것 | 단일 유효 owner / 진입점 | lower-service 경계 |
|---|---|---|
| picking orientation, horizontal target, high-level arrival radius/yaw | `Box_picking/box_picking.py`: `HORIZONTAL_PICK_*`, `pick_box` | estimator/servo tuning, freshness, hysteresis, stop/stream/grasp detail은 `parcel_pose_picking`과 `Common` |
| placing slot 선택과 stage order | `Box_placing/box_pallet.py`: `_selected_slot`, `PLACING_STAGE_ORDER`, `place_box` | slot 데이터 검증은 `slot_contract.py`; controller/stream/ack는 lower service |
| slot별 시연 reference/ready/place/retreat 데이터 | `Box_placing/configs/placing_config.json#/pallet/slots/<N>` | 다른 slot/global 값으로 fallback, mirror, synthesis 금지 |
| slot-1 x/y/yaw servo와 low-level safety tuning | `placing_config.json#/servo`, `#/perception`, `#/placement`, `#/safety` | entrypoint는 선택 및 stage를 소유하고 lower service가 gate를 집행 |
| 공통 mode/readiness authority | `Common/src/parcel_pose_common/operation_authority.py` | verdict 전 robot/controller/ready/stream/sequencer construction 금지 |
| picking lifecycle evidence | `Box_picking/src/parcel_pose_picking/auto_grab.py` | exact-zero, wheel-stop, stream release, disconnect와 grasp evidence 소유 |
| placing lifecycle evidence | `Box_placing/src/parcel_pose_placing/placement_lifecycle.py` | exact-zero, wheel-stop, release, place ack, retreat ack, close retry 소유 |
| perception façade | `box_perception.py`, `pallet_perception_adapter.py` | SDK/robot/stream/command side effect 금지; `PoseResult`만 반환 |
| slot-5 offline diagnostics | `Box_placing/src/parcel_pose_placing/slot5_replay.py` | replay/dry-run authority만 허용; live reference/place/retreat를 만들지 않음 |

Picking의 공개 순서는 `preflight → authorize → initialize → ready → acquire/perceive/error/align → grasp/lift → teardown`이다. Placing의 공개 순서는 `preflight → authorize → initialize → ready → acquire/perceive/error/align → place-alignment-stop → place → ack/release → retreat → teardown`이다. 실제 command 구성과 lifecycle side effect를 entrypoint로 옮기지 않는다.

## 2. Branch별 데이터 상태

| branch | required fields | 현재 상태 | 결과 |
|---|---|---|---|
| horizontal pick live | `perception_validation`, `ready_pose`, `grasp_pose` | complete | 유일한 live pick branch |
| vertical pick live | 위 3개 | `orientation_estimator`만 존재 | pre-construction refusal |
| slot 1 live | `hole_reference`, `ready_pose`, `place_pose`, `retreat_pose` | complete | 유일한 live place branch |
| slot 2 live | 위 4개 | 모두 없음 | pre-construction refusal |
| slot 5 live | 위 4개 | `ready_pose`만 존재 | `hole_reference`, `place_pose`, `retreat_pose` 누락 refusal |
| slot 5 replay/dry-run | `rgbd_recordings`, `camera_intrinsics`, `camera_extrinsics` | complete | offline perception diagnostics only |
| slot 6 live | live 4개 | 모두 없음 | pre-construction refusal |

Slot 데이터 형식은 `README.md`의 “슬롯 추가하기”와 `slot_contract.py`가 규정한다. Slot 5의 ready 배열은 입력값을 element-wise 그대로 유지하지만, 이를 slot-1 reference/place/retreat와 결합하지 않는다.

## 3. Authority / zero-effect matrix

| branch | 허용 capability | 금지 효과 |
|---|---|---|
| horizontal pick live | demonstrated live pick | vertical data fallback 없음 |
| vertical pick live | 없음 | robot/controller/ready/stream/sequencer construction 0, dependent command 0 |
| slot 1 live | demonstrated live placement | matching lifecycle evidence 없는 place/retreat 금지 |
| slot 2/5/6 live | 없음 | robot/controller/ready/stream/sequencer construction 0, dependent command 0 |
| slot 5 replay/dry-run | `offline_perception`만 | robot connection/controller/ready/stream/sequencer 0, place/retreat actuation 0 |

미래의 live alignment-only 동작은 이 표의 예외가 아니다. 별도 mode, matrix row, ADR, 승인 없이는 추가할 수 없다.

## 4. Software-only 명령

저장소 root(`/home/kgs/palletizing_vision`)에서 실행한다.

### 도움말 및 parser

```bash
python Box_picking/box_picking.py --help
python Box_placing/box_pallet.py --help
python Box_placing/box_pallet.py slot5-replay --help
```

### Phase gate / full test

```bash
python -m pytest -q Common/tests Box_picking/tests Box_placing/tests/test_layer_boundaries.py
python -m pytest -q Box_picking/tests
python -m pytest -q Box_placing/tests/test_slot_configuration.py Box_placing/tests/test_slot1_motion_sequence.py Box_placing/tests/test_placement_sequencer.py
python -m pytest -q
```

### Bounded slot-5 dry-run (CI용)

아래 명령은 recording만 읽고 `rby1_sdk`, robot, camera connection을 만들지 않는다.

```bash
python Box_placing/box_pallet.py slot5-replay \
  --session recordings/codex_640x480/pallet_slot5 \
  --config Box_placing/configs/placing_config.json \
  --expected-frames 96 \
  --max-frames 12 \
  --dry-run \
  --output-artifact /tmp/slot5-bounded-dry-run.json \
  --overwrite
```

### Full reviewed replay audit

```bash
python Box_placing/box_pallet.py slot5-replay \
  --session recordings/codex_640x480/pallet_slot5 \
  --expected-frames 96 \
  --output-artifact docs/evidence/slot5-static-96-replay.json \
  --overwrite

python Box_placing/box_pallet.py slot5-replay \
  --session recordings/codex_640x480/pallet_slot5_moving \
  --expected-frames 938 \
  --output-artifact docs/evidence/slot5-moving-938-replay.json \
  --overwrite
```

Static artifact SHA-256은 `7a72ff7dae7e6e18c92ee58eae6449849c4a958fb65c37e9142a4752c47c0c15`, moving artifact SHA-256은 `9ea8159e660f9e0bcb22e0f4049c40858db75a4443f01295ab76f4c2ac243684`이다. 두 artifact 모두 manifest 전 프레임 load, intrinsics/extrinsics/timestamp 검증, live construction 0, place/retreat actuation 0을 기록한다.

현재 estimator 결과는 static 96프레임 모두 `inner_opening_not_found`, moving 938프레임은 `inner_opening_not_found` 507개와 `stack_plane_not_horizontal` 431개다. 따라서 **slot-5 reference candidate는 0개이며 live reference로 승격된 값도 0개**다. 이 named diagnostic은 software acceptance에는 충분하지만 hardware placement 승인 근거는 아니다.

## 5. Automated validation과 commissioning 분리

### 이 handoff가 증명하는 것

- dependency-neutral perception contract와 recursive AST boundary
- horizontal/slot-1 fake-SDK normalized trace equivalence
- lifecycle evidence, failure containment, idempotent cleanup
- incomplete branch의 pre-construction zero effect
- slot-5 manifest/replay determinism과 no-actuation
- software test exit code와 warning 상한

### 별도 supervised commissioning이 필요한 것

- 실제 RB-Y1 model/version, power state, emergency stop과 operator escape 절차
- 현재 D435 serial/profile, 장착 transform, 작업 위치의 광학·가림·조명
- payload 안정성, 양손 지지 연속성, 상자 치수 편차
- 환경/팔레트/로봇 self-collision과 workspace clearance
- 실제 wheel stop, stream ownership handoff, controller acknowledgement
- slot-1 반복 placement 정확도와 실패 복구
- slot 5의 독립 validated reference, place pose, retreat pose

`live --execute` 또는 picking 기본 실행은 실제 robot motion path이므로 이 software gate에서 실행하지 않았다. Perception-only `live`도 D435를 여는 hardware path이며 offline dry-run으로 간주하지 않는다.

## 6. Evidence

- Phase 0 inventory: `docs/inventory/palletizing_literal_ownership_phase0.json`
- Phase 5 artifacts: `docs/evidence/slot5-static-96-replay.json`, `docs/evidence/slot5-moving-938-replay.json`
- Phase 6 ledger: `docs/phase6-cleanup-equivalence-ledger.md`
- Final gate report: `docs/evidence/palletizing-refactor-software-gate.md`
