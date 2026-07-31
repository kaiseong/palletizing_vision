# Palletizing refactor software gate report

- Date: 2026-07-31
- Scope: `.omx/plans/prd-palletizing-pick-place-refactor.md` Phase 0–7
- Verdict: **PASS — software-only**
- Physical RB-Y1/D435 commissioning: **NOT RUN / NOT CLAIMED**

## Gate results

| gate | result | evidence |
|---|---|---|
| Common + picking + recursive layer boundary | exit 0 | `python -m pytest -q Common/tests Box_picking/tests Box_placing/tests/test_layer_boundaries.py` |
| Phase 3 slot/config/trace/authority | exit 0 | slot configuration, slot-1 motion/sequencer, selected-slot, slot data, authority integration tests |
| Phase 4 lifecycle/orchestration | 40 tests, exit 0 | includes close exception/False pending latch and all later lifecycle operations/actuation 0 |
| Phase 5 replay/authority | 34 tests, exit 0 | manifest, deterministic bounded replay, REPLAY/DRY_RUN zero construction/actuation |
| Phase 6 targeted equivalence | 73 tests, exit 0 | valid SHA unchanged; named invalid traces, lifecycle/authority and perception boundaries |
| pre-corrective Phase 7 baseline full suite | 320 passed, exit 0 | JUnit: tests 320, failures/errors/skipped 0; warning count exactly 14 |
| corrective explicit-entrypoint orchestration full suite | 322 passed, exit 0 | JUnit: tests 322, failures/errors/skipped 0; warnings exactly 14; architecture and safety reviewers both `CLEAR` |
| forbidden perception AST/dependency matches | 0 accepted matches | recursive adversarial boundary tests pass |
| incomplete live branches | constructors 0, dependent operations 0 | vertical and slot 2/5/6 authority integration tests pass |
| slot-5 offline branches | robot/controller/ready/stream/sequencer 0; place/retreat actuation 0 | authority tests plus replay artifact `actuation` blocks |
| documented help/parser commands | 3 commands, exit 0 | picking help, placing help, `slot5-replay --help` |
| bounded documented dry-run | exit 0, 12 frames, SHA `5e3e02786915f261495016a437bae7cb8a3c3a22cd084258476b0c553ec4db44` | all construction/actuation counts 0 |
| SDK/live-module import blocker | exit 0, blocked imports observed 0 | dry-run completed while `rby1_sdk`, `pyrealsense2`, control/ready/session imports would raise |

## Full replay evidence

| session | frames | repeatability | result |
|---|---:|---|---|
| `pallet_slot5` | 96/96 loaded | two full runs exact-equal; SHA `7a72ff7dae7e6e18c92ee58eae6449849c4a958fb65c37e9142a4752c47c0c15` | 0 candidates; 96 `inner_opening_not_found` |
| `pallet_slot5_moving` | 938/938 loaded | two full runs exact-equal; SHA `9ea8159e660f9e0bcb22e0f4049c40858db75a4443f01295ab76f4c2ac243684` | 0 candidates; 507 opening-not-found, 431 non-horizontal plane; 69 state transitions |

Both artifacts report complete manifests, valid intrinsics, mutually inverse factory extrinsics within tolerance, strictly increasing timestamps/frame numbers, and no live capability. Every façade result remains invalid with reason `slot_5_pose_unavailable`; no diagnostic row is a live reference.

## Ownership and cleanup

- Picking orientation/target/tolerance와 `acquire → perceive → x/y/yaw decision → record → loop exit → stop/release → grasp/lift` 순서는 `box_picking.py`가 직접 소유한다; picking JSON에는 중복 high-level 값이 없다.
- Placing slot 선택, alignment frame loop, stop/place, post-place release-authorization frame loop, retreat 순서는 `box_pallet.py`가 직접 소유한다. Execute와 perception-only 모두 이 flow를 사용하며, entrypoint는 `session.align`, `session.await_release_authorization`, `pallet_runtime.align_and_place`를 호출하지 않는다. Independently demonstrated slot records는 canonical config 한 곳만 사용하며 cross-slot/global fallback이 없다.
- Config keeps geometry/calibration/tuning/safety detail and bulky posture arrays.
- Phase 6 removed only redundant exception-union members. The before/after bounded artifact SHA remained `a2ba539d254154dc7f8068761c9e103524f6ef0909c2eb42d55aaf2f26f1f98e`.
- Safety-relevant or uncertain removals: 0. Independent Phase 6 reviewer verdict: `CLEAR`.

## Warning report

The only warnings are the existing 14 `RuntimeWarning: FORCED RB-Y1 stream cancellation: carried-load support continuity is not acknowledged by a successor` instances from `test_slot1_motion_sequence.py`. The accepted ceiling is 14; no new warning class or count was introduced.

## Explicit non-claims / remaining work

This report does not validate collision avoidance, payload support, D435 field optics, emergency-stop behavior, operator safety, wheel behavior, placement accuracy, or repetition on physical hardware. Slot 5 still lacks a validated hole reference, place pose, and retreat pose. Those items require separately approved supervised commissioning; they must not be inferred from replay.
