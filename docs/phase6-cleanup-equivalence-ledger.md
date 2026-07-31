# Phase 6 cleanup / equivalence ledger

이 ledger는 `prd-palletizing-pick-place-refactor.md` Phase 6의 “one smell family at a time” gate를 기록한다. LOC 감소는 목표가 아니며, safety/authority/lifecycle 의미가 불확실한 코드는 삭제하지 않는다.

## Pass P6-01 — redundant exception-union members

| 항목 | 기록 |
|---|---|
| candidate | `Box_placing/src/parcel_pose_placing/slot5_replay.py`의 세 `except` tuple에 `ValueError`와 그 하위 타입 `SessionValidationError`가 함께 기재됨 |
| classification | provably redundant exception-union syntax; control-flow와 허용되는 예외 집합은 동일 |
| proof | `issubclass(SessionValidationError, ValueError) is True`; Python의 exception matching상 `(ValueError, SessionValidationError)`와 `ValueError`는 이 경로에서 동일 |
| valid trace before | bounded static 12-frame replay SHA `a2ba539d254154dc7f8068761c9e103524f6ef0909c2eb42d55aaf2f26f1f98e`; state histogram `{"rejected:inner_opening_not_found": 12}` |
| invalid trace before | `test_missing_manifest_frame_has_a_named_frame_local_failure`와 `test_non_increasing_timestamp_has_a_named_frame_local_failure` 모두 통과; frame-local named refusal 유지 |
| action | `except` tuple에서 중복된 `SessionValidationError` member만 제거. 예외 class, raise 지점, wrap message, authority/lifecycle 코드는 유지 |
| valid trace after | bounded static 12-frame replay SHA `a2ba539d254154dc7f8068761c9e103524f6ef0909c2eb42d55aaf2f26f1f98e`; before와 exact match, state histogram도 동일 |
| invalid trace after | 두 frame-local named refusal 테스트 통과; exception type/message pattern 유지 |
| targeted/full gate | lifecycle/authority/perception-boundary 포함 targeted 73 tests exit 0; `python -m pytest -q` exit 0, 기존 경고 14개 |
| independent reviewer | **CLEAR** — `SessionValidationError(ValueError)` 상속, 세 tuple의 의미 동등성, named failure 2 tests, authority/lifecycle guard 보존을 독립 확인 |

## 검토했지만 제거하지 않은 후보

| candidate | classification | action / reason |
|---|---|---|
| placement lifecycle `_closing`, attempted flags, runtime-bound evidence identity checks | safety-relevant | **retain**; close 실패·불확실한 one-shot 재시도·forged evidence를 차단하므로 삭제 금지 |
| operation authority capability checks and pre-construction verdicts | safety-relevant | **retain**; incomplete branch constructor/command 0의 근거이므로 삭제 금지 |
| legacy `align_and_place` compatibility path | uncertain/brownfield compatibility | **retain**; 이전 public replay/perception flow와의 호출자가 남아 있어 별도 characterization 없이는 삭제하지 않음 |
| slot-5 diagnostic rejection rows with zero candidates | diagnostic evidence | **retain**; 기록이 live reference로 승격되지 못하는 이유를 명시하며 large RGB/depth array는 저장하지 않음 |

Safety-relevant 또는 uncertain 분류의 제거 수: **0**.
