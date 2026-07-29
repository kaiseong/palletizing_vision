# 코드 슬림화 행동강령 (Code Slimming Directive)

이 문서는 `Palletizing/Codex`를 줄이는 작업의 규칙이다. 목적은 단 하나다:
**성공률과 비전 인식 동작을 한 비트도 바꾸지 않고 읽을 코드의 양을 줄인다.**

슬림화는 리팩터링이 아니다. 리팩터링은 구조를 바꾸고, 슬림화는 **없어도 되는 것을
없앤다**. 둘을 한 커밋에 섞지 않는다.

## 0. 최우선 원칙

> 동작을 바꾸는 변경은 슬림화가 아니다. 그것은 별도 작업이며 별도 승인이 필요하다.

숫자, 임계값, 게이트 순서, 부동소수 연산 순서가 바뀌면 그것은 실패한 슬림화다.
"더 깔끔해 보인다"는 이유로 계산식을 재배열하지 않는다.

## 1. 절대 손대지 않는 구역

아래는 근거 없이 건드리면 로봇이 다치거나 인식률이 떨어지는 영역이다.
슬림화 대상에서 **제외**한다.

| 구역 | 이유 |
|---|---|
| 안전 게이트 상수와 검증식 | `PalletControlConfig.__post_init__`, `PlacementConfig.__post_init__`의 상한/하한은 물리 안전 계약이다 |
| 하드 리밋 | `HARD_MAX_LINEAR_SPEED_MPS`, `HARD_MAX_ANGULAR_SPEED_RADPS`, 클리어런스 플로어 `0.050 m` |
| 비전 알고리즘의 수식 | `rectangle_fit._candidate` 점수식, `plane` RANSAC, `pallet_geometry` 평면/선 피팅 |
| 결정론 장치 | 고정 seed, `kind="mergesort"`, tie-break 정렬 키 |
| 관측 불가능성 표현 | `underconstrained` / `censored` / `axis_90_ambiguous` 분기. 실패를 값으로 표현하는 설계는 자산이다 |
| fail-closed 경로 | 예외를 삼키지 않는 방향의 분기는 "안 쓰이는 것처럼 보여도" 유지 |
| 성능 특화 코드 | `_linear_quantile_pair`, `_densest_fixed_window` 등 주석에 벤치마크 근거가 있는 것 |

`_linear_quantile_pair`처럼 "np.quantile로 한 줄이면 되는데?" 싶은 코드는
Jetson 지연을 줄이려 일부러 특화한 것이다. 주석을 먼저 읽는다.

## 2. 제거를 허용하는 근거 등급

제거는 아래 등급의 **증거를 첨부해야** 한다. 인상이나 추측은 근거가 아니다.

**A급 — 참조 0건 (즉시 제거 가능)**
정적 참조가 0이고 동적 참조도 없음을 확인한 심볼.
- 확인 범위: `src/parcel_pose/**`, `tests/**`, `live_view.py`, `pallet.py`,
  `record.py`, **그리고 `workspace/.omx/verification/**`**
- `__all__` 항목과 정의 라인은 참조로 세지 않는다
- 동적 참조(`importlib.import_module`, `getattr`, 문자열 키)를 반드시 별도 확인

**B급 — 도달 불가 분기**
같은 파일 안에서 선행 조건이 항상 그 분기를 막는 것을 코드로 증명한 경우.
예: 생성자가 `valid=False`를 거부하면 `if not obj.valid:`는 도달 불가.

**C급 — 완전 중복**
두 구현이 같은 입력에 같은 출력을 내고, 한쪽이 다른 쪽의 단순 위임인 경우.
남길 쪽은 호출부가 많은 쪽으로 한다.

**D급 — 설정만 있고 읽는 코드가 없는 필드**
`from_root_config`가 읽지 않아 사용자가 조절할 수 없는 설정 필드.
제거하거나 배선한다. 둘 중 하나를 고르고 기록한다.

## 3. 외부 소비자 계약

`Palletizing/Codex`는 배포 라이브러리가 아니지만 **외부 소비자가 하나 있다**:
`/home/kgs/workspace/.omx/verification/*.py` (32개 스크립트). 이 하네스는 다른
세션이 실행하며 이 저장소가 소유하지 않는다.

- 하네스가 임포트하는 심볼은 **참조 0건이 아니다.** 제거 금지
- 하네스가 생성하는 dataclass의 필드는 제거 금지(키워드 인자 계약)
- 하네스를 고쳐서 제거를 정당화하지 않는다. 우리 소유가 아니다

제거 후보를 정할 때 반드시 실행할 확인:

    grep -rl "\bSYMBOL\b" /home/kgs/workspace/.omx/verification/*.py

## 4. 작업 순서 (위험도 오름차순)

한 단계를 끝내고 검증이 통과한 뒤에 다음 단계로 간다. 단계를 병합하지 않는다.

1. **미사용 별칭·export** — 순수 `A = B` 형태와 `__all__` 항목. 위험 최저
2. **미사용 함수·클래스** — A급 증거가 있는 것만
3. **도달 불가 분기** — B급 증거가 있는 것만
4. **중복 구현 통합** — C급. 남길 구현을 명시
5. **미배선 설정 필드** — D급. 제거 또는 배선

각 단계 후 새로 죽은 심볼이 생긴다(연쇄 사멸). 재고조사를 다시 돌린다.

## 5. 검증 절차 (매 단계 필수)

    cd Palletizing/Codex
    ruff check .
    python3 -m compileall -q src live_view.py pallet.py record.py
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests
    python3 pallet.py replay --session recordings/codex_640x480/pallet_slot1 --no-default-artifacts
    python3 pallet.py replay --session recordings/codex_640x480/pallet_demo  --no-default-artifacts

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`은 ROS humble이 `PYTHONPATH`에 있는 호스트에서
필수다(ROS pytest 플러그인이 `lark` 부재로 크래시).

### 동일성 판정 기준

replay 요약 JSON을 슬림화 **전/후로 저장해 비교**한다. 아래가 하나라도 달라지면
되돌린다.

- `acceptance.passed`
- `valid_frame_count`
- `acceptance.checks.*` 전 항목
- `yaw_base_mean_deg`, `yaw_base_std_deg`
- 프레임별 수락/거부 사유 집합

비전 수치가 비트 단위로 같아야 한다. `pytest.approx`로 넘기지 않는다.

## 6. 하지 말아야 할 것

- **큰 모듈을 쪼개는 것.** `pallet_control.py` 3,772줄은 분할 대상이지만 그것은
  리팩터링이다. 슬림화 커밋에 섞으면 동일성 검증이 무의미해진다
- **주석·docstring 삭제.** 줄 수는 줄지만 근거가 사라진다. 슬림화의 목적은
  줄 수가 아니라 **읽어야 할 로직의 양**이다
- **테스트 삭제.** 이 저장소는 한 번 163개 테스트를 잃었다. 다시 잃지 않는다
- **방어적 검증 제거.** `__post_init__`의 유효성 검사는 중복처럼 보여도 계약이다
- **`except` 범위 축소.** containment 경로의 `except BaseException`은 의도된 것
- **한 커밋에 여러 등급 섞기.** 되돌릴 단위를 잃는다

## 7. 커밋 규율

- 커밋 1개 = 근거 등급 1개 = 되돌릴 수 있는 단위
- 커밋 메시지에 제거한 심볼과 근거 등급을 적는다
- 동일성 검증 결과를 커밋 메시지에 남긴다
- 작업 내역은 `docs/SLIMMING_LOG.md`에 누적 기록한다

## 8. 중단 조건

아래에 해당하면 즉시 멈추고 사람에게 보고한다.

- replay 수치가 달라졌고 원인을 5분 안에 특정하지 못함
- 제거 대상이 안전 게이트와 얽혀 있음
- 제거하려면 `.omx/verification` 하네스를 고쳐야 함
- 워크트리가 외부 프로세스에 의해 초기화됨(이 저장소에서 실제로 발생한 사례가 있다)
