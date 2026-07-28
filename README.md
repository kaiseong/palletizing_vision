# palletizing_vision

RB-Y1 M v1.2와 헤드 고정 RealSense D435를 사용해 단일 택배 박스의
base-frame 중심과 장축 yaw를 추정하고, 선택적으로 mobile alignment와
양팔 grasp까지 수행하는 Python 3.12 구현입니다.

실행 코드와 설정은 모두 [`Codex/`](Codex/)에 있습니다. 기본 실행 명령은:

```bash
cd Codex
python live_view.py
```

실제 로봇을 움직이는 자동 파지는 명시적으로 활성화해야 합니다.

```bash
python live_view.py --auto-grab --allow-nominal-registration
```

설치, 캘리브레이션, 좌표계, 안전 조건과 전체 동작 순서는
[`Codex/README.md`](Codex/README.md)를 참고하세요. 녹화 데이터와 생성 영상은
Git에 포함하지 않습니다. 새 RGB-D 데이터를 수집할 때는
[`RECORDING_GUIDE.md`](RECORDING_GUIDE.md)를 사용하세요.
