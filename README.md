# palletizing_vision

RB-Y1 양팔로봇 팔레타이징을 위한 비전. 헤드 고정 **RealSense D435**(RGB-D)로
택배 박스의 pose를 추정하고, 나중에 놓기(push-flush)까지 지원하는 것이 목표.

- **`claude/`** — `box_orient`: **색-무관(depth 기반)** 단일 박스 orientation(**0°/90° 기준 + 편차**)
  추정 + 자체 녹화 스크립트 + CLI. 자세한 내용은 [`claude/README.md`](claude/README.md).
- (예정) **`codex/`** — 병행 구현.

각 하위 폴더는 독립 실행 가능(런타임 의존성 self-contained). 카메라 intrinsic은 런타임에 읽고,
extrinsic(camera→`link_torso_5`)은 코드에 반영됨. 녹화 데이터(`recordings/`)는 용량이 커서
git이 아니라 scp/rsync로 이동.
