# OpticalNav 편광 시뮬레이터 인수인계 (2026-10-05)

다른 PC(실시간 측정은 Device 1 · RTX 5090을 권장)에서 이어서 작업하기 위한 문서입니다. 사용법 전체는 `README.md`, 결과 보고서는 아티팩트 <https://claude.ai/artifact/J4asmZykf9eoKcpTB2SSpz>에 있습니다.

## 1. 무엇이 어디에 있나

| 항목 | 위치 | 비고 |
|---|---|---|
| 소스 (git) | `/jarvis/project/opticalnav_sim` | `master` 브랜치. NAS 경로라 같은 마운트가 있으면 그대로 쓰면 됩니다. 없으면 `git clone /jarvis/project/opticalnav_sim` |
| 씬 팩 | `packs/opticalnav-v0.2` | git에 없음(`.gitignore`). 71 GB(겉보기), 파일 60,740개. 16개 씬. 에셋은 원본 robomituba 파일의 하드링크라 NAS에서는 추가 용량이 거의 없습니다. 단, 아래 경고를 보세요 |
| 보고서 아티팩트 | 위 링크 | 같은 URL로 갱신하려면 그 PC의 Claude 세션에서 `url`을 지정해 publish |
| 측정 원자료 (Device 2) | `/tmp/claude-0/…/scratchpad/` | 로컬 임시 디렉터리. 필요한 수치는 이 문서와 보고서에 옮겨 두었습니다 |

팩을 다른 PC로 옮길 때는 둘 중 하나를 씁니다.

- NAS가 같은 경로로 마운트되어 있으면 아무것도 복사하지 않습니다.
- 아니면 `rsync -aH packs/opticalnav-v0.2 <pc>:…`로 옮깁니다. 원본과의 하드링크가 실제 파일로 풀려 약 71 GB가 됩니다. `-H`는 팩 안에서 서로 링크된 에셋을 하나로 유지합니다. 일부 씬만 필요하면 `scenes/<scene>/`, `connectivity/<scene>_connectivity.json`, `tasks/`만 옮겨도 됩니다(씬당 1–9 GB).

> **경고: 팩을 지우거나 다시 만들지 마세요.**
> - 13개 씬의 정합성 기준 프레임(`scenes/<scene>/reference/`, 씬당 9뷰)은 데이터셋 렌더에서 링크해 온 것입니다. 그중 12개 씬은 원본 렌더가 그 뒤 지워져 링크 수가 1입니다. 즉 팩 안의 파일이 유일한 사본입니다(20260811만 원본이 남아 있음).
> - `build_pack.py`로 다시 만들면 이 기준 프레임을 다시 얻을 수 없으니 복사만 하세요.
> - 20262004와 추정 팩 3개(아래 6절)는 원래부터 기준 프레임이 없습니다. 이 씬들은 `check_parity.py`를 돌릴 수 없습니다.

## 2. 구성 요약

- `opticalnav_sim/MatterSim.py`: Matterport3DSimulator와 같은 API(`newEpisode`, `makeAction`, `getState`, `navigableLocations`, 이산 시야각). 확장: `setVariant`, `setRenderMode`, `setRenderSpp`, `setDenoiser`, `teleport`, `renderCamera`.
- `opticalnav_sim/server.py`: HTTP 렌더 서버(기본 포트 18770). 씬을 한 번 올려 상주시키고 요청마다 카메라만 옮깁니다. 모든 Mitsuba 호출은 메인 스레드에서 실행합니다(변형·파일 리졸버가 스레드 로컬이라 작업 스레드에서 RGB 씬 로드가 segfault 났습니다). 같은 설정의 연속 뷰는 `batch` 센서로 한 번의 호출에 렌더합니다. Dr.Jit에 `freeze`가 있으면 자동으로 씁니다.
- `opticalnav_sim/frames.py`: 데이터셋 ↔ R2R 좌표, 프로덕션 리그 카메라 행렬(데이터셋 행렬을 1e-9 미만 오차로 재현).
- `tools/`: `build_pack.py`(팩 생성), `check_parity.py`(데이터셋 프레임 대비 검증), `frame_overhead.py`(프레임 시간 분해), `benchmark_modes.py`(모드·spp·디노이저별 fps), `batch_scaling.py`(호출당 뷰 수 K에 따른 비용), `sample_frames.py`(예시 이미지), `record_episode.py`(실시간 주행 MP4), `run_all_checks.sh`(위 전부를 순서대로).
- `examples/run_agent.py` + `python -m opticalnav_sim.eval`: R2R 형식 평가 루프와 지표(NE, SR, OSR, SPL).

## 3. 다른 PC에서 할 일 (순서대로)

### 3.1 환경 확인

```bash
cd /jarvis/project/opticalnav_sim
# Device 1 예시 (CLAUDE.md 기준)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
source /home/jinnyeong/robomituba-build/mitsuba3/setpath.sh
python3 -c "import mitsuba as mi, drjit as dr, numpy, PIL; print(mi.variants(), 'freeze', hasattr(dr, 'freeze'))"
```

필요한 것:

- polar 모드: `cuda_rgb_polarized`(없으면 `cuda_ad_rgb_polarized`를 씁니다).
- rgb 모드: `cuda_rgb` 또는 `cuda_ad_rgb`.
- 실시간: `freeze True`. Device 1의 Mitsuba 3.7 빌드에는 있고, Device 2(Dr.Jit 0.4)에는 없습니다.
- `active_polar`: robomituba Mitsuba 포크의 `polarized_area`와 `path_nocaustics` 플러그인. 서버 시작 줄의 `path_nocaustics=True`로 확인합니다.

### 3.2 단위 테스트 (GPU 불필요)

```bash
python3 tests/test_sim.py      # 6개 테스트. 팩의 기준 프레임 108개로 카메라 행렬 재현을 확인
```

### 3.3 한 번에 전부 돌리기

```bash
CARD=0 PY=python3 LIMIT=3h tools/run_all_checks.sh infinigen_apartment_natural_v1_20268504 runs/device1
```

polar 서버(18770)와 rgb 서버(18771)를 차례로 띄워 정합성, 프레임 시간 분해, 벤치마크, 배치 스케일링, 예시 이미지, 영상을 만들고 `runs/device1/`에 남깁니다. 서버는 `LIMIT`이 지나면 스스로 종료됩니다.

### 3.4 직접 써 보기

```bash
CUDA_VISIBLE_DEVICES=0 timeout 3h python3 -m opticalnav_sim.server --pack packs/opticalnav-v0.2 \
    --port 18770 --modes polar --preload infinigen_apartment_natural_v1_20268504:base:polar
```

시작 줄에서 `modes={'polar': 'cuda_rgb_polarized'}`, `freeze=True`를 확인합니다. 클라이언트는 numpy만 있으면 됩니다.

```python
import math
from opticalnav_sim import MatterSim
sim = MatterSim.Simulator()
sim.setDatasetPath("http://127.0.0.1:18770")
sim.setNavGraphPath("packs/opticalnav-v0.2/connectivity")
sim.setDiscretizedViewingAngles(True)
sim.setRenderSpp(128)
sim.initialize()
sim.newEpisode(["infinigen_apartment_natural_v1_20268504"], ["support_pose_00085"], [math.radians(90)], [0.0])
st = sim.getState()[0]          # st.rgb (BGR), st.stokes["s0".."s3"], st.navigableLocations
sim.makeAction([0], [1], [0])   # 오른쪽 30도
print(sim.timingInfo())
```

다른 머신에서 접속하려면 SSH 터널(`ssh -L 18770:127.0.0.1:18770 <host>`)을 쓰거나 `--host 0.0.0.0 --token <비밀>`로 띄우고 클라이언트에 `OPTICALNAV_SIM_TOKEN`을 줍니다. 토큰 없이는 localhost 밖으로 열리지 않습니다.

## 4. 기대값 (Device 2 · RTX 3090 측정)

씬 상주 상태, 512×384 기준입니다. freeze가 있으면 프레임 시간 ≈ GPU + 후처리입니다.

| 항목 | rgb (`cuda_rgb`) | polar (`cuda_ad_rgb_polarized` · `cuda_rgb_polarized`) |
|---|---|---|
| 씬 로드 (1회) | 27.7 s | 25–35 s |
| CPU 트레이싱 / 패스 | 8.2–9.3 s (64×48 · 1 spp도 8.9 s) | 40.8 s (AD) · 41.3 s (non-AD) |
| 코드 생성 / 패스 | 1.7–1.9 s | 8.0 s (AD) · 8.5 s (non-AD) |
| 컴파일 | 새 해상도·커널에서만 3.5 s | 첫 회만 (non-AD 23.8 s) |
| GPU / 128 spp | 0.56 s (spp에 정비례) | 2.33 s (AD) · 2.31 s (non-AD) |
| freeze 예상 128 spp | 0.63 s · 1.6 fps | 2.44–2.46 s · 0.4 fps |
| S0 오차 vs 1024 spp 데이터셋 | 128: 4.6% · 256: 3.46% · 512: 2.71% | 같음 (1024: 2.23%) |

비교 기준:

- **Device 1 프로덕션:** `cuda_rgb_polarized` + freeze + 8–12 뷰 배치로 1024 spp에서 2.6–3.9 s/뷰(20268504 2.84 s, 20262004 3.89 s, 20270704 2.61 s).
- **Device 1에서 확인할 것:** 두 번째 프레임부터 트레이싱·코드 생성이 거의 0이 되는지, 그리고 GPU 시간이 3090보다 얼마나 짧은지.

## 5. 검증된 것과 아닌 것

검증됨 (Device 2):

- 카메라 행렬 재현 108/108.
- base 뷰 정합성: S0 평균 0.18925 vs 0.18922, 8×8 블록 상관 S0 0.99998 · S1 0.954 · S2 0.925.
- 잡음 모델(시뮬레이터 잡음 4.3%·√(128/spp)과 데이터셋 프레임 잡음 1.67%)이 512·1024 spp 오차를 예측.
- 최단 경로 에이전트 SR 1.00, SPL 1.00 (val_unseen 123개).
- rgb 모드 radiance와 polar S0의 평균 차이 0.095%.
- AD와 non-AD가 트레이싱 비용·결과 모두 같음.
- 디노이저: 128 spp 이상에서 프레임 변화 0.5%, 오차 개선 없음(합성 잡음에서는 정상 동작).
- 배치 센서 경로: CPU `scalar_rgb`에서 슬라이스가 단일 렌더와 일치(상관 0.985–0.993), 슬롯 포즈 갱신 확인.

검증 안 됨 (다른 PC에서 확인 필요):

- **freeze 경로 전체** (`Resident.render`, `Resident.render_many`의 freeze 분기). 배치 freeze는 센서를 인자로 넘기도록 고쳤지만 실행해 보지 못했습니다. 프로덕션은 배치 센서를 장면 안에 둡니다. 재생된 프레임이 포즈를 따라가는지 `frame_overhead.py`의 "new view" 결과와 이미지로 꼭 확인하세요.
- **`active_polar`:**
  - 2-pass(`flash_nocaustics_v1`)는 `path_nocaustics`가 필요해 Device 2에서는 못 돌렸습니다.
  - 1-pass(`compact_flash_v2`)는 3090에서 64 spp에도 메모리가 넘쳤습니다.
  - `check_parity.py`가 기본으로 모든 변형을 검사합니다.
- **GPU에서의 배치 렌더(K>1)와 스케일링 수치.** `batch_scaling.py` 결과로 트레이싱이 K에 따라 거의 일정한지 보세요. 일정하면 뷰당 비용이 1/K로 줄어듭니다.
- **한 프로세스에서 두 모드**(`--modes polar,rgb`): 메인 스레드 전환으로 바꾼 뒤 시험하지 않았습니다. 모드별로 서버를 따로 띄우는 것을 권장합니다.

## 6. 주의할 점

- **팩 품질:**
  - 20260827 · 20260904 · 20260828_lit_texturecan은 관측 매니페스트가 지워져 staged XML을 객체 id로 맞춘 팩이라 미검증입니다(`scene.json`의 `variants.<v>.inferred`).
  - 20273004의 base는 원본 장면 파일이 지워져 없습니다(`variants_unavailable`). perturbed · active_polar는 있습니다.
  - 데이터셋 18개 씬 중 2개는 팩이 없습니다. `infinigen_office_20260823`은 support 그래프가 없고, `infinigen_office_20260824`는 렌더도 staged 장면도 없습니다.
- **좌표:**
  - 시뮬레이터 상태는 (x, −y, 높이) z-up이고, 헤딩은 π − yaw입니다(MatterSim처럼 양수 = 오른쪽 회전).
  - elevation 0이 데이터셋 카메라 기울기(약 −8.5°)입니다.
  - 데이터셋 `heading_id` h_XXX ↔ heading π − radians(XXX).
- **에피소드:** 0.25 m 격자라 최단 경로도 평균 약 230스텝입니다. 평가 성공 반경 기본 3 m(R2R), OpticalNav 자체 프로토콜은 0.5 m였습니다.
- **freeze 없이 돌릴 때:** 패스 크기 `OPTICALNAV_SIM_SPP_CHUNK`(polar, 기본 64), `OPTICALNAV_SIM_SPP_CHUNK_RGB`(기본 256)가 메모리를 정하고, 패스마다 트레이싱이 반복됩니다. 배치에서는 이 값이 K개 뷰의 합계 예산입니다.
- **서버 운영:**
  - 서버는 시작할 때 팩의 씬 목록을 읽으므로 씬을 추가하면 재시작하세요.
  - 공유 장비에서는 `timeout`으로 띄워 GPU를 붙잡은 채 남지 않게 하세요. Device 2에서 3시간 대여를 7시간 넘긴 적이 있습니다.

## 7. 남은 일

1. Device 1에서 `tools/run_all_checks.sh`를 실행하고, 결과(`bench_*.txt`, `overhead_*.jsonl`, `scaling_*.json`, `samples/`, `*.mp4`)로 보고서 아티팩트에 "Device 1 실측" 행과 영상을 넣습니다.
2. freeze 재생이 포즈를 제대로 따라가는지, active_polar 정합성이 맞는지 확인합니다.
3. 필요하면 R2R 파노라마 에이전트를 위해 한 viewpoint의 12개 헤딩을 한 번에 받는 도우미(`batch` 렌더 활용)를 추가합니다.
4. 디노이저는 16–64 spp에서 다시 평가할 만합니다.
