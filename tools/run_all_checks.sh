#!/usr/bin/env bash
# Every check on one GPU host, in order: parity (all variants), frame-time split, mode benchmark,
# batch scaling and demo videos, for polar then rgb. Each server runs under a hard time limit.
#
#   CARD=0 PY=/usr/bin/python3 tools/run_all_checks.sh [scene] [out_dir]
#
# PY must import mitsuba (set PYTHONPATH / source setpath.sh first) and have numpy + pillow.
set -u
cd "$(dirname "$0")/.."
SCENE=${1:-infinigen_apartment_natural_v1_20268504}
OUT=${2:-runs/$(date -u +%Y%m%dT%H%M%SZ)}
CARD=${CARD:-0}; PY=${PY:-python3}; PACK=${PACK:-packs/opticalnav-v0.2}; LIMIT=${LIMIT:-3h}
mkdir -p "$OUT"

start() {  # mode port
  CUDA_VISIBLE_DEVICES=$CARD timeout "$LIMIT" "$PY" -u -m opticalnav_sim.server --pack "$PACK" --port "$2" \
    --modes "$1" --max-resident 2 --preload "$SCENE:base:$1" > "$OUT/server_$1.log" 2>&1 &
  SPID=$!
  until grep -qE "serve\]|Error|Traceback|Segmentation" "$OUT/server_$1.log" || ! kill -0 $SPID 2>/dev/null; do sleep 3; done
  grep -E "preload|serve" "$OUT/server_$1.log"
}
stop() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; }

URL=http://127.0.0.1:18770
start polar 18770
"$PY" tools/check_parity.py --pack "$PACK" --scene "$SCENE" --spp 256 --server $URL | tee "$OUT/parity_polar.jsonl"
"$PY" tools/frame_overhead.py --pack "$PACK" --scene "$SCENE" --server $URL --mode polar | tee "$OUT/overhead_polar.jsonl"
"$PY" tools/benchmark_modes.py --pack "$PACK" --scene "$SCENE" --server $URL --modes polar \
  --spp 128,256,512,1024,2048 --denoise off,on --frames 3 --out "$OUT/bench_polar.json" | tee "$OUT/bench_polar.txt"
"$PY" tools/batch_scaling.py --pack "$PACK" --scene "$SCENE" --server $URL --mode polar --spp 64 \
  --k 1,2,4,8,12 --out "$OUT/scaling_polar.json"
"$PY" tools/sample_frames.py --pack "$PACK" --scene "$SCENE" --server $URL --mode polar --spp 128,1024 \
  --denoise off,on --out "$OUT/samples" --tag polar
"$PY" tools/record_episode.py --pack "$PACK" --scene "$SCENE" --server $URL --mode polar --spp 128 \
  --agents 2 --max-steps 60 --fps 6 --out "$OUT/sim_polar_2agents.mp4"
stop

URL=http://127.0.0.1:18771
start rgb 18771
"$PY" tools/frame_overhead.py --pack "$PACK" --scene "$SCENE" --server $URL --mode rgb | tee "$OUT/overhead_rgb.jsonl"
"$PY" tools/benchmark_modes.py --pack "$PACK" --scene "$SCENE" --server $URL --modes rgb \
  --spp 128,256,512,1024,2048 --denoise off,on --frames 3 --out "$OUT/bench_rgb.json" | tee "$OUT/bench_rgb.txt"
"$PY" tools/batch_scaling.py --pack "$PACK" --scene "$SCENE" --server $URL --mode rgb --spp 64 \
  --k 1,2,4,8,12 --out "$OUT/scaling_rgb.json"
"$PY" tools/sample_frames.py --pack "$PACK" --scene "$SCENE" --server $URL --mode rgb --spp 128 \
  --denoise off,on --out "$OUT/samples" --tag rgb
"$PY" tools/record_episode.py --pack "$PACK" --scene "$SCENE" --server $URL --mode rgb --spp 128 \
  --agents 4 --max-steps 80 --fps 8 --out "$OUT/sim_rgb_4agents.mp4"
stop
echo "done -> $OUT"
