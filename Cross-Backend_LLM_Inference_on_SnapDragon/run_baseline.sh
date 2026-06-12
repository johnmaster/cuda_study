#!/bin/bash

MODEL="/sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf"
BENCH="./llama-bench"
LOG="$HOME/tripath/results/baseline_threads.log"

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

log() {
  echo "$@" | tee -a "$LOG"
}

if [ ! -f "$MODEL" ]; then
  log "ERROR: model not found: $MODEL"
  exit 1
fi

if [ ! -x "$BENCH" ]; then
  log "ERROR: llama-bench not found or not executable: $BENCH"
  log "Please run this script in the directory containing llama-bench."
  exit 1
fi

THREADS=(1 2 4 6 8)
TOTAL=${#THREADS[@]}
COUNT=0

log "=== baseline thread sweep started ==="
log "time: $(date)"
log "model: $MODEL"
log "bench: $BENCH"
log "log: $LOG"
log ""

for t in "${THREADS[@]}"; do
  COUNT=$((COUNT + 1))
  CMD="$BENCH -m $MODEL -p 512 -n 128 -t $t"

  log "=== [$COUNT/$TOTAL] start: threads=$t ==="
  log "CMD: $CMD"
  log "start time: $(date)"

  $CMD 2>&1 | tee -a "$LOG"

  log "end time: $(date)"
  log "=== [$COUNT/$TOTAL] done: threads=$t ==="
  log ""
done

log "=== all completed ==="
log "time: $(date)"
log "log saved to: $LOG"
