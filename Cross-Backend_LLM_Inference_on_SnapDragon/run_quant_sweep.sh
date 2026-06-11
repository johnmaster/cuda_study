#!/bin/bash

BENCH="$HOME/llama.cpp/build/bin/llama-bench"
DIR="/sdcard/Download/models"
LOG="$HOME/tripath/results/quant_sweep.log"

MODELS=(
  "qwen2.5-3b-instruct-q4_0.gguf"
  "qwen2.5-3b-instruct-q6_k.gguf"
  "qwen2.5-3b-instruct-q4_k_m.gguf"
  "qwen2.5-3b-instruct-q8_0.gguf"
  "qwen2.5-3b-instruct-q5_k_m.gguf"
)

THREADS=(2 4 6 8)

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

if [ ! -x "$BENCH" ]; then
  echo "llama-bench not found or not executable: $BENCH" | tee -a "$LOG"
  exit 1
fi

TOTAL=$(( ${#MODELS[@]} * ${#THREADS[@]} ))
COUNT=0

for model in "${MODELS[@]}"; do
  FULL_PATH="$DIR/$model"

  if [ ! -f "$FULL_PATH" ]; then
    echo "skip $model, file not found: $FULL_PATH" | tee -a "$LOG"
    continue
  fi

  for t in "${THREADS[@]}"; do
    COUNT=$((COUNT + 1))

    CMD="$BENCH -m $FULL_PATH -p 512 -n 128 -t $t"

    echo ""
    echo "[$COUNT/$TOTAL] running $model with -t $t..."
    echo "" >> "$LOG"
    echo "=== [$COUNT/$TOTAL] MODEL: $model THREADS: $t ===" >> "$LOG"
    echo "=== CMD: $CMD ===" >> "$LOG"

    $CMD 2>&1 | tee -a "$LOG"

    echo "" >> "$LOG"
    echo "[$COUNT/$TOTAL] completed $model with -t $t"
  done
done

echo ""
echo "all completed! log: $LOG"
