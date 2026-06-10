#!/bin/bash
MODEL=/sdcard/Download/models/qwen2.5-3b-instruct-q4_k_m.gguf
BENCH=./llama-bench
LOG=~/tripath/results/baseline_threads.log
mkdir -p ~/tripath/results
for t in 1 2 4 6 8; do
 CMD="$BENCH -m $MODEL -p 512 -n 128 -t $t"
 echo "=== CMD: $CMD ===" >> $LOG
 $CMD >> $LOG
 echo "" >> $LOG
done

echo "=== Done ===" >> $LOG
cat $LOG
