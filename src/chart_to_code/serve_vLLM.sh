vllm serve /trained_model/snapshots/files \
  --host 0.0.0.0 \
  --port 8501 \
  --dtype float16 \
  --quantization awq_marlin \
  --limit-mm-per-prompt image=3,video=0 \
  --max-model-len 8048 \
  --max-num-seqs 1
