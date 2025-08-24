export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

vllm serve /trained_model/snapshots/files \
  --host 0.0.0.0 \
  --port 8501 \
  --dtype float16 \
  --quantization awq_marlin \
  --limit-mm-per-prompt image=3,video=0 \
  --max-model-len 8048 \
  --max-num-seqs 1
