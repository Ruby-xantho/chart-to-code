export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL


vllm serve /workspace/PDF-AI/hf/hub/models--Qwen--Qwen2.5-VL-72B-Instruct-AWQ/snapshots/c8b87d4b81f34b6a147577a310d7e75f0698f6c2 \
  --host 0.0.0.0 \
  --port 8501 \
  --dtype float16 \
  --quantization awq_marlin \
  --tensor-parallel-size 2 \
  --generation-config /workspace/PDF-AI/production \
  --limit-mm-per-prompt image=1,video=0 \
  --max-model-len 2048 \
  --max-num-seqs 1


--host 0.0.0.0 --port 8000
--model/workspace/ctc/hf/hub/models--Qwen--Qwen2.5-VL-72B-Instruct-AWQ/snapshots/
--quantization awq_marlin --dtype float16 --gpu-memory-utilization 0.95 --max-model-len 8192 --tensor-parallel-size 2
--generation-config /workspace/ctc/production --limit-mm-per-prompt image=3,video=0
