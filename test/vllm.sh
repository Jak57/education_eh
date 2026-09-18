#!/bin/bash
#SBATCH --job-name=qwen_tree
#SBATCH --partition=gpu_standard
#SBATCH --account=hawshiuan
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=slurm_qwen_%j.log
#SBATCH --error=slurm_qwen_%j.err

set -e
cd /groups/hawshiuan/waqar/COMPREHEND_public
export HF_HOME=/groups/hawshiuan/waqar/COMPREHEND_public/.cache_hf
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p $HF_HOME

echo "=== STEP 1: Starting vLLM 7B on port 8000 ==="
vllm_env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.55 \
    --dtype bfloat16 > vllm_7b.log 2>&1 &
VLLM_7B_PID=$!
echo "vLLM 7B PID: $VLLM_7B_PID"

echo "=== STEP 2: Waiting for 7B server ==="
for i in {1..90}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "7B READY after $i attempts"
        break
    fi
    sleep 5
done

curl http://localhost:8000/v1/models || (echo "7B FAILED" && cat vllm_7b.log && exit 1)

echo "=== STEP 3: Starting vLLM 3B on port 8001 ==="
vllm_env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8001 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --dtype bfloat16 > vllm_3b.log 2>&1 &
VLLM_3B_PID=$!
echo "vLLM 3B PID: $VLLM_3B_PID"

echo "=== STEP 4: Waiting for 3B server ==="
for i in {1..60}; do
    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "3B READY after $i attempts"
        break
    fi
    sleep 5
done

curl http://localhost:8001/v1/models || (echo "3B FAILED" && cat vllm_3b.log && exit 1)

echo "=== STEP 5: Running repo pipeline ==="
source comprehen_env/bin/activate
python3 main.py run --method blossom

echo "=== STEP 6: Running evaluation ==="
python3 evaluation/evaluate.py Results/UF_random_prompts_64_base_blossom.json

echo "=== STEP 7: Cleanup ==="
kill $VLLM_7B_PID $VLLM_3B_PID || true


echo "=== DONE ==="
ls -la Results/UF_random_prompts_64_base_blossom.json
ls -la Results/eval_outputs/