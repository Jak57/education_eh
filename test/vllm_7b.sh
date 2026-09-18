#!/bin/bash
#SBATCH --job-name=qwen_comprehend
#SBATCH --partition=gpu_windfall
#SBATCH --gres=gpu:1
#SBATCH --account=hawshiuan
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_qwen_comprehend_%j.log
#SBATCH --error=slurm_qwen_comprehend_%j.err

set -euo pipefail

# ============================================================
# Paths
# ============================================================

COMPREHEND_DIR="/groups/hawshiuan/jakir/projects/education_eh"
VLLM_PYTHON="/groups/hawshiuan/jakir/projects/vllm_test/.venv/bin/python"

# Change this if your COMPREHEND environment has a different path.
COMPREHEND_PYTHON="${COMPREHEND_DIR}/venv/bin/python"

PORT=8000

VLLM_PID=""

# ============================================================
# Cleanup
# ============================================================

cleanup() {
    echo
    echo "======================================"
    echo "Cleaning up..."
    echo "======================================"

    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping vLLM (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    echo "Cleanup complete."
}

trap cleanup EXIT

# ============================================================
# STEP 1: Go to COMPREHEND
# ============================================================

cd "${COMPREHEND_DIR}"

echo "======================================"
echo "Qwen + COMPREHEND test"
echo "======================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Directory:  ${COMPREHEND_DIR}"
echo "======================================"

# ============================================================
# STEP 2: Check GPU
# ============================================================

echo
echo "===== GPU ====="
nvidia-smi

# ============================================================
# STEP 3: Check vLLM environment
# ============================================================

echo
echo "===== vLLM environment ====="

if [[ ! -x "${VLLM_PYTHON}" ]]; then
    echo "ERROR: vLLM Python not found:"
    echo "${VLLM_PYTHON}"
    exit 1
fi

"${VLLM_PYTHON}" --version

echo
echo "Checking vLLM installation..."
"${VLLM_PYTHON}" -c \
    "import vllm; print('vLLM version:', vllm.__version__)"

# ============================================================
# STEP 4: Start Qwen 7B
# ============================================================

echo
echo "======================================"
echo "Starting Qwen 7B vLLM server"
echo "======================================"

"${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    > vllm_qwen_7b.log 2>&1 &

VLLM_PID=$!

echo "vLLM PID: ${VLLM_PID}"

# ============================================================
# STEP 5: Wait for vLLM
# ============================================================

echo
echo "======================================"
echo "Waiting for Qwen 7B server..."
echo "======================================"

READY=0

for i in {1..120}; do

    if curl -s "http://127.0.0.1:${PORT}/v1/models" \
        > /tmp/qwen_models.json 2>/dev/null; then

        echo
        echo "Qwen 7B server is READY."
        echo
        cat /tmp/qwen_models.json

        READY=1
        break
    fi

    # Check whether vLLM process has crashed
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo
        echo "ERROR: vLLM process exited unexpectedly."
        echo
        echo "===== vLLM LOG ====="
        cat vllm_qwen_7b.log
        exit 1
    fi

    echo "Waiting... attempt ${i}/120"
    sleep 5
done

if [[ "${READY}" -ne 1 ]]; then
    echo
    echo "ERROR: Qwen 7B server did not become ready."
    echo
    echo "===== vLLM LOG ====="
    cat vllm_qwen_7b.log
    exit 1
fi

# ============================================================
# STEP 6: Test actual generation
# ============================================================

echo
echo "======================================"
echo "Testing Qwen API"
echo "======================================"

"${VLLM_PYTHON}" - <<'PY'
from openai import OpenAI

client = OpenAI(
    api_key="DUMMY",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What is photosynthesis? Answer in one sentence."
        }
    ],
    temperature=0,
    max_tokens=100,
)

print("===== QWEN RESPONSE =====")
print(response.choices[0].message.content)
print("==========================")
PY

echo
echo "Qwen API test successful."

# ============================================================
# STEP 7: Configure Qwen endpoint for COMPREHEND
# ============================================================

export QWEN_API_KEY="DUMMY"
export QWEN_BASE_URL="http://127.0.0.1:8000/v1"
export QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"

echo
echo "======================================"
echo "Qwen environment variables"
echo "======================================"

echo "QWEN_BASE_URL=${QWEN_BASE_URL}"
echo "QWEN_MODEL=${QWEN_MODEL}"

# ============================================================
# STEP 8: Activate COMPREHEND environment
# ============================================================

echo
echo "======================================"
echo "Checking COMPREHEND environment"
echo "======================================"

if [[ ! -x "${COMPREHEND_PYTHON}" ]]; then
    echo "ERROR: COMPREHEND Python not found:"
    echo "${COMPREHEND_PYTHON}"
    echo
    echo "If your environment has a different path,"
    echo "change COMPREHEND_PYTHON at the top of this script."
    exit 1
fi

"${COMPREHEND_PYTHON}" --version

# ============================================================
# STEP 9: Run COMPREHEND
# ============================================================

echo
echo "======================================"
echo "Running COMPREHEND"
echo "======================================"

# "${COMPREHEND_PYTHON}" main.py run --method blossom

# ============================================================
# STEP 10: Finish
# ============================================================

echo
echo "======================================"
echo "COMPREHEND finished successfully."
echo "======================================"

echo
echo "Qwen server log:"
tail -n 20 vllm_qwen_7b.log

echo
echo "Job completed."