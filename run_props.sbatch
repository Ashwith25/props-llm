#!/bin/bash
#SBATCH --job-name=propsp_loop
#SBATCH --array=1-10                 # Run 10 parallel jobs
#SBATCH --nodes=1                    # Each job on a single node
#SBATCH --ntasks=1                   # Each job is a single task
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_output_IDP_b10r10/trial_%j_%a.log
#SBATCH --error=logs/slurm_output_IDP_b10r10/trial_%j_%a.err

# --- Job Configuration ---
CONFIG_FILE="configs/inverted_double_pendulum/inverteddoublependulum_propsp.yaml"
BASE_LOG_DIR="logs/IDP_summarize_traj_20_b10R10"

# --- Create a unique directory for this specific trial's output ---
TRIAL_LOG_DIR="${BASE_LOG_DIR}/trial_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TRIAL_LOG_DIR"

# --- Load Modules ---
module load mamba/latest
module load ollama/0.12.3

# --- START THE OLLAMA SERVER IN THE BACKGROUND ---
echo "--- Starting Ollama server on $(hostname) ---"
export OLLAMA_HOST=0.0.0.0
ollama serve &
OLLAMA_PID=$!
sleep 15
echo "--- Checking Ollama status ---"
ollama list

# --- Activate Mamba/Conda Environment ---
# We keep this line, as it's good practice and may set paths
source activate thesis
# --- Run the Python Script ---
echo "--- Starting SLURM Trial ${SLURM_ARRAY_TASK_ID} ---"
echo "--- Config: ${CONFIG_FILE} ---"
echo "--- Logging to: ${TRIAL_LOG_DIR} ---"

# --- THIS IS THE MODIFIED EXECUTION LINE ---
# Use the absolute path to the environment's python
/home/apoojar4/.conda/envs/thesis/bin/python3 main.py --config "$CONFIG_FILE" --logdir "$TRIAL_LOG_DIR"

# Check exit status
EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "--- ERROR: Trial ${SLURM_ARRAY_TASK_ID} failed with exit status ${EXIT_STATUS} ---"
else
    echo "--- Success: Trial ${SLURM_ARRAY_TASK_ID} completed ---"
fi

# --- CLEAN UP: Stop the Ollama server ---
echo "--- Shutting down Ollama server (PID: $OLLAMA_PID) ---"
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null
exit $EXIT_STATUS