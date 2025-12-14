#!/bin/bash
#SBATCH --job-name=abl
#SBATCH --array=1-60                 
#SBATCH --nodes=1                    # Each job on a single node
#SBATCH --ntasks=1                   # Each job is a single task
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --output=sbatch-logs/trial_%j_%a.log
#SBATCH --error=sbatch-logs/trial_%j_%a.err

manifest="${1:?ERROR -- must pass a manifest file}"
taskid=$SLURM_ARRAY_TASK_ID
CONFIG_FILE=$(getline $taskid $manifest | cut -f 1 -d ' ')
BASE_LOG_DIR=$(getline $taskid $manifest | cut -f 2 -d ' ')

# --- Job Configuration ---
# CONFIG_FILE="configs/mountaincar/mountaincar_propsp.yaml"
# BASE_LOG_DIR="mcd-logs/run2/MCD_traj_20_summary"

# --- Create a unique directory for this specific trial's output ---
TRIAL_LOG_DIR="${BASE_LOG_DIR}/trial_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TRIAL_LOG_DIR"

# --- Load Modules ---
module load mamba/latest
module load ollama/0.12.3

# --- START THE OLLAMA SERVER IN THE BACKGROUND ---
echo "--- Starting Ollama server on $(hostname) ---"
ollama-start

sleep 15
echo "--- Checking Ollama status ---"
ollama list

# --- Activate Mamba/Conda Environment ---
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