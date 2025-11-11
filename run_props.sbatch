#!/bin/bash
#SBATCH --job-name=mcd_summ
#SBATCH --array=1-5                 # Run 5 parallel jobs
#SBATCH --nodes=1                    # Each job on a single node
#SBATCH --ntasks=1                   # Each job is a single task
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --output=mcd-logs/slurm/slurm_output_MCD_traj_20_summary/trial_%j_%a.log
#SBATCH --error=mcd-logs/slurm/slurm_output_MCD_traj_20_summary/trial_%j_%a.err

# --- Job Configuration ---
CONFIG_FILE="configs/mountaincar/mountaincar_propsp.yaml"
BASE_LOG_DIR="mcd-logs/MCD_traj_20_summary"

# --- Create a unique directory for this specific trial's output ---
TRIAL_LOG_DIR="${BASE_LOG_DIR}/trial_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TRIAL_LOG_DIR"

# --- Load Modules ---
module load mamba/latest
module load ollama/0.12.3

# --- START THE OLLAMA SERVER IN THE BACKGROUND ---
echo "--- Starting Ollama server on $(hostname) ---"
# export OLLAMA_HOST=0.0.0.0
# ollama serve &
# OLLAMA_PID=$!


# --- OLLAMA shared daemon (per-node, refcounted) ----------------------------
# Optional: pin a common port for all jobs (recommended)
# export OLLAMA_HOST="$(hostname):11437"   # or 0.0.0.0:11437 if you need remote access

STATE_DIR="/tmp/ollama-$(hostname)"
LOCK="${STATE_DIR}/lock"
STATE="${STATE_DIR}/state"
mkdir -p "$STATE_DIR"; chmod 700 "$STATE_DIR"
exec 9>"$LOCK"  # lock fd

resolve_hostport() {
  local hp="${OLLAMA_HOST:-}"
  if [[ -z "$hp" ]]; then
    # Discover an existing server (first ollama listener)
    if command -v ss >/dev/null 2>&1; then
      hp=$(ss -plnt 2>/dev/null | awk '/ollama/ {gsub(/\[|\]/,"",$4); print $4; exit}')
    elif command -v netstat >/dev/null 2>&1; then
      hp=$(netstat -plnt 2>/dev/null | awk '/ollama/ {gsub(/\[|\]/,"",$4); print $4; exit}')
    fi
    [[ -n "$hp" ]] || hp="$(hostname):11434"
  fi
  hp="${hp#*://}"; [[ "$hp" == *:* ]] || hp="${hp}:11434"
  echo "$hp"
}

curl_hp() {
  local hp="$1"; local h="${hp%:*}" p="${hp##*:}"
  case "$h" in 0.0.0.0|127.0.0.1|localhost|::) h="127.0.0.1";; esac
  echo "${h}:${p}"
}

is_up() {
  local chp; chp="$(curl_hp "$1")"
  export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,${chp%:*}"
  curl -fsS --max-time 2 "http://${chp}/api/version" >/dev/null \
  || curl -fsS --max-time 2 "http://${chp}/api/tags" >/dev/null
}

read_state() { [[ -f "$STATE" ]] || return 1; . "$STATE"; }   # loads HOSTPORT PID COUNT
write_state() {
  umask 0177
  printf 'HOSTPORT=%q\nPID=%q\nCOUNT=%q\n' "$HOSTPORT" "${PID:-}" "${COUNT:-0}" > "$STATE"
}

HOSTPORT="$(resolve_hostport)"

# Acquire node-scoped lock
flock 9

if is_up "$HOSTPORT"; then
  # Reuse existing server; increment refcount
  if ! read_state || [[ "$HOSTPORT" != "${HOSTPORT:-}" ]]; then
    # populate PID for this port if missing or mismatched
    PID=$(ss -plnt 2>/dev/null | awk -v p=":${HOSTPORT##*:}$" '$4 ~ p && /ollama/ {match($0,/pid=([0-9]+)/,m); if(m[1]) print m[1]; exit}')
    COUNT=0
  fi
  COUNT=$(( ${COUNT:-0} + 1 ))
  write_state
  echo "--- Using Ollama at http://$(curl_hp "$HOSTPORT") (users=$COUNT) ---"
else
  # Start new server
  export OLLAMA_HOST="$HOSTPORT"
  echo "--- Starting Ollama on $OLLAMA_HOST ---"
  nohup ollama serve >"${TRIAL_LOG_DIR:-/tmp}/ollama_${SLURM_JOB_ID:-$$}.log" 2>&1 &
  PID=$!
  # Wait up to 60s
  for i in {1..60}; do is_up "$HOSTPORT" && break; sleep 1; done
  is_up "$HOSTPORT" || { echo "!!! Ollama failed to start"; flock -u 9; exit 1; }
  COUNT=1; write_state
  echo "--- Ollama ready at http://$(curl_hp "$HOSTPORT") (PID=$PID) ---"
fi

flock -u 9
export OLLAMA_HOST="$HOSTPORT"

cleanup_ollama() {
  # Decrement refcount; only kill when last user leaves
  flock 9
  if read_state; then
    COUNT=$(( ${COUNT:-1} - 1 ))
    if (( COUNT <= 0 )); then
      echo "--- Last user exiting; stopping Ollama PID ${PID:-?} ---"
      [[ -n "${PID:-}" ]] && kill "$PID" 2>/dev/null || true
      rm -f "$STATE"
    else
      write_state
      echo "--- Leaving Ollama running (remaining users=$COUNT) ---"
    fi
  fi
  flock -u 9
}
trap cleanup_ollama EXIT
# ---------------------------------------------------------------------------


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