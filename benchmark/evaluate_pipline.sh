#!/bin/bash
set -eo pipefail

# --- Config ---
# --- Path to saved samples, formatted as 0/ 1/ 2/ ... ---
# --- For each sample folder, there are {index}_.sdf, {index}_.xyz files for generated molecules ---
# --- For each sample folder, there are core_.sdf, core_.xyz, gt_{index}.sdf, gt_{index}.xyz, pocket_.xyz and pocket_.pdb files as ground truth---
SAMPLES_PATH="../sample_results"
EXPERIMENT_NAME="TEST"


PROCESSED_DATA_DIR="evaluate"
FORMATTED_RESULT_DIR="${PROCESSED_DATA_DIR}/${EXPERIMENT_NAME}/formatted"
PRED_GT_CSV_PATH="${FORMATTED_RESULT_DIR}/pred_gt_map.csv"

DOCKING_RESULT_DIR="${PROCESSED_DATA_DIR}/${EXPERIMENT_NAME}/docking"
RESULTS_PRED_PATH="${DOCKING_RESULT_DIR}/docking_result_pred.pt"
RESULTS_BASE_PATH="utils/docking_result_ref.pt" # We have provided the reference docking results

LOG_FILE="${PROCESSED_DATA_DIR}/${EXPERIMENT_NAME}/evaluate_result.log"


if [ ! -d "$SAMPLES_PATH" ]; then
    echo "ERROR - No Samples directory: $SAMPLES_PATH"
    exit 1 
fi

mkdir -p "$FORMATTED_RESULT_DIR"
mkdir -p "$DOCKING_RESULT_DIR"
echo "INFO - Create directories: $FORMATTED_RESULT_DIR, $DOCKING_RESULT_DIR"


> "$LOG_FILE"

# --- Start evaluation ---
echo "Pipeline started at $(date) (Experiment Name: ${EXPERIMENT_NAME})" | tee -a "$LOG_FILE"
echo "All output will be recorded in $LOG_FILE and shown in the console." | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# === Step 2/4: run reformat.py ===
echo "[Step 2/4] Running 'reformat.py' to convert xyz files to SMILES for docking and evaluation..." | tee -a "$LOG_FILE"
python reformat.py --samples "$SAMPLES_PATH" \
                   --formatted "$PRED_GT_CSV_PATH"
echo "[Step 2/4] 'reformat.py' finished" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# === Step 3/4: run vina_docking.py ===
echo "[Step 3/4] Running 'vina_docking.py' for molecular docking" | tee -a "$LOG_FILE"
python vina_docking.py --formatted "$PRED_GT_CSV_PATH" \
    --results_pred_path "$RESULTS_PRED_PATH" \
    --results_ref_path "$RESULTS_BASE_PATH" \
    > >(tee -a "$LOG_FILE")  # Log stdout.
echo "[Step 3/4] 'vina_docking.py' finished" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# === Step 4/4: run computer_metrics.py ===
echo "[Step 4/4] Running 'computer_metrics.py' to compute metrics" | tee -a "$LOG_FILE"
python -W ignore computer_metrics.py "$PRED_GT_CSV_PATH" > >(tee -a "$LOG_FILE")  # Log stdout.
echo "[Step 4/4] 'computer_metrics.py' finished" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

echo "Pipeline finished successfully at $(date)." | tee -a "$LOG_FILE"
