#!/bin/bash
set -eo pipefail

# --- Config ---
# --- Path to saved samples, formatted as 0/ 1/ 2/ ... ---
# --- For each sample folder, there are {index}_.sdf, {index}_.xyz files for generated molecules ---
# --- For each sample folder, there are core_.sdf, core_.xyz, gt_{index}.sdf, gt_{index}.xyz, pocket_.xyz and pocket_.pdb files as ground truth---
SAMPLES_PATH="/home/zhangxiaohong/chengcheng/Code/DCDiff/sample_results"
EXPERIMENT_NAME="TEST"


# === 【核心修改】新的目录结构逻辑 ===


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

# --- 开始执行 ---
# 【保持不变】从这里开始的所有代码，都和您之前的版本完全一样
echo "流水线开始运行于 $(date) (Experiment Name: ${EXPERIMENT_NAME})" | tee -a "$LOG_FILE"
echo "将把所有输出【同时记录到 $LOG_FILE 并显示在控制台】" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# === 步骤 2/4：运行 reformat.py ===
echo "[步骤 2/4] Runing 'reformat.py' to covert xyz file to SMILES for docking and evaluating..." | tee -a "$LOG_FILE"
python reformat.py --samples "$SAMPLES_PATH" \
                   --formatted "$PRED_GT_CSV_PATH"
echo "[步骤 2/4] 'reformat.py' finished" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# === 步骤 3/4：运行 vina_docking.py ===
echo "[步骤 3/4] Runing 'vina_docking.py' for docking mols" | tee -a "$LOG_FILE"
python vina_docking.py --formatted "$PRED_GT_CSV_PATH" \
    --results_pred_path "$RESULTS_PRED_PATH" \
    --results_ref_path "$RESULTS_BASE_PATH" \
    > >(tee -a "$LOG_FILE")  # stdout 进日志
echo "[步骤 3/4] 'vina_docking.py' finished" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# === 步骤 3/5：运行 computer_metrics.py ===
echo "[步骤 4/4] Runing 'computer_metrics.py' to compute metrics" | tee -a "$LOG_FILE"
python -W ignore computer_metrics.py "$PRED_GT_CSV_PATH" > >(tee -a "$LOG_FILE")  # stdout 进日志
echo "[步骤 4/4] 'computer_metrics.py' finished" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

echo "流水线成功结束于 $(date)." | tee -a "$LOG_FILE"