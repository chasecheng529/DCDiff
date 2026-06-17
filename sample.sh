#!/bin/bash
# 使用 set -e 让脚本在任何命令失败时立即退出
set -eo pipefail

# --- Can be edited ---
CHECKPOINT_PATH="/home/zhangxiaohong/chengcheng/ADOptDiff/ADOptDiff/models/Case_study/DCDiff_NewTrain_E73.ckpt"
GPU_ID="2" 
N_SAMPLES=100
AFFINITY=1.0
GUIDENCE=20.0

# --- Do not modify ---
DEVICE="cuda:0"
DATA_PATH="data"
PREFIX="test.full"



CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH" .ckpt)
SAMPLES_DIR="sample_results"
mkdir -p "$SAMPLES_DIR"
echo "Make sure directories exist: $SAMPLES_DIR"


echo "Sample begin at $(date) (Checkpoint: ${CHECKPOINT_NAME}))"
echo "----------------------------------------"


echo "Runing 'sample.py' to sample molecules "
CUDA_VISIBLE_DEVICES=$GPU_ID python -W ignore sample.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --samples "$SAMPLES_DIR" \
    --data "$DATA_PATH" \
    --prefix "$PREFIX" \
    --n_samples $N_SAMPLES \
    --device "$DEVICE" \
    --affinity $AFFINITY \
    --guidance_scale $GUIDENCE \
    --batch_size 16
echo "sample finished"
echo "----------------------------------------"