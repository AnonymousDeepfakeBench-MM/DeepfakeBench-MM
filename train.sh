#!/usr/bin/env bash
# Usage:
#   ./train.sh <GPUS_PER_NODE> <train-args...>
#
# Example:
#   # Single-server 1 GPU
#   ./train.sh 1 --config configs/train.yaml
#
#   # Single-server 8 GPUs
#   ./train.sh 8 --config configs/train.yaml

GPUS_PER_NODE=$1
shift

if [ "$GPUS_PER_NODE" -gt 1 ]; then
    torchrun --nproc_per_node=$GPUS_PER_NODE train.py --ddp "$@"
else
    python train.py "$@"
fi
