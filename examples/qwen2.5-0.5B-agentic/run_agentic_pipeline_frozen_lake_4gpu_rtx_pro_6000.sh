#!/bin/bash
set +x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
source .venv/bin/activate

CONFIG_PATH=$(basename $(dirname $0))
.venv/bin/python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH --config_name agent_val_frozen_lake_4gpu_rtx_pro_6000
