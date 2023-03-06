#!/bin/bash
#SBATCH -n 40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=48:00:00
#SBATCH -J lidat_sup

set -x

# file definition
CONFIG_FILE=$1
CONFIG_PY="${CONFIG_FILE##*/}"
CONFIG="${CONFIG_PY%.*}"
WORK_DIR="./work_dirs/${CONFIG}"
CHECKPOINT=$2

# train config
GPUS=4
PORT=${PORT:-29511}
RANDOM_SEED=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

if [ ! -d "${WORK_DIR}" ]; then
  mkdir -p "${WORK_DIR}"
  cp "${CONFIG_FILE}" $0 "${WORK_DIR}"
fi

echo -e "\n config file: ${CONFIG}\n"

# training
python train_net_nuscenes.py  --config-file ${CONFIG_FILE} --num-gpus 4  --eval-only  MODEL.WEIGHTS ${CHECKPOINT}

