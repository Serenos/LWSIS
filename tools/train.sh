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
folder_name=$2
exp_id=$3
# train config
GPUS=4
PORT=${PORT:-29510}
RANDOM_SEED=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

if [ ! -d "${WORK_DIR}" ]; then
  mkdir -p "${WORK_DIR}"
  cp "${CONFIG_FILE}" $0 "${WORK_DIR}"
fi

echo -e "\n config file: ${CONFIG}\n"

# training
python tools/train_net_nuscenes.py --config-file ${CONFIG_FILE} --num-gpus 4 --folder-name ${folder_name} --exp-id ${exp_id} --dist-url "tcp://127.0.0.1:29511"
