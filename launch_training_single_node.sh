#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
## Usage:
#./launch_training_single_node.sh -p $datatype

### Parse Args. ###
DATATYPE="TF32"  # Default value
while getopts "p:" opt; do
    case "$opt" in
        p) DATATYPE="$OPTARG" ;;
        *) echo "Usage: $0 [-p precision]"; exit 1 ;;
    esac
done

### Configure Workload ###
echo "Setting env. vars."
#source ./utils/set_env_variables.sh
source ./training_config.sh

### Set Paths ###
if [ ! -d "./training_logs" ]; then
  mkdir ./training_logs
fi
TIME=$(date +"%Y:%m:%d_%H:%M:%S")
LOG_FILE="./training_logs/log_${TIME}.txt"
WORKDIR=./torchrec_dlrm
export PYTHONPATH=$PYTHONPATH:$WORKDIR

### Launch Training ###
echo "Launching DLRM training"

if [[ "$DATATYPE" == "TF32" ]]; then
  
  echo "TF32 Training Mode"
  HIPBLASLT_ALLOW_TF32=1 TORCH_NCCL_HIGH_PRIORITY=1 GPU_MAX_HW_QUEUES=4 torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 --role trainer ${WORKDIR}/dlrm_main.py -- \
  --embedding_dim 128 \
  --dense_arch_layer_sizes 512,256,128 \
  --over_arch_layer_sizes 1024,1024,512,256,1 \
  --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
  --allow_tf32 \
  --epochs 1 \
  --pin_memory \
  --mmap_mode \
  --shuffle_batches \
  --batch_size $BATCH_SIZE \
  --limit_train_batches $NUM_TRAIN_BATCHES \
  --validation_freq_within_epoch $VALIDATION_FREQ \
  --limit_val_batches $NUM_VAL_BATCHES \
  --validation_auroc 1.0 \
  --interaction_type=dcn \
  --dcn_num_layers=3 \
  --dcn_low_rank_dim=512 \
  --adagrad \
  --learning_rate $LEARNING_RATE \
  --lr_warmup_steps $WARMUP_STEPS \
  --lr_decay_start $DECAY_START_STEP \
  --lr_decay_steps $DECAY_STEPS \
  --print_progress \
  --multi_hot_distribution_type uniform \
  --multi_hot_sizes=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
  --seed $RANDOM_SEED > "$LOG_FILE" 2>&1
  
else

  echo "FP32 Training Mode"
  TORCH_NCCL_HIGH_PRIORITY=1 GPU_MAX_HW_QUEUES=4 torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 --role trainer ${WORKDIR}/dlrm_main.py -- \
  --embedding_dim 128 \
  --dense_arch_layer_sizes 512,256,128 \
  --over_arch_layer_sizes 1024,1024,512,256,1 \
  --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
  --epochs 1 \
  --pin_memory \
  --mmap_mode \
  --shuffle_batches \
  --batch_size $BATCH_SIZE \
  --limit_train_batches $NUM_TRAIN_BATCHES \
  --validation_freq_within_epoch $VALIDATION_FREQ \
  --limit_val_batches $NUM_VAL_BATCHES \
  --validation_auroc 1.0 \
  --interaction_type=dcn \
  --dcn_num_layers=3 \
  --dcn_low_rank_dim=512 \
  --adagrad \
  --learning_rate $LEARNING_RATE \
  --lr_warmup_steps $WARMUP_STEPS \
  --lr_decay_start $DECAY_START_STEP \
  --lr_decay_steps $DECAY_STEPS \
  --print_progress \
  --multi_hot_distribution_type uniform \
  --multi_hot_sizes=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
  --seed $RANDOM_SEED > "$LOG_FILE" 2>&1

fi

python ./utils/process_training_log.py --log_file $LOG_FILE
