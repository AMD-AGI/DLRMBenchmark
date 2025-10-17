# Training Config,
export TF32_MODE=1 # 1: TF32, 0: FP32
export BATCH_SIZE=32768
export NUM_TRAIN_BATCHES=1000
export VALIDATION_FREQ=1500
export NUM_VAL_BATCHES=100
export RANDOM_SEED=1024
#export DATASET_PATH= # For using Criteo-1TB dataset, uncomment and set path + Add --in_memory_binary_criteo_path $DATASET_PATH to launch script
export LEARNING_RATE=0.001
export WARMUP_STEPS=8000
export DECAY_START_STEP=8000
export DECAY_STEPS=24000