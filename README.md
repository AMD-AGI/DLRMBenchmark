# DLRMBenchmark
Repository for Showcasing DLRM v2 functionality on a single AMD single-node. This codebase is not meant to showcase the peak achievable performance. The model has been derived from the [ML Perf DLRM v2](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm) repo.

# Installation

## Requirements
- Single-node w/ 8x MI300X/325X/350X
- Docker / Torch wheels for ROCm

## Setup
1. Clone repository:
   ```
   git clone https://github.com/AMD-AGI/DLRMBenchmark.git
   ``` 
2. Update permissions to 777 for all shell scripts
3. Pull ROCm + torch image w/ FBGEMM and torchrec installed: egs ```docker pull rocm/pytorch-training:latest```
4. Launch container. Ensure all required paths including codebase are mounted (similar to /home_dir/ below).
    ```
    docker run -d \
    --ipc=host \
    -v /dev/shm:/dev/shm \
    -v /home_dir/:/home_dir/ \
    -e USER=$user -e UID=$uid -e GID=$gid \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --ulimit memlock=-1:-1 \
    --shm-size 32G \
    --privileged \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --group-add video \
    --network=host \
    --name dlrm_demo \
    -it rocm/pytorch-training:latest \
    tail -f /dev/null
   ```

# Model Training
1. Start interactive shell session within container: ```docker exec -it dlrm_demo bash```
2. Modify training configuration (if required) in ```training_config.sh```
3. Launch training using ```./launch_training_single_node.sh```. Check for training progress in the ./training_logs folder. Upon completion, the final row in the ```results.csv``` file shows the mean rec/s. 
