<!-- Badges -->
[![License](https://img.shields.io/github/license/AMD-AGI/DLRMBenchmark.svg?style=flat)](LICENSE)
[![Contributors](https://img.shields.io/github/contributors/AMD-AGI/DLRMBenchmark.svg?style=flat)](https://github.com/AMD-AGI/DLRMBenchmark/graphs/contributors)

# DLRMBenchmark

> DLRM v2 benchmark for single-node AMD MI3xx GPUs.

This repository showcases DLRM v2 training on a single AMD node (8x MI3xx). The model is derived from the [MLPerf DLRM v2](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm) repo and targets MI300X, MI325X, and MI350X accelerators. We use the [ROCm PyTorch Training Docker](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html?model=pyt_train_dlrm), which comes pre-installed with FBGEMM and TorchRec. This codebase is not meant to highlight peak achievable performance.

## Installation

### Requirements

- Single-node w/ 8x MI300X/325X/350X
- Docker / Torch wheels for ROCm

### Setup

1. Clone repository:
   ```bash
   git clone https://github.com/AMD-AGI/DLRMBenchmark.git
   ```
2. Update permissions to 777 for all shell scripts
3. Container with ROCm, PyTorch, FBGEMM and torchrec installed are available at [https://hub.docker.com/r/rocm/primus/](https://hub.docker.com/r/rocm/primus/tags). Pull the container:
   ```bash
   docker pull rocm/primus:v26.1
   ```
4. Launch container. Ensure all required paths including codebase are mounted (similar to /home_dir/).
   ```bash
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
     --cap-add=SYS_PTRACE \
     --security-opt seccomp=unconfined \
     --group-add video \
     --network=host \
     --name dlrm_demo \
     -it rocm/primus:v26.1 \
     tail -f /dev/null
   ```

## Model Training

1. Start interactive shell session within container:
   ```bash
   docker exec -it dlrm_demo bash
   ```
2. Modify training configuration (if required) in `training_config.sh`
3. Launch training using `./launch_training_single_node.sh`. Check for training progress in the ./training_logs folder. Upon completion, the final row in the `results.csv` file shows the mean rec/s.

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, branch strategy, coding standards, and the pull request process.

For bugs and feature requests, open a [GitHub Issue](../../issues).

---

## Security

To report a security vulnerability, **do not open a public GitHub issue**.
See [SECURITY.md](SECURITY.md) for our responsible disclosure policy.

---

## Contact

For questions, issues, or contributions, please reach out to the maintainers:

- Tharun Adithya Srikrishnan — [@tsrikris](https://github.com/tsrikris)

See [CODEOWNERS](.github/CODEOWNERS) for the full ownership list.

---

## License

This project is licensed under the [MIT License](LICENSE).
