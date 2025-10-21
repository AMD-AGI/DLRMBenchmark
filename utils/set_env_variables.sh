# Set Global Env Vars for DLRM
GPU_ID=$(rocminfo | grep -o 'gfx[0-9]*' | sed 's/gfx//' | head -n 1)
if [ "$GPU_ID" -eq 942 ]; then
    echo "Setting environment variables for MI300X"
    export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7
    export NCCL_IB_GID_INDEX=3 
    export NCCL_NCHANNELS_PER_NET_PEER=8
    export RCCL_MSCCL_ENABLE=0
    export RCCL_MSCCLPP_ENABLE=0
    export HSA_ENABLE_IPC_MODE_LEGACY=1
elif [ "$GPU_ID" -eq 944 ]; then
    echo "Setting environment variables for MI325X"
    export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7
    export NCCL_IB_GID_INDEX=3 
    export NCCL_NCHANNELS_PER_NET_PEER=8
    export RCCL_MSCCL_ENABLE=0
    export RCCL_MSCCLPP_ENABLE=0
    export HSA_ENABLE_IPC_MODE_LEGACY=1
elif [ "$GPU_ID" -eq 950 ]; then
    echo "Setting environment variables for MI350X"
    export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7
    export NCCL_IB_GID_INDEX=3 
    export NCCL_NCHANNELS_PER_NET_PEER=8
    export RCCL_MSCCL_ENABLE=0
    export RCCL_MSCCLPP_ENABLE=0
    export HSA_ENABLE_IPC_MODE_LEGACY=1
else
    echo "Unknown GPU ID: $GPU_ID"
    exit 1
fi