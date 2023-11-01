torchrun \
    --nnodes=1:3\
    --nproc_per_node=2\
    --rdzv_id=1\
    --rdzv_backend=c10d\
    --rdzv_endpoint=""\
    train_ms.py