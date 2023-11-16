#多机多卡训练

#--nnodes=1:3 表示 使用一到三台机器 弹性分配资源
#--nnodes=<最小节点数>:<最大节点数>
#--nproc_per_node=每台机器上可用的GPU数
#--rdzv_endpoint=主节点（最先启动的）ip:端口号
#其他不需要变

#注意： 此版本的分布式训练是基于数据并行的，多机多卡相当于开更大的batchsize，此时epoch迭代速度会增加,
#但由于 该版本的代码中 保存模型是按照global step来计算的，所以会出现的效果就是 ： 保存模型的时间不会有明显加速，
#但每次保存模型时epoch都比之前迭代了更多次,也就是 “更少的步数，实现更好的效果”

#*************************
# torchrun \
#     --nnodes=1:3\
#     --nproc_per_node=2\
#     --rdzv_id=1\
#     --rdzv_backend=c10d\
#     --rdzv_endpoint="inspur1:8880"\
#     train_ms.py
#****************************

#多卡训练
#nproc_per_node = 机器上可用的GPU数

#*************************
torchrun \
    --nnodes=1\
    --nproc_per_node=2\
    train_ms.py
#*************************
