#!/usr/bin/env bash

CONFIG=$1  # 第一个参数
GPUS=$2  # 第二个参数
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}  # 有默认值，如果大于两个结点需要指定
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \  # 有几个结点(机器)
    --node_rank=$NODE_RANK \  # 第几个结点
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \   # 每个结点使用几个显卡
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch \
    ${@:3}

# 如 python -m torch.distributed.launch --nproc_per_node=8 xxx/train.py --launcher pytorch