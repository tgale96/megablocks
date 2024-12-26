#!/bin/bash

EXP_DIR=$1

# 512 * 1k * 400k = 200b tokens.
# 512 * 1k * 200k = 100b tokens.
# 512 * 1k * 100k = 50b tokens (default).
# 512 * 1k * 20k = 10b tokens.
TRAINING_STEPS=20000
if [ -n "${2}" ]; then
    TRAINING_STEPS=$2;
fi

NUM_EXPERTS=64
if [ -n "${3}" ]; then
    NUM_EXPERTS=$3;
fi

CAPACITY_FACTOR=1
if [ -n "${4}" ]; then
    CAPACITY_FACTOR=$4;
fi

TOP_K=1
if [ -n "${5}" ]; then
    TOP_K=$5;
fi

LOSS_WEIGHT=0.1
if [ -n "${6}" ]; then
    LOSS_WEIGHT=$6;
fi

BATCH_SIZE=8
if [ -n "${7}" ]; then
    BATCH_SIZE=$7;
fi

##
### Pre-training for MoE 356M parameter.
##

# MoE hyperparameters.
MOE_ARGUMENTS="\
--moe-num-experts=${NUM_EXPERTS} \
--moe-capacity-factor=${CAPACITY_FACTOR} \
--moe-loss-weight=${LOSS_WEIGHT} \
--moe-top-k=${TOP_K}"

# Distributed hyperparameters.
DISTRIBUTED_ARGUMENTS="\
--nproc_per_node 8 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000"

# Model hyperparameters.
MODEL_ARGUMENTS="\
--num-layers 24 \
--hidden-size 1024 \
--num-attention-heads 16 \
--seq-length 1024 \
--max-position-embeddings 1024"

# Training hyperparameters.
TRAINING_ARGUMENTS="\
--micro-batch-size ${BATCH_SIZE} \
--global-batch-size 512 \
--train-iters ${TRAINING_STEPS} \
--lr-decay-iters ${TRAINING_STEPS} \
--lr 0.00015 \
--min-lr 0.00001 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--init-method-std 0.01"

PILE_DATASET="\
1.0 \
../pile_gpt2/01_text_document \
1.0 \
../pile_gpt2/02_text_document \
1.0 \
../pile_gpt2/03_text_document \
1.0 \
../pile_gpt2/04_text_document \
1.0 \
../pile_gpt2/05_text_document \
1.0 \
../pile_gpt2/06_text_document \
1.0 \
../pile_gpt2/07_text_document \
1.0 \
../pile_gpt2/08_text_document \
1.0 \
../pile_gpt2/09_text_document \
1.0 \
../pile_gpt2/10_text_document \
1.0 \
../pile_gpt2/11_text_document \
1.0 \
../pile_gpt2/12_text_document \
1.0 \
../pile_gpt2/13_text_document \
1.0 \
../pile_gpt2/14_text_document \
1.0 \
../pile_gpt2/15_text_document \
1.0 \
../pile_gpt2/16_text_document \
1.0 \
../pile_gpt2/17_text_document \
1.0 \
../pile_gpt2/18_text_document \
1.0 \
../pile_gpt2/19_text_document \
1.0 \
../pile_gpt2/20_text_document \
1.0 \
../pile_gpt2/21_text_document \
1.0 \
../pile_gpt2/22_text_document \
1.0 \
../pile_gpt2/23_text_document \
1.0 \
../pile_gpt2/24_text_document \
1.0 \
../pile_gpt2/25_text_document \
1.0 \
../pile_gpt2/26_text_document \
1.0 \
../pile_gpt2/27_text_document \
1.0 \
../pile_gpt2/28_text_document \
1.0 \
../pile_gpt2/29_text_document"

# NOTE: We don't train for enough tokens for the
# split to matter.
DATA_ARGUMENTS="\
--data-path ${PILE_DATASET} \
--vocab-file ../gpt2-vocab.json \
--merge-file ../gpt2-merges.txt \
--make-vocab-size-divisible-by 1024 \
--split 969,30,1"

COMPUTE_ARGUMENTS="\
--fp16 \
--DDP-impl local \
--moe-expert-model-parallelism \
--no-async-tensor-model-parallel-allreduce"

CHECKPOINT_ARGUMENTS="\
--save-interval 2000 \
--save ./${EXP_DIR}"

EVALUATION_ARGUMENTS="\
--eval-iters 100 \
--log-interval 100 \
--eval-interval 1000"

torchrun ${DISTRIBUTED_ARGUMENTS} \
       third_party/Megatron-LM/pretrain_gpt.py \
       ${MOE_ARGUMENTS} \
       ${MODEL_ARGUMENTS} \
       ${TRAINING_ARGUMENTS} \
       ${DATA_ARGUMENTS} \
       ${COMPUTE_ARGUMENTS} \
       ${CHECKPOINT_ARGUMENTS} \
       ${EVALUATION_ARGUMENTS} |& tee ./${EXP_DIR}/train.log
