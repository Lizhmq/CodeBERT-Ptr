export CUDA_VISIBLE_DEVICES=1
LANG=JAVA
DATADIR=../bigJava/datasets
# DATADIR=../great/
# DATADIR=../CuBert/wrong_op
# OUTPUTDIR=./save/varmis
OUTPUTDIR=./save/java
# PRETRAINDIR=microsoft/codebert-base-mlm
PRETRAINDIR=../.code-bert-cache/codebert-base
# LOGFILE=varmis.log
LOGFILE=java.log
PER_NODE_GPU=1

# 4e-5
# 1e-4 too big
# varmis 2 * 4 * 8, 2epoch
# calls 1 * 1 * 64
# batchsize 72 better than 24

# wrong op 12 * 3 * 2
# -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU
python  train_classifier.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --block_size=512 \
        --do_train \
        --node_index 0 \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=4e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=8 \
        --per_gpu_eval_batch_size=16 \
        --gradient_accumulation_steps=4 \
        --num_train_epochs=5 \
        --logging_steps=100 \
        --save_steps=4000 \
        --overwrite_output_dir \
        --seed=2233 \
        --mlm 
