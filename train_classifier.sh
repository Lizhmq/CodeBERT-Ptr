export CUDA_VISIBLE_DEVICES=0
LANG=JAVA
DATADIR=./dataset
PRETRAINDIR=microsoft/codebert-base-mlm
LOGFILE=java-op.log
PER_NODE_GPU=1

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
