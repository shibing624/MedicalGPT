SFT_MODEL=path/to/SFT_MODEL
REWARD_MODEL=path/to/REWARD_MODEL

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 rl_training.py \
    --model_name_or_path ${SFT_MODEL} \
    --reward_model_name_or_path ${REWARD_MODEL} \
    --dataset_name shibing624/medical \
    --dataset_config_name finetune \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size 1 \
    --mini_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --seed 42 \
    --fp16 \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --max_steps 1000 \
    --learning_rate 1e-5 \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 1 \
    --max_source_length 256 \
    --max_target_length 256 \
    --min_target_length 4 \
    --output_dir outputs-medical-llama-rl-v1 \
    --overwrite_output_dir \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --device_map auto \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --early_stopping True \
    --target_kl 0.1 \
    --reward_baseline 0.0
