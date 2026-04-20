CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training/opd_training.py \
    --model_name_or_path Qwen/Qwen3.5-2B \
    --teacher_model_name_or_path Qwen/Qwen3.5-7B-Instruct \
    --train_file_dir ./data/sft \
    --validation_file_dir ./data/sft \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --teacher_load_in_4bit True \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --max_prompt_length 1024 \
    --max_new_tokens 512 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --warmup_steps 5 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --output_dir outputs-opd-qwen-v1 \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --opd_lambda 0.5 \
    --opd_beta 0.5 \
    --temperature 0.9 \
    --tool_format default \
    --cache_dir ./cache

# Notes:
# 1. Student and teacher should share the same tokenizer family / chat template whenever possible.
# 2. Teacher is frozen in OPD v1; only the student is updated.
# 3. If the tokenizer does not provide a built-in chat_template, pass --template_name explicitly.
