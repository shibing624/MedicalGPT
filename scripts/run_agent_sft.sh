# Agent SFT: tool call finetuning
# Supports mixed training with normal SFT data and tool call data
# --train_file_dir can point to a folder containing both normal and tool call jsonl/json files
# --tool_format controls how function_call / observation roles are formatted

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training/supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen3.5-2B \
    --train_file_dir ./data/toolcall \
    --validation_file_dir ./data/toolcall \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
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
    --output_dir outputs-agent-sft-v1 \
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
    --tool_format default \
    --cache_dir ./cache
