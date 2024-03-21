echo "init"
git pull
pip install -r algorithm/llm/requirements.txt
pip install Logbook
cp ./algorithm/llm/train/merge_peft_adapter.py ./
cp ./algorithm/llm/train/inference.py ./

git lfs install
git clone

export RUN_PACKAGE=algorithm.llm.train.pretraining
export RUN_CLASS=PreTraining
export USE_MODELSCOPE_HUB=1
echo $RUN_PACKAGE.$RUN_CLASS

# pretraining
python main.py \
    --model_type bloom \
    --model_name_or_path AI-ModelScope/bloomz-560m \
    --train_file_dir ../MedicalGPT/data/pretrain \
    --validation_file_dir ../MedicalGPT/data/pretrain \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --max_train_samples 10000 \
    --max_eval_samples 10 \
    --num_train_epochs 0.5 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 10 \
    --block_size 512 \
    --group_by_length True \
    --output_dir outputs-pt-bloom-v1 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache

echo $RUN_PACKAGE.$RUN_CLASS.done
python merge_peft_adapter.py --model_type bloom \
    --base_model AI-ModelScope/bloomz-560m --lora_model outputs-pt-bloom-v1 --output_dir merged-pt/
export RUN_PACKAGE=algorithm.llm.train.supervised_finetuning
export RUN_CLASS=Supervised_Finetuning
echo $RUN_PACKAGE.$RUN_CLASS
# sft
python main.py \
    --model_type bloom \
    --model_name_or_path merged-pt \
    --train_file_dir ../MedicalGPT/data/finetune \
    --validation_file_dir ../MedicalGPT/data/finetune \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --fp16 \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 1 \
    --output_dir outputs-sft-v1 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True
echo $RUN_PACKAGE.$RUN_CLASS.done
python merge_peft_adapter.py --model_type bloom \
    --base_model merged-pt --lora_model outputs-sft-v1 --output_dir merged-sft/
export RUN_PACKAGE=algorithm.llm.train.reward_modeling
export RUN_CLASS=RewardModeling
echo $RUN_PACKAGE.$RUN_CLASS
#reward
python main.py \
    --model_type bloom \
    --model_name_or_path merged-sft \
    --train_file_dir ../MedicalGPT/data/reward \
    --validation_file_dir ../MedicalGPT/data/reward \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --use_peft True \
    --seed 42 \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.001 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --max_source_length 256 \
    --max_target_length 256 \
    --output_dir outputs-rm-v1 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype float32 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False \
    --gradient_checkpointing True
python merge_peft_adapter.py --model_type bloom \
    --base_model merged-sft --lora_model outputs-rm-v1 --output_dir merged-rm/
echo $RUN_PACKAGE.$RUN_CLASS.done
export RUN_PACKAGE=algorithm.llm.train.ppo_training
export RUN_CLASS=PPOTraining
echo $RUN_PACKAGE.$RUN_CLASS

## ppo training
python main.py \
    --model_type bloom \
    --model_name_or_path merged-sft \
    --reward_model_name_or_path merged-rm \
    --torch_dtype float16 \
    --device_map auto \
    --train_file_dir ../MedicalGPT/data/finetune \
    --validation_file_dir ../MedicalGPT/data/finetune \
    --batch_size 4 \
    --max_source_length 256 \
    --max_target_length 256 \
    --max_train_samples 1000 \
    --use_peft True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --do_train \
    --max_steps 64 \
    --learning_rate 1e-5 \
    --save_steps 50 \
    --output_dir outputs-rl-v1 \
    --early_stopping True \
    --target_kl 0.1 \
    --reward_baseline 0.0 \
    --use_fast_tokenizer
echo $RUN_PACKAGE.$RUN_CLASS.done
python merge_peft_adapter.py --model_type bloom \
    --base_model merged-sft --lora_model outputs-rl-v1 --output_dir merged-ppo/
python inference.py --model_type bloom --base_model merged-ppo --interactive