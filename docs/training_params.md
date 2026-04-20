## 训练脚本


- 第一阶段：PT(Continue PreTraining)增量预训练 `run_pt.sh`
- 第二阶段：SFT(Supervised Fine-tuning)有监督微调 `run_sft.sh`
- 第三阶段
  - RLHF(Reinforcement Learning from Human Feedback)分为两步：
    - RM(Reward Model)奖励模型建模 `run_rm.sh`
    - RL(Reinforcement Learning)基于人类反馈的强化学习 `run_ppo.sh`
  - DPO(Direct Preference Optimization)直接偏好优化 `run_dpo.sh`
  - OPD(On-Policy Distillation)独立蒸馏 `run_opd.sh`

## 训练参数说明

1. 如果想要单卡训练，仅需将nproc_per_node设置为1即可，或者去掉torchrun命令，直接运行python脚本，如`python supervised_finetuning.py`
2. 指定训练的base模型（默认llama），训练代码也兼容ChatGLM/BLOOM/BaiChuan等GPT模型，以baichuan模型为例，调整`--model_name_or_path baichuan-inc/Baichuan-13B-Chat`，特别的，如果未训练只推理，base model是类似`baichuan-inc/Baichuan-13B-Chat`已经对齐的模型，则需要指定`--template_name baichuan`；如果在base model基础上训练，默认采用`vicuna`模板，后续用训练好的模型推理时，也指定相同的`--template_name vicuna`即可
3. 指定训练集，`--train_file_dir`指定训练数据目录，`--validation_file_dir`指定验证数据目录，如果不指定，默认使用`--dataset_name`指定的HF datasets数据集，训练集字段格式见[数据集格式](https://github.com/shibing624/MedicalGPT/wiki/%E6%95%B0%E6%8D%AE%E9%9B%86)，建议领域训练集中加入一些通用对话数据，数据集链接见[📚 Dataset](https://github.com/shibing624/MedicalGPT#-dataset)，当前默认多轮对话格式，兼容单轮对话，微调训练集如果是alpaca格式，可以用[convert_dataset.py](https://github.com/shibing624/MedicalGPT/blob/main/convert_dataset.py)转为shareGPT格式，即可传入训练
4. 如果运行环境支持deepspeed，加上`--deepspeed zero2.json`参数启动zero2模式；显存不足，加上`--deepspeed zero3.json --fp16`参数启动zero3混合精度模式
5. 如果gpu支持int8/int4量化，加上`--load_in_4bit True`代表采用4bit量化训练，或者`--load_in_8bit True`代表采用8bit量化训练，均可显著减少显存占用
6. 训练集条数控制，`--max_train_samples`和`--max_eval_samples`指定训练和验证数据集的最大样本数，用于快速验证代码是否可用，训练时建议设置为`--max_train_samples -1`表示用全部训练集，`--max_eval_samples 50`表示用50条验证数据
7. 训练方式，指定`--use_peft False`为全参训练（要移除`--fp16`），`--use_peft True`是LoRA训练；注意：全参训练LLaMA-7B模型需要120GB显存，LoRA训练需要13GB显存
8. 支持恢复训练，LoRA训练时指定`--peft_path`为旧的adapter_model.bin所在文件夹路径；全参训练时指定`--resume_from_checkpoint`为旧模型权重的文件夹路径
9. PT和SFT支持qlora训练，如果使用的是 RTX4090、A100 或 H100 GPU，支持nf4，使用`--qlora True --load_in_4bit True`参数启用qlora训练，开启qlora训练，会减少显存占用，训练加速，同时建议设置`--torch_dtype bfloat16 --optim paged_adamw_32bit`保证训练精度
10. 扩词表后的增量预训练，PT阶段加上`--modules_to_save embed_tokens,lm_head`参数，后续SFT等阶段不用加
11. 新增了RoPE插值来扩展GPT模型的上下文长度，通过[位置插值方法](https://arxiv.org/abs/2306.15595)，在增量数据上进行训练，使模型获得长文本处理能力，使用 `--rope_scaling linear` 参数训练模型，使用`--rope_scaling dynamic` 参数预测模型
12. 针对LLaMA模型支持了[FlashAttention-2](https://github.com/Dao-AILab/flash-attention)，如果您使用的是 RTX3090、RTX4090、A100 或 H100 GPU，SFT中请使用 `--flash_attn` 参数以启用 FlashAttention-2
13. 新增了[LongLoRA](https://github.com/dvlab-research/LongLoRA) 提出的 **$S^2$-Attn**，使模型获得长文本处理能力，SFT中使用 `--shift_attn` 参数以启用该功能
14. 支持了[NEFTune](https://github.com/neelsjain/NEFTune)给embedding加噪SFT训练方法，[NEFTune paper](https://arxiv.org/abs/2310.05914), SFT中使用 `--neft_alpha` 参数启用 NEFTune，例如 `--neft_alpha 5`
15. 支持微调Mixtral混合专家MoE模型 **[Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)**，SFT中如果用lora微调模型，可以开启4bit量化和QLoRA`--load_in_4bit True --qlora True`以节省显存，建议设置`--target_modules q_proj,k_proj,v_proj,o_proj`，这样可以避免对MoE专家网络的MLP层量化，因为它们很稀疏且量化后会导致性能效果下降。
16. 独立OPD训练依赖 `trl>=0.29.0` 的 `trl.experimental.gkd.GKDTrainer`，更新依赖后可直接运行 `bash scripts/run_opd.sh`
17. OPD第一版复用 `data/sft` 的 ShareGPT 多轮对话数据格式，每条样本会拆成若干个以 assistant turn 结尾的 `messages` 训练样本，不需要 `chosen/rejected` 偏好数据
18. OPD常用参数：`--teacher_model_name_or_path` 指定更强的teacher，`--max_prompt_length` 控制提示词长度，`--max_new_tokens` 控制on-policy rollout长度，`--opd_lambda` 对应 on-policy rollout 比例，`--opd_beta` 对应 GKD/JSD 的KL插值，`--temperature` 和 `--seq_kd` 直接映射到 `GKDConfig`
19. OPD建议默认只训练student，teacher保持冻结；显存紧张时可给teacher加 `--teacher_load_in_4bit True` 或 `--teacher_load_in_8bit True`
20. OPD最好让student和teacher共用同一tokenizer family / chat template；如果底模tokenizer没有内置 `chat_template`，请显式传 `--template_name`


**关于LoRA Training**

默认使用LoRA训练，每个stage的LoRA模型权重都需要合并到base model中，使用以下命令合并，下一个stage的`model_name_or_path`指定为合并后的模型文件夹。

LoRA layers were using at all stages to reduce memory requirements.
At each stage the peft adapter layers were merged with the base model, using:
```shell
python merge_peft_adapter.py \
  --base_model base_model_dir \
  --tokenizer_path base_model_dir \
  --lora_model lora_model_dir \
  --output_dir outputs-merged
```

- this script requires `peft>=0.4.0`
- 合并后的权重保存在output_dir目录下，后续可通过from_pretrained直接加载
- OPD的LoRA输出与SFT/DPO相同，也可以用同样的方式合并和部署

**关于模型结果**

训练日志和模型保存在output_dir目录下，目录下的文件结构如下：

```shell
output_dir/
|-- adapter_config.json
|-- adapter_model.bin
|-- checkpoint-24000
|   |-- adapter_config.json
|   |-- adapter_model.bin
|   |-- trainer_state.json
|   `-- training_args.bin
|-- train_results.txt
|-- eval_results.txt
|-- special_tokens_map.json
|-- tokenizer_config.json
|-- training_args.bin
|-- logs
|   |-- 1685436851.18595
|   |   `-- events.out.tfevents.1685436851.ts-89f5028ad154472e99e7bcf2c9bf2343-launcher.82684.1
└── config.json

```

- `trainer_state.json`记录了loss、learning_rate的变化
- logs目录下的文件可用于tensorboard可视化，启动tensorboard命令如下：
```shell
tensorboard --logdir output_dir/logs --host 0.0.0.0 --port 8008
```


**关于deepspeed**

deepspeed 的参数配置`deepspeed_config.json`可参考：

1. https://www.deepspeed.ai/docs/config-json/
2. https://huggingface.co/docs/accelerate/usage_guides/deepspeed
3. https://github.com/huggingface/transformers/blob/main/tests/deepspeed

如果显存充足，可优先考虑stage 2，对应的配置文件是`deepspeed_zero_stage2_config.json`。如果显存不足，可采用stage 3，对应的配置文件是`deepspeed_zero_stage3_config.json`，该模式采用模型参数并行，可显著减小显存占用，但是训练速度会变慢很多。


**关于多机多卡训练**

以两台机器为例，每台机器上有8张卡

```shell
node_rank=$1
echo ${node_rank}
master_addr="10.111.112.223"

torchrun --nproc_per_node 8 --nnodes 2 --master_addr ${master_addr} --master_port 14545 --node_rank ${node_rank} run_supervised_finetuning.py ...
```


- node_rank 代表节点的rank，第一台机器（主机器）的node_rank设置为0，第二台机器的node_rank设置为1
- nnodes 代表节点机器的数量
- master_addr 代表主机器的ip地址
- master_port 代表与主机器通信的端口号

以上命令在两台机器各执行一次，两台机器的node_rank设置不同。
