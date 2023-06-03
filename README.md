<div align="center">
  <a href="https://github.com/shibing624/MedicalGPT">
    <img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/logo.png" width="120" alt="Logo">
  </a>
</div>

-----------------

# MedicalGPT: Training Medical GPT Model
[![HF Models](https://img.shields.io/badge/Hugging%20Face-shibing624-green)](https://huggingface.co/shibing624)
[![Github Stars](https://img.shields.io/github/stars/shibing624/MedicalGPT?color=yellow)](https://star-history.com/#shibing624/MedicalGPT&Timeline)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/MedicalGPT.svg)](https://github.com/shibing624/MedicalGPT/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

## 📖 Introduction

**MedicalGPT** training medical GPT model with ChatGPT training pipeline, implemantation of Pretraining, 
Supervised Finetuning, Reward Modeling and Reinforcement Learning.

**MedicalGPT** 训练医疗大模型，实现包括二次预训练、有监督微调、奖励建模、强化学习训练。

<img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/GPT_Training.jpg" width="860" />

训练领域模型--医疗模型，分四阶段：

- 第一阶段：PT(Continue PreTraining)增量预训练，在海量领域文档数据上二次预训练LLaMA模型，以注入领域知识，如有需要可以扩充领域词表，比如医疗领域词表
- 第二阶段：SFT(Supervised Fine-tuning)有监督微调，构造指令微调数据集，在预训练模型基础上做指令精调，以对齐指令意图
- 第三阶段：RM(Reward Model)奖励模型，构造人类偏好排序数据集，训练奖励模型，用来对齐人类偏好，主要是"HHH"原则，具体是"helpful, honest, harmless"
- 第四阶段：RL(Reinforcement Learning)基于人类反馈的强化学习(RLHF)，用奖励模型来训练SFT模型，生成模型使用奖励或惩罚来更新其策略，以便生成更高质量、更符合人类偏好的文本

## ▶️ Demo

- Hugging Face Demo: doing

我们提供了一个简洁的基于gradio的交互式web界面，启动服务后，可通过浏览器访问，输入问题，模型会返回答案。

1. 安装依赖库：
```shell
pip install gradio
pip install mdtex2html
```

2. 启动服务，命令如下：
```shell
python scripts/gradio_demo.py --base_model path_to_llama_hf_dir --lora_model path_to_lora_dir
```

如果已经执行了`scripts/merge_peft_adapter.py`脚本将lora权重合并到预训练模型，那么无需再指定--lora_model：
```shell
python scripts/gradio_demo.py --base_model path_to_merged_alpaca_hf_dir 
```

参数说明：

- `--base_model {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录，也可使用HF Model Hub模型调用名称
- `--lora_model {lora_model}`：LoRA解压后文件所在目录，也可使用HF Model Hub模型调用名称。若不提供此参数，则只加载--base_model指定的模型
- `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与--lora_model相同；若也未提供--lora_model参数，则其默认值与--base_model相同
- `--use_cpu`: 仅使用CPU进行推理
- `--gpus {gpu_ids}`: 指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如0,1,2


## 🚀 Training Pipeline

### Stage 1: Continue Pretraining
基于llama-7b模型，使用医疗百科类数据继续预训练，期望注入医疗知识到预训练模型，得到llama-7b-pt模型，此步骤可选

Continue pretraining of the base llama-7b model to create llama-7b-pt:

```shell
torchrun --nnodes 1 --nproc_per_node 8 scripts/run_pretraining.py \
    --model_name_or_path minlik/chinese-llama-plus-7b-merged \
    --tokenizer_name_or_path minlik/chinese-llama-plus-7b-merged \
    --dataset_name shibing624/medical \
    --dataset_config_name pretrain \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --seed 42 \
    --fp16 \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --num_train_epochs 1.0 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 8 \
    --block_size 1024 \
    --output_dir outputs-pt \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules q_proj,v_proj,k_proj,o_proj \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --device_map auto \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --ddp_find_unused_parameters False
```

[参数说明](#参数说明)

### Stage 2: Supervised FineTuning
基于llama-7b-pt模型，使用医疗问答类数据进行有监督微调，得到llama-7b-sft模型

Supervised fine-tuning of the base llama-7b-pt model to create llama-7b-sft

```shell
torchrun --nnodes 1 --nproc_per_node 8 scripts/run_supervised_finetuning.py \
    --model_name_or_path <LLAMA_MODEL_PATH> \
    --tokenizer_name_or_path <LLAMA_MODEL_PATH> \
    --dataset_name shibing624/medical \
    --dataset_config_name finetune \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --seed 42 \
    --fp16 \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --num_train_epochs 5.0 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 8 \
    --max_length 512 \
    --output_dir outputs-sft \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules q_proj,v_proj,k_proj,o_proj \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --device_map auto \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --ddp_find_unused_parameters False
```

[参数说明](#参数说明)

### Stage 3: Reward Modeling

RM(Reward Model)：奖励模型，原则上，我们可以直接用人类标注来对模型做 RLHF 微调。

然而，这将需要我们给人类发送一些样本，在每轮优化后计分。这是贵且慢的，因为收敛需要的训练样本量大，而人类阅读和标注的速度有限。
一个比直接反馈更好的策略是，在进入 RL 循环之前用人类标注集来训练一个奖励模型RM。奖励模型的目的是模拟人类对文本的打分。

构建奖励模型的最佳实践是预测结果的排序，即对每个 prompt (输入文本) 对应的两个结果 (yk, yj)，模型预测人类标注的比分哪个更高。
RM模型是通过人工标注SFT模型的打分结果来训练的，目的是取代人工打分，本质是个回归模型，用来对齐人类偏好，主要是"HHH"原则，具体是"helpful, honest, harmless"。


基于llama-7b-sft模型，使用医疗问答偏好数据训练奖励偏好模型，训练得到llama-7b-reward模型

Reward modeling using dialog pairs from the reward dataset using the llama-7b-sft to create llama-7b-reward:

```shell
torchrun --nnodes 1 --nproc_per_node 8 scripts/run_reward_modeling.py \
    --model_name_or_path <LLAMA_SFT_MODEL> \
    --tokenizer_name_or_path <LLAMA_SFT_MODEL> \
    --dataset_name shibing624/medical \
    --dataset_config_name reward \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --seed 42 \
    --fp16 \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --num_train_epochs 5.0 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 8 \
    --max_length 512 \
    --output_dir outputs-reward \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules q_proj,v_proj,k_proj,o_proj \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --device_map auto \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --ddp_find_unused_parameters False
```
[参数说明](#参数说明)

### Stage 4: Reinforcement Learning

RL(Reinforcement Learning)模型的目的是最大化奖励模型的输出，基于上面步骤，我们有了微调的语言模型(llama-7b-sft)和奖励模型(llama-7b-reward)，
可以开始执行 RL 循环了。

这个过程大致分为三步：

1. 输入prompt，模型生成答复
2. 用奖励模型来对答复评分
3. 基于评分，进行一轮策略优化的强化学习(PPO)

<img src=https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/stackllama/trl_loop.png height=400 />


基于llama-7b-reward模型 RL 微调训练llama-7b-sft模型，得到llama-7b-rl模型

Reinforcement Learning fine-tuning of llama-7b-sft with the llama-7b-reward reward model to create llama-7b-rl

```shell
torchrun --nnodes 1 --nproc_per_node 8 scripts/run_rl_training.py \
    --model_name_or_path <LLAMA_SFT_MODEL> \
    --reward_model_name_or_path <LLAMA_REWARD_MODEL> \
    --tokenizer_name_or_path <LLAMA_TOKENIZER> \
    --dataset_name shibing624/medical \
    --dataset_config_name finetune \
    --validation_split_percentage 0.001 \
    --mini_batch_size 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --seed 42 \
    --fp16 \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --num_train_epochs 5.0 \
    --learning_rate 1.4e-5 \
    --save_steps 50 \
    --save_strategy steps \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 8 \
    --max_length 512 \
    --output_max_length 128 \
    --output_dir outputs-rl \
    --overwrite_output_dir \
    --logging_first_step True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules q_proj,v_proj,k_proj,o_proj \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --device_map auto \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --early_stopping True \
    --target_kl 0.1 \
    --reward_baseline 0.0
```

### 参数说明

1. 如果想要单卡训练，仅需将nproc_per_node设置为1即可
2. 默认预训练模型是LLaMA，如果训练其他GPT模型，适当调整`tokenzier_name_or_path`和`model_name_or_path`即可
3. 如果运行环境支持deepspeed，加上`--deepspeed deepspeed_config.json`
4. 如果gpu支持int8，加上`--load_in_8bit True`代表采用8bit量化训练，可显著减少显存占用

**关于LoRA Training**

默认使用LoRA训练，每个stage的LoRA模型权重都需要合并到base model中，使用以下命令合并，下一个stage的`model_name_or_path`指定为合并后的模型文件夹。

LoRA layers were using at all stages to reduce memory requirements. 
At each stage the peft adapter layers were merged with the base model, using: 
```shell
python scripts/merge_peft_adapter.py --base_model_name_or_path X_folder --peft_model_path Y_folder --output_dir X_folder
```

- this script requires `peft>=0.3.0`
- 合并后的权重保存在output_dir目录下，后续可通过from_pretrained直接加载

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
└── config.json
|-- logs
|   |-- 1685436851.18595
|   |   `-- events.out.tfevents.1685436851.ts-89f5028ad154472e99e7bcf2c9bf2343-launcher.82684.1
```

trainer_state.json记录了loss、learning_rate的变化

logs目录下的文件可用于tensorboard可视化，启动tensorboard命令如下：
```shell
tensorboard --logdir output_dir/logs --host 0.0.0.0 --port 8008
```


**关于deepspeed**

deepspeed 的参数配置`deepspeed_config.json`可参考：

1. https://www.deepspeed.ai/docs/config-json/
2. https://huggingface.co/docs/accelerate/usage_guides/deepspeed
3. https://github.com/huggingface/transformers/blob/main/tests/deepspeed

如果显存充足，可优先考虑stage 2，对应的配置文件是deepspeed_config.json。如果显存不足，可采用stage 3，该模式采用模型参数并行，可显著减小显存占用，但是训练速度会变慢很多。


**关于多机多卡训练**

以两台机器为例，每台机器上有8张卡

```shell
node_rank=$1
echo ${node_rank}
master_addr="10.111.112.223"

torchrun --nproc_per_node 8 --nnodes 2 --master_addr ${master_addr} --master_port 14545 --node_rank ${node_rank} srcipts/run_supervised_finetuning.py ... 
```

- node_rank 代表节点的rank，第一台机器（主机器）的node_rank设置为0，第二台机器的node_rank设置为1
- nnodes 代表节点机器的数量
- master_addr 代表主机器的ip地址
- master_port 代表与主机器通信的端口号


## 😊 Inference 
训练完成后，现在我们加载训练好的模型，验证模型生成文本的效果。

```shell
python scripts/inference.py \
    --base_model path_to_llama_hf_dir \
    --lora_model path_to_lora \
    --with_prompt \
    --interactive
```

参数说明：

- `--base_model {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录
- `--lora_model {lora_model}`：LoRA解压后文件所在目录，也可使用🤗Model Hub模型调用名称。如果已经合并了LoRA权重到预训练模型，则可以不提供此参数
- `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与--lora_model相同；若也未提供--lora_model参数，则其默认值与--base_model相同
- `--with_prompt`：是否将输入与prompt模版进行合并。如果加载Alpaca模型，请务必启用此选项！
- `--interactive`：以交互方式启动，以便进行多次单轮问答
- `--data_file {file_name}`：非交互方式启动下，按行读取file_name中的的内容进行预测
- `--predictions_file {file_name}`：非交互式方式下，将预测的结果以json格式写入file_name
- `--use_cpu`: 仅使用CPU进行推理
- `--gpus {gpu_ids}`: 指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如0,1,2




#### Inference Examples

<details>

| Human | Assitant |
| ----- | ------ |
| 怎么让自己精力充沛，列5点建议 | 以下是一些可以让你保持精力充沛的建议： <br/>1. 睡眠充足。确保每天晚上得到充足的睡眠时间。 <br/>2. 多运动。进行适度的有氧运动，如慢跑、游泳等，可以帮助你提高能量和耐力。<br/>3. 吃健康食品。选择富含蛋白质的食物，如鸡肉、鱼肉和鸡蛋，以及新鲜的水果和蔬菜。 <br/>4. 减少咖啡因的摄入量。尽量避免在白天喝含咖啡因的饮料或吃含有咖啡因的食物。 <br/>5. 放松身心。尝试冥想、瑜伽或其他形式的放松活动来减轻压力和焦虑感。 |

</details>
<br/>


## 📚 Dataset 

- 240万条中文医疗数据集(包括预训练、指令微调和奖励数据集)：[https://huggingface.co/datasets/shibing624/medical](https://huggingface.co/datasets/shibing624/medical)

## ✅ Todo

1. [ ] 新增多轮对话数据微调方法
2. [x] add reward model finetuning
3. [x] add rl finetuning
4. [x] add medical reward dataset
5. [x] add llama in8/int4 training
6. [ ] add all training and predict demo in colab

## ☎️ Contact

- Issue(建议)
  ：[![GitHub issues](https://img.shields.io/github/issues/shibing624/MedicalGPT.svg)](https://github.com/shibing624/MedicalGPT/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我： 加我*微信号：xuming624, 备注：姓名-公司名-NLP* 进NLP交流群。

<img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/wechat.jpeg" width="200" />

## ⚠️ 局限性、使用限制与免责声明

基于当前数据和基础模型训练得到的SFT模型，在效果上仍存在以下问题：

1. 在涉及事实性的指令上可能会产生违背事实的错误回答。

2. 对于具备危害性的指令无法很好的鉴别，由此会产生危害性言论。

3. 在一些涉及推理、代码、多轮对话等场景下模型的能力仍有待提高。

基于以上模型局限性，我们要求开发者仅将我们开源的模型权重及后续用此项目生成的衍生物用于研究目的，不得用于商业，以及其他会对社会带来危害的用途。

本项目仅可应用于研究目的，项目开发者不承担任何因使用本项目（包含但不限于数据、模型、代码等）导致的危害或损失。详细请参考[免责声明](https://github.com/shibing624/MedicalGPT/blob/main/DISCLAIMER)。

项目代码的授权协议为 [The Apache License 2.0](/LICENSE)，代码可免费用做商业用途，模型权重和数据只能用于研究目的。请在产品说明中附加MedicalGPT的链接和授权协议。


## 😇 Citation

如果你在研究中使用了MedicalGPT，请按如下格式引用：

```latex
@misc{MedicalGPT,
  title={MedicalGPT: Training Medical GPT Model},
  author={Ming Xu},
  year={2023},
  howpublished={\url{https://github.com/shibing624/MedicalGPT}},
}
```

## 😍 Contribute

项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

- 在`tests`添加相应的单元测试
- 使用`python -m pytest`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

## 💕 Acknowledgements 

- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/finetune.py)
- [ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

Thanks for their great work!
