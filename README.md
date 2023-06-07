[**🇨🇳中文**](https://github.com/shibing624/MedicalGPT/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/MedicalGPT/blob/main/README_EN.md) | [**📖文档/Docs**](https://github.com/shibing624/MedicalGPT/wiki) | [**🤖模型/Models**](https://huggingface.co/shibing624) 

<div align="center">
  <a href="https://github.com/shibing624/MedicalGPT">
    <img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/logo.png" height="100" alt="Logo">
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

分四阶段训练GPT模型，来自Andrej Karpathy的演讲PDF[State of GPT](https://karpathy.ai/stateofgpt.pdf)，视频[Video](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2)

基于此，训练领域模型--医疗模型，分四阶段：

- 第一阶段：PT(Continue PreTraining)增量预训练，在海量领域文档数据上二次预训练LLaMA模型，以注入领域知识
- 第二阶段：SFT(Supervised Fine-tuning)有监督微调，构造指令微调数据集，在预训练模型基础上做指令精调，以对齐指令意图
- 第三阶段：RM(Reward Model)奖励模型建模，构造人类偏好排序数据集，训练奖励模型，用来对齐人类偏好，主要是"HHH"原则，具体是"helpful, honest, harmless"
- 第四阶段：RL(Reinforcement Learning)基于人类反馈的强化学习(RLHF)，用奖励模型来训练SFT模型，生成模型使用奖励或惩罚来更新其策略，以便生成更高质量、更符合人类偏好的文本

## ▶️ Demo

- Hugging Face Demo: doing

我们提供了一个简洁的基于gradio的交互式web界面，启动服务后，可通过浏览器访问，输入问题，模型会返回答案。

启动服务，命令如下：
```shell
python scripts/gradio_demo.py --base_model path_to_llama_hf_dir --lora_model path_to_lora_dir
```

参数说明：

- `--base_model {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录，也可使用HF Model Hub模型调用名称
- `--lora_model {lora_model}`：LoRA文件所在目录，也可使用HF Model Hub模型调用名称。若lora权重已经合并到预训练模型，则删除--lora_model参数
- `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与--lora_model相同；若也未提供--lora_model参数，则其默认值与--base_model相同
- `--use_cpu`: 仅使用CPU进行推理
- `--gpus {gpu_ids}`: 指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如0,1,2


## 🚀 Training Pipeline

Training Stage:

| Stage                           | Introduction                             | Python script                                                                                       | Shell script                                                                                 | Notebook | Colab                                                                                                                                                                                           |                     
|:--------------------------------|:-----------------------------------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|:---------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage 1: Continue Pretraining   | 增量预训练，在海量领域文本数据上继续预训练GPT模型，以注入领域知识       | [scripts/pretraining.py](https://github.com/shibing624/MedicalGPT/blob/main/scripts/pretraining.py) | [scripts/run_pt.sh](https://github.com/shibing624/MedicalGPT/blob/main/scripts/run_pt.sh)    | [run_pretraining.ipynb](https://github.com/shibing624/MedicalGPT/blob/main/notebook/run_pretraining.ipynb)    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/notebook/run_pretraining.ipynb)           |
| Stage 2: Supervised Fine-tuning | 有监督微调，构造指令微调数据集，在预训练模型基础上做指令精调，以对齐指令意图   | [scripts/supervised_finetuning.py](https://github.com/shibing624/MedicalGPT/blob/main/scripts/supervised_finetuning.py) | [scripts/run_sft.sh](https://github.com/shibing624/MedicalGPT/blob/main/scripts/run_sft.sh)  | [notebook/run_supervised_finetuning.ipynb](https://github.com/shibing624/MedicalGPT/blob/main/notebook/run_supervised_finetuning.ipynb)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/notebook/run_supervised_finetuning.ipynb) |
| Stage 3: Reward Modeling        | RM(Reward Model)奖励模型建模，构造人类偏好排序数据集，训练奖励模型，用来对齐人类偏好，主要是"helpful, honest, harmless"原则   | [scripts/reward_modeling.py](https://github.com/shibing624/MedicalGPT/blob/main/scripts/reward_modeling.py) | [scripts/run_rm.sh](https://github.com/shibing624/MedicalGPT/blob/main/scripts/run_rm.sh)    | [notebook/run_reward_modeling.ipynb](https://github.com/shibing624/MedicalGPT/blob/main/notebook/run_reward_modeling.ipynb)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/notebook/run_reward_modeling.ipynb)       |
| Stage 4: Reinforcement Learning | 基于人类反馈的强化学习(RLHF)，用奖励模型来训练SFT模型，生成模型使用奖励或惩罚来更新其策略，以便生成更高质量、更符合人类偏好的文本   | [scripts/rl_training.py](https://github.com/shibing624/MedicalGPT/blob/main/scripts/rl_training.py) | [scripts/run_rl.sh](https://github.com/shibing624/MedicalGPT/blob/main/scripts/run_rl.sh)    | [notebook/run_rl_training.ipynb](https://github.com/shibing624/MedicalGPT/blob/main/notebook/run_rl_training.ipynb)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/notebook/run_rl_training.ipynb)           |


[训练参数说明wiki](https://github.com/shibing624/MedicalGPT/wiki/%E8%AE%AD%E7%BB%83%E7%BB%86%E8%8A%82%E8%AF%B4%E6%98%8E)
## 🔥 Inference 
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
- `--lora_model {lora_model}`：LoRA解压后文件所在目录，也可使用HF Model Hub模型调用名称。如果已经合并了LoRA权重到预训练模型，则可以不提供此参数
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

- 240万条中文医疗数据集(包括预训练、指令微调和奖励数据集)：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)

**附上一些通用数据集和医疗数据集的链接**

- 50万条中文ChatGPT指令Belle数据集：[BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
- 100万条中文ChatGPT指令Belle数据集：[BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
- 5万条英文ChatGPT指令Alpaca数据集：[50k English Stanford Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca#data-release)
- 2万条中文GPT-4指令Alpaca数据集：[shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)
- 69万条中文指令Guanaco数据集(Belle50万条+Guanaco19万条)：[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)
- 22万条中文医疗对话数据集(华佗项目)：[FreedomIntelligence/HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1)

## ✅ Todo

1. [ ] Added multi-round dialogue data fine-tuning method
2. [x] add reward model fine-tuning
3. [x] add rl fine-tuning
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
