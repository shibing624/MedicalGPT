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

分四阶段训练GPT模型，来自Andrej Karpathy的演讲PDF [State of GPT](https://karpathy.ai/stateofgpt.pdf)，视频 [Video](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2)

## 🔥 News
[2023/08/02] v1.3版本: 新增LLaMA, LLaMA2, Bloom, ChatGLM, ChatGLM2, Baichuan模型的多轮对话微调训练；新增领域词表扩充功能；新增中文预训练数据集和中文ShareGPT微调训练集，详见[Release-v1.3](https://github.com/shibing624/MedicalGPT/releases/tag/1.3.0)

[2023/07/13] v1.1版本: 发布中文医疗LLaMA-13B模型[shibing624/ziya-llama-13b-medical-merged](https://huggingface.co/shibing624/ziya-llama-13b-medical-merged)，基于Ziya-LLaMA-13B-v1模型，SFT微调了一版医疗模型，医疗问答效果有提升，发布微调后的完整模型权重，详见[Release-v1.1](https://github.com/shibing624/MedicalGPT/releases/tag/1.1)

[2023/06/15] v1.0版本: 发布中文医疗LoRA模型[shibing624/ziya-llama-13b-medical-lora](https://huggingface.co/shibing624/ziya-llama-13b-medical-lora)，基于Ziya-LLaMA-13B-v1模型，SFT微调了一版医疗模型，医疗问答效果有提升，发布微调后的LoRA权重，详见[Release-v1.0](https://github.com/shibing624/MedicalGPT/releases/tag/1.0.0)

[2023/06/05] v0.2版本: 以医疗为例，训练领域大模型，实现了四阶段训练：包括二次预训练、有监督微调、奖励建模、强化学习训练。详见[Release-v0.2](https://github.com/shibing624/MedicalGPT/releases/tag/0.2.0)


## 😊 Feature
基于ChatGPT Training Pipeline，本项目实现了领域模型--医疗模型的四阶段训练：

- 第一阶段：PT(Continue PreTraining)增量预训练，在海量领域文档数据上二次预训练GPT模型，以注入领域知识
- 第二阶段：SFT(Supervised Fine-tuning)有监督微调，构造指令微调数据集，在预训练模型基础上做指令精调，以对齐指令意图
- 第三阶段：RM(Reward Model)奖励模型建模，构造人类偏好排序数据集，训练奖励模型，用来对齐人类偏好，主要是"HHH"原则，具体是"helpful, honest, harmless"
- 第四阶段：RL(Reinforcement Learning)基于人类反馈的强化学习(RLHF)，用奖励模型来训练SFT模型，生成模型使用奖励或惩罚来更新其策略，以便生成更高质量、更符合人类偏好的文本


### Release Models


| Model                                                                                                       | Base Model                                                                                                     | Introduction                                                                                                                                                                 | 
|:------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [shibing624/ziya-llama-13b-medical-lora](https://huggingface.co/shibing624/ziya-llama-13b-medical-lora)     | [IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)                              | 在240万条中英文医疗数据集[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)上SFT微调了一版Ziya-LLaMA-13B模型，医疗问答效果有提升，发布微调后的LoRA权重(单轮对话)                                 |
| [shibing624/ziya-llama-13b-medical-merged](https://huggingface.co/shibing624/ziya-llama-13b-medical-merged) | [IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)                              | 在240万条中英文医疗数据集[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)上SFT微调了一版Ziya-LLaMA-13B模型，医疗问答效果有提升，发布微调后的完整模型权重(单轮对话)                                 |
| [shibing624/vicuna-baichuan-13b-chat-lora](https://huggingface.co/shibing624/vicuna-baichuan-13b-chat-lora) | [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)                        | 在10万条多语言ShareGPT GPT4多轮对话数据集[shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4)上SFT微调了一版baichuan-13b-chat多轮问答模型，日常问答和医疗问答效果有提升，发布微调后的LoRA权重 |
| [shibing624/vicuna-baichuan-13b-chat](https://huggingface.co/shibing624/vicuna-baichuan-13b-chat)           | [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)                        | 在10万条多语言ShareGPT GPT4多轮对话数据集[shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4)上SFT微调了一版baichuan-13b-chat多轮问答模型，日常问答和医疗问答效果有提升，发布微调后的完整模型权重 |

演示[shibing624/vicuna-baichuan-13b-chat](https://huggingface.co/shibing624/vicuna-baichuan-13b-chat)模型效果：
<img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/demo-screen.gif" width="860" />
具体case见[Inference Examples](#inference-examples)

## ▶️ Demo


我们提供了一个简洁的基于gradio的交互式web界面，启动服务后，可通过浏览器访问，输入问题，模型会返回答案。

启动服务，命令如下：
```shell
CUDA_VISIBLE_DEVICES=0 python gradio_demo.py --model_type base_model_type --base_model path_to_llama_hf_dir --lora_model path_to_lora_dir
```

参数说明：

- `--model_type {base_model_type}`：预训练模型类型，如llama、bloom、chatglm等
- `--base_model {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录，也可使用HF Model Hub模型调用名称
- `--lora_model {lora_model}`：LoRA文件所在目录，也可使用HF Model Hub模型调用名称。若lora权重已经合并到预训练模型，则删除--lora_model参数
- `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与--base_model相同
- `--only_cpu`: 仅使用CPU进行推理
- `--gpus {gpu_ids}`: 指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如0,1,2
- `--resize_emb`：是否调整embedding大小，若不调整，则使用预训练模型的embedding大小，默认不调整


## 💾 Install
#### Updating the requirements
From time to time, the `requirements.txt` changes. To update, use this command:

```markdown
git clone https://github.com/shibing624/MedicalGPT
conda activate gpt
cd MedicalGPT
pip install -r requirements.txt --upgrade
```

## 🚀 Training Pipeline

Training Stage:

| Stage                           | Introduction |  Python script                                                                                                           | Shell script                                                                        |                      
|:--------------------------------|:-------------|:------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
| Stage 1: Continue Pretraining   | 增量预训练        |          [pretraining.py](https://github.com/shibing624/MedicalGPT/blob/main/pretraining.py)                     | [run_pt.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_pt.sh)   | 
| Stage 2: Supervised Fine-tuning | 有监督微调        | [supervised_finetuning.py](https://github.com/shibing624/MedicalGPT/blob/main/supervised_finetuning.py) | [run_sft.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_sft.sh) | 
| Stage 3: Reward Modeling        | 奖励模型建模       | [reward_modeling.py](https://github.com/shibing624/MedicalGPT/blob/main/reward_modeling.py)             | [run_rm.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_rm.sh)   | 
| Stage 4: Reinforcement Learning | 强化学习         |  [rl_training.py](https://github.com/shibing624/MedicalGPT/blob/main/rl_training.py)                     | [run_rl.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_rl.sh)   | 

- 提供完整四阶段串起来训练的pipeline：[run_training_pipeline.ipynb](https://github.com/shibing624/MedicalGPT/blob/main/run_training_pipeline.ipynb) ，其对应的colab： [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/run_training_pipeline.ipynb) ，运行完大概需要15分钟，我运行成功后的副本colab：[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RGkbev8D85gR33HJYxqNdnEThODvGUsS?usp=sharing)
- [训练参数说明wiki](https://github.com/shibing624/MedicalGPT/wiki/%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E)
- [数据集wiki](https://github.com/shibing624/MedicalGPT/wiki/%E6%95%B0%E6%8D%AE%E9%9B%86)
- [扩充词表wiki](https://github.com/shibing624/MedicalGPT/wiki/%E6%89%A9%E5%85%85%E4%B8%AD%E6%96%87%E8%AF%8D%E8%A1%A8)
- [FAQ](https://github.com/shibing624/MedicalGPT/wiki/FAQ)
#### Supported Models
The following models are tested:

bloom:
- [bigscience/bloomz-560m](https://huggingface.co/bigscience/bloomz-560m)
- [bigscience/bloomz-1b7](https://huggingface.co/bigscience/bloomz-1b7)
- [bigscience/bloomz-7b1](https://huggingface.co/bigscience/bloomz-7b1)

llama:
- [shibing624/chinese-alpaca-plus-7b-hf](https://huggingface.co/shibing624/chinese-alpaca-plus-7b-hf)
- [shibing624/chinese-alpaca-plus-13b-hf](https://huggingface.co/shibing624/chinese-alpaca-plus-13b-hf)
- [minlik/chinese-llama-plus-7b-merged](https://huggingface.co/minlik/chinese-llama-plus-7b-merged)
- [shibing624/chinese-llama-plus-13b-hf](https://huggingface.co/shibing624/chinese-llama-plus-13b-hf)
- [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)
- [IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)

llama2:
- [daryl149/llama-2-7b-chat-hf](https://huggingface.co/daryl149/llama-2-7b-chat-hf)
- [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

chatglm:
- [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
- [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)

baichuan:
- [baichuan-inc/baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B)
- [baichuan-inc/Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base)
- [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)

## 💻 Inference 
训练完成后，现在我们加载训练好的模型，验证模型生成文本的效果。

```shell
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_type base_model_type \
    --base_model path_to_model_hf_dir \
    --tokenizer_path path_to_model_hf_dir \
    --lora_model path_to_lora \
    --interactive
```

参数说明：

- `--model_type {base_model_type}`：预训练模型类型，如llama、bloom、chatglm等
- `--base_model {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录
- `--tokenizer_path {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录
- `--lora_model {lora_model}`：LoRA解压后文件所在目录，也可使用HF Model Hub模型调用名称。如果已经合并了LoRA权重到预训练模型，则可以不提供此参数
- `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与--base_model相同
- `--template_name`：模板名称，如`vicuna`、`alpaca`等。若不提供此参数，则其默认值是alpaca
- `--interactive`：以交互方式启动多轮问答，使用流式推理
- `--data_file {file_name}`：非交互方式启动下，按行读取file_name中的的内容进行预测
- `--predictions_file {file_name}`：非交互式方式下，将预测的结果以json格式写入file_name
- `--resize_emb`：是否调整embedding大小，若不调整，则使用预训练模型的embedding大小，默认不调整
- `--only_cpu`：仅使用CPU进行推理
- `--gpus {gpu_ids}`：指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如0,1,2


#### Inference Examples
[shibing624/vicuna-baichuan-13b-chat](https://huggingface.co/shibing624/vicuna-baichuan-13b-chat) inference examples:

<details><summary>Model output case</summary>

| Human                                                                              | Assitant                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 小孩发烧怎么办                                                                            | 发烧是身体对感染或其他疾病的自然反应。以下是一些可以帮助缓解发烧症状的建议：<br/>1. 保持身体水分充足：发烧时，身体会失去水分，因此需要保持水分充足。<br/>2. 使用冰袋：将冰袋放在额头或颈部可以帮助降低体温。<br/>3. 使用退烧药：退烧药可以帮助降低体温。<br/>4. 保持身体温度：保持身体温度可以帮助降低体温。<br/>5. 使用冷水或温水擦拭身体：使用冷水或温水擦拭身体可以帮助降低体温。<br/>6. 使用退烧药：退烧药可以帮助降低体温。<br/>7. 保持身体温度：保持身体温度可以帮助降低体温。<br/>8. 使用冷水或温水擦拭身体：使用冷水或温水擦拭身体可以帮助降低体温。<br/>9. 使用退烧药：退烧药可以帮助降低体温。<br/>10. 保持身体温度：保持身体温度可以帮助降低体温。                                                                                                                                                                                                                                                         |
| 一岁宝宝发烧能吃啥药？                                                                        | 1岁宝宝发烧时，可以考虑以下几种药物：<br/>1. 退烧药：如布洛芬(Ibuprofen)或对乙酰氨基酚(Paracetamol)。这些药物可以帮助降低体温，缓解发烧症状。<br/>2. 抗生素：如果发烧是由细菌感染引起的，可以考虑使用抗生素。但请务必在医生的建议下使用。<br/>3. 抗病毒药物：如果发烧是由病毒感染引起的，可以考虑使用抗病毒药物。但请务必在医生的建议下使用。<br/>4. 解热镇痛药：如阿司匹林(Aspirin)或对乙酰氨基酚(Paracetamol)。这些药物可以帮助缓解发烧症状。<br/>5. 抗组胺药：如苯海拉明(Diphenhydramine)或非索非那定(Naproxen)。这些药物可以帮助缓解发烧症状。<br/>请务必在医生的建议下使用这些药物，并遵循药物说明书上的用量和用法。同时，保持宝宝充足的水分摄入，并确保宝宝有足够的休息。                                                                                                                                                                                                                             |
| 肛门病变可能是什么疾病的症状                                                                     | 肛门病变可能由多种原因引起，包括感染、炎症、肿瘤或其他疾病。一些常见的肛门病变包括：<br/>1. 肛裂：这是一种肛门皮肤撕裂，通常由便秘或其他压力因素引起。<br/>2. 肛瘘：这是一种肛门感染，导致肛门皮肤和直肠之间形成瘘管。<br/>3. 肛门疣：这是一种肛门皮肤上形成的疣状病变，通常由人类乳头状瘤病毒(HPV)引起。<br/>4. 肛门脓肿：这是一种肛门感染，导致肛门皮肤下形成脓肿。<br/>5. 肛门肿瘤：这是一种肛门皮肤或直肠内形成的肿瘤。<br/>6. 肛门溃疡：这是一种肛门皮肤上形成的溃疡，通常由感染或炎症引起。<br/>7. 肛门瘙痒：这是一种肛门皮肤瘙痒的症状，可能由感染、炎症或其他疾病引起。<br/>如果您有肛门病变的症状，请咨询医疗保健专业人员以获得适当的诊断和治疗。                                                                                                                                                                                                                                                                  |
| 医生，您好白带增多泛黄怎么回事？                                                                   | 您好，白带增多和发黄可能是由多种原因引起的，包括感染、荷尔蒙变化、阴道干燥或其他妇科问题。以下是一些可能的原因：<br/>1. 感染：白带增多和发黄可能是由阴道或尿道感染引起的。感染可能由细菌、病毒或真菌引起。<br/>2. 荷尔蒙变化：荷尔蒙变化，如月经周期或更年期，可能会导致白带增多和发黄。<br/>3. 阴道干燥：阴道干燥可能会导致白带增多和发黄。<br/>4. 其他妇科问题：其他妇科问题，如子宫颈炎、子宫内膜炎或子宫内膜异位症，也可能导致白带增多和发黄。<br/>如果您担心自己的白带，建议您咨询妇科医生进行检查和诊断。医生可以根据您的症状和检查结果确定原因并提供适当的治疗。                                                                                                                                                                                                                                                                                                                          |
| 药流后有巧克力白带是没流干净吗                                                                    | 药流后出现巧克力白带并不一定意味着没有流干净。巧克力白带是由于子宫内膜脱落和出血引起的，通常在药流后的一段时间内会出现。<br/><br/>药流后出现巧克力白带的原因可能包括：<br/><br/>1. 子宫内膜脱落：药流后，子宫内膜会脱落，导致出血。这种出血可能表现为巧克力白带。<br/>2. 感染：药流后，子宫内膜和宫颈口可能存在创伤，容易受到细菌感染。感染可能导致巧克力白带。<br/>3. 激素水平变化：药流后，激素水平可能发生变化，导致巧克力白带。<br/><br/>如果巧克力白带持续时间较长，或者伴有其他症状，如腹痛、发热、分泌物异味等，建议及时就诊，进行相关检查，如妇科检查、B超等，以确定是否存在感染或其他问题。在医生的指导下进行相应的治疗。                                                                                                                                                                                                                                                                                      |
| 维胺酯维E乳膏能治理什么疾病                                                                     | 维胺酯维E乳膏是一种外用药物，主要用于治疗痤疮、脂溢性皮炎等皮肤病。它可以帮助减少皮肤油脂分泌，减轻炎症，促进皮肤修复。<br/><br/>需要注意的是，维胺酯维E乳膏并非万能药，对于某些皮肤病可能并不适用。在使用前，建议咨询医生或药师，了解药物的适应症、禁忌症、用法用量等信息。同时，使用药物时应遵循医嘱，避免自行滥用。                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 痤疮是什么原因导致的?                                                                        | 痤疮，也称为粉刺，是由皮肤毛孔堵塞引起的。堵塞毛孔的原因有很多，包括：<br/>1. 皮脂腺过度活跃：皮脂腺产生过多油脂，导致毛孔堵塞。<br/>2. 毛孔堵塞：皮肤表面的死皮细胞、油脂和细菌堵塞毛孔。<br/>3. 炎症：痤疮通常伴随着炎症，这可能是由于皮肤感染或免疫系统对痤疮的反应引起的。<br/>4. 遗传因素：某些人可能更容易患痤疮，这可能是由于遗传因素引起的。<br/>5. 荷尔蒙变化：痤疮在青春期和月经期间更为常见，这可能是由于荷尔蒙变化引起的。<br/>6. 生活方式因素：某些生活方式因素，如压力、不良饮食和缺乏运动，也可能导致痤疮。<br/>7. 化妆品和护肤品：某些化妆品和护肤品可能导致痤疮，特别是那些含有刺激性成分的产品。<br/>8. 药物：某些药物，如口服避孕药，也可能导致痤疮。<br/>要预防痤疮，可以采取一些措施，如保持皮肤清洁、使用温和的护肤品、避免过度使用化妆品和护肤品、保持健康的生活方式和饮食习惯、避免压力和荷尔蒙变化。如果痤疮严重，可能需要寻求皮肤科医生的帮助。                                                                                                                                                         |
| 脸上长青春痘怎么办,能吃啥药,需要注意啥                                                               | 青春痘，又称痤疮，是一种常见的皮肤病。青春痘的治疗需要综合考虑饮食、生活习惯、药物等多个方面。以下是一些建议：<br/><br/>1. 饮食：<br/>* 避免高糖、高脂、高油的食物，如快餐、甜点、油炸食品等。<br/>* 多吃新鲜蔬菜、水果、全谷类食物，以增加纤维摄入，促进肠道蠕动。<br/>* 避免摄入过多的咖啡因、酒精、辛辣食物等刺激性食物。<br/><br/>2. 生活习惯：<br/>* 保持良好的作息时间，保证充足的睡眠。<br/>* 避免长时间使用电脑、手机等电子产品，以免辐射对皮肤造成刺激。<br/>* 保持皮肤清洁，每天早晚使用温和的洁面产品清洁皮肤。<br/>* 避免挤压青春痘，以免感染和留下疤痕。<br/><br/>3. 药物：<br/>* 外用药物：可以使用含有水杨酸、苯甲酸、壬二酸等成分的护肤品，帮助去除皮肤表面的角质，减少痘痘的形成。<br/>* 口服药物：如果青春痘严重，可以考虑口服抗生素、维生素B族、锌等药物，但需在医生指导下使用。<br/><br/>4. 注意：<br/>* 不要随意使用化妆品、护肤品，以免刺激皮肤。<br/>* 不要过度清洁皮肤，以免破坏皮肤屏障。<br/>* 保持心情舒畅，避免焦虑、抑郁等负面情绪。<br/><br/>总之，青春痘的治疗需要综合考虑饮食、生活习惯、药物等多个方面。在治疗过程中，要保持耐心和信心，遵循医生的建议，逐步改善皮肤状况。 |
| 由于输精管堵塞会不会导致男性不育？，医生您好，我同事这几天不开心，我问她怎么了，她说她老公输精管堵塞，所以一直没孩子，不是不想要，请问输精管堵塞会不会导致男性不育？ | 输精管堵塞可能会导致男性不育。输精管是连接睾丸和尿道的一条管道，负责将精子从睾丸输送到尿道。如果输精管堵塞，精子就无法通过输精管到达尿道，从而导致不育。<br/><br/>输精管堵塞的原因有很多，包括先天性发育异常、炎症、肿瘤、外伤等。输精管堵塞的治疗方法包括手术疏通、药物治疗等。如果输精管堵塞严重，可能需要进行手术疏通。<br/><br/>需要注意的是，输精管堵塞并不一定导致不育，因为还有其他途径可以将精子输送到女性体内。例如，精子可以通过附睾、前列腺等组织到达女性体内。因此，如果输精管堵塞，建议及时就医，寻求专业医生的帮助。                                                                                                                                                                                                                                                                                                                                                     |
| 心肌梗塞能治疗吗？                                                                          | 心肌梗死是一种严重的疾病，如果及时治疗，可以挽救生命并防止长期残疾。治疗心肌梗死的关键是尽早识别和治疗。<br/><br/>心肌梗死的治疗通常包括以下步骤：<br/><br/>1. 立即就医：如果出现胸痛、呼吸急促或晕厥等症状，立即就医。<br/><br/>2. 药物治疗：医生可能会开处方药物，如硝酸甘油或阿司匹林，以帮助缓解症状并防止进一步的损伤。<br/><br/>3. 溶栓治疗：如果患者在到达医院前出现心肌梗死，医生可能会进行溶栓治疗，通过注射溶解血栓的药物来帮助恢复血流。<br/><br/>4. 手术治疗：如果药物治疗或溶栓治疗无效，医生可能会建议进行手术，如冠状动脉搭桥手术或心脏支架植入术。<br/><br/>5. 康复和预防：心肌梗死的治疗通常包括康复计划，以帮助患者恢复身体功能并预防未来的心脏事件。此外，患者还应遵循医生的建议，以降低患心脏病的风险。<br/><br/>需要注意的是，心肌梗死的治疗取决于患者的具体情况和症状。因此，如果您或您认识的人出现心肌梗死症状，请立即就医。                                                                                                                                                            |
| 怎么让自己精力充沛，列5点建议                                                                    | 1. 保持充足的睡眠：睡眠对于保持精力充沛至关重要。确保每晚获得7-9小时的高质量睡眠。<br/>2. 均衡饮食：均衡饮食对于保持能量水平至关重要。确保摄入足够的蛋白质、碳水化合物和健康脂肪。<br/>3. 定期锻炼：定期锻炼可以帮助提高能量水平。尝试每天进行至少30分钟的有氧运动，如散步、跑步或骑自行车。<br/>4. 管理压力：压力会消耗能量，因此管理压力对于保持精力充沛至关重要。尝试找到放松和减压的方法，如冥想、瑜伽或深呼吸练习。<br/>5. 保持水分：脱水会降低能量水平。确保每天喝足够的水，并根据需要补充水分。                                                                                                                                                                                                                                                                                                                                                         |
| 写个春天的诗                                                                             | 春天，<br/>是大地苏醒的季节，<br/>是万物复苏的季节，<br/>是生命绽放的季节。<br/>春天，<br/>是阳光明媚的季节，<br/>是鸟语花香的季节，<br/>是绿意盎然的季节，<br/>是希望无限的季节。<br/>春天，<br/>是人们欢笑的季节，<br/>是孩子们嬉戏的季节，<br/>是恋人们相拥的季节，<br/>是人们追求梦想的季节。                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 南极洲为啥没有北极熊?                                                                        | 南极洲没有北极熊，因为北极熊生活在北极地区，而南极洲位于南半球，气候和环境与北极地区有很大不同。南极洲的气候非常寒冷，常年被冰雪覆盖，而北极熊则生活在北极地区，那里有丰富的海洋生物和陆地哺乳动物。此外，南极洲的生态系统也与北极地区不同，没有北极熊所需的生存条件。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

</details>

## 📚 Dataset 
### 医疗数据集

- 240万条中文医疗数据集(包括预训练、指令微调和奖励数据集)：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)
- 22万条中文医疗对话数据集(华佗项目)：[FreedomIntelligence/HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1)

### 通用数据集

#### Pretraining datasets
- 16GB中英文无监督、平行语料[Linly-AI/Chinese-pretraining-dataset](https://huggingface.co/datasets/Linly-AI/Chinese-pretraining-dataset)
- 524MB中文维基百科语料[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
#### SFT datasets
- 10万条多语言ShareGPT GPT4多轮对话数据集：[shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) [本项目支持格式]
- 9万条英文ShareGPT多轮对话数集：[anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) [本项目支持格式]
- 50万条中文ChatGPT指令Belle数据集：[BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
- 100万条中文ChatGPT指令Belle数据集：[BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
- 5万条英文ChatGPT指令Alpaca数据集：[50k English Stanford Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca#data-release)
- 2万条中文ChatGPT指令Alpaca数据集：[shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)
- 69万条中文指令Guanaco数据集(Belle50万条+Guanaco19万条)：[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)
- 5万条英文ChatGPT多轮对话数据集：[RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)
- 80万条中文ChatGPT多轮对话数据集：[BelleGroup/multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
- 116万条中文ChatGPT多轮对话数据集：[fnlp/moss-002-sft-data](https://huggingface.co/datasets/fnlp/moss-002-sft-data)
- 3.8万条中文ShareGPT多轮对话数据集：[FreedomIntelligence/ShareGPT-CN](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-CN)

#### Reward Model datasets
- 原版的oasst1数据集：[OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- 2万条多语言oasst1的reward数据集：[tasksource/oasst1_pairwise_rlhf_reward](https://huggingface.co/datasets/tasksource/oasst1_pairwise_rlhf_reward)
- 11万条英文hh-rlhf的reward数据集：[Dahoas/full-hh-rlhf](https://huggingface.co/datasets/Dahoas/full-hh-rlhf)
- 9万条英文reward数据集(来自Anthropic's Helpful Harmless dataset)：[Dahoas/static-hh](https://huggingface.co/datasets/Dahoas/static-hh)
- 7万条英文reward数据集（来源同上）：[Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static)
- 7万条繁体中文的reward数据集（翻译自rm-static）[liswei/rm-static-m2m100-zh](https://huggingface.co/datasets/liswei/rm-static-m2m100-zh)
- 7万条英文Reward数据集：[yitingxie/rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets)
- 3千条中文知乎问答偏好数据集：[liyucheng/zhihu_rlhf_3k](https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k)

## ✅ Todo

1. [x] add multi-round dialogue data fine-tuning method
2. [x] add reward model fine-tuning
3. [x] add rl fine-tuning
4. [x] add medical reward dataset
5. [x] add llama in8/int4 training
6. [x] add all training and predict demo in colab

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
