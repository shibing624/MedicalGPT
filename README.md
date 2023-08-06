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


| Model                                                                                                     | Base Model                                                                        | Introduction                                                                                                                           | 
|:----------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| [shibing624/ziya-llama-13b-medical-lora](https://huggingface.co/shibing624/ziya-llama-13b-medical-lora)   | [IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1) | 在240万条中英文医疗数据集[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)上SFT微调了一版Ziya-LLaMA-13B模型，医疗问答效果有提升，发布微调后的LoRA权重 |
| [shibing624/ziya-llama-13b-medical-merged](https://huggingface.co/shibing624/ziya-llama-13b-medical-merged) | [IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1) | 在240万条中英文医疗数据集[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)上SFT微调了一版Ziya-LLaMA-13B模型，医疗问答效果有提升，发布微调后的完整模型权重 |


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
[shibing624/ziya-llama-13b-medical-merged](https://huggingface.co/shibing624/ziya-llama-13b-medical-merged) inference examples:

<details><summary>Model output case</summary>

| Human                                                                              | Assitant                                                                                                                                                                                                                                                                                                                                                                                            |
|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 小孩发烧怎么办                                                                            | 宝宝的抵抗力是比较差的，所以很容易就会出现感冒或者其他疾病。如果宝宝在平时没有注意保暖，那么也可能导致身体受凉而引起了感冒等症状，这样不仅对于健康造成影响，还可能让宝宝出现高热的情况。因此当家长们发现自己的宝宝出现了高热的问题之后一定要及早进行治疗和护理才好。<br/>1、首先我们应该给宝宝服用退烧药物来帮助降温，但是千万不要盲目使用退烧药物，否则只会加重宝宝的病情，最好带着宝宝到医院去检查清楚具体原因之后再针对性地采取措施处理。<br/>2、另外大人们也需要做好预防工作，尤其是天气变化的时候更应该多关心下宝宝的身体状态，避免宝宝受寒受凉从而诱发各种疾病，同时也要尽量减少宝宝接触细菌病毒的机会，增强宝宝的免疫力。<br/>3、除了上面两点之外，大人们还可以通过按摩推拿穴位的方式来缓解宝宝的病痛，促进血液循环，提升新陈代谢速度，达到快速消炎止疼的效果         |
| 一岁宝宝发烧能吃啥药？                                                                        | 孩子的身体是很脆弱的，如果不注意保护好自己就会出现各种疾病。有些家长在看到孩子生病后都非常着急，想要立刻给他们用药治疗，但其实并没有那么简单，因为每个人对于药物的敏感度和耐受性都是不同的，所以我们需要根据具体情况来选择合适的方法进行治疗才可以达到最佳效果。<br/>1、首先大多数小儿退热药中含有解热镇痛成分阿司匹林或布洛芬等，这类药品虽然副作用较少，但也存在过量使用引起胃肠道损伤甚至消化道溃疡出血的风险；<br/>2、还有部分小儿退热药中含有扑尔敏等抗组胺药，它们本身属于处方药，应该严格按照医嘱服用，切勿擅自加减剂量或者延长用药时间;<br/>3、此外，某些小儿退热药中含有维生素C及葡萄糖酸钙等营养补充剂，只有当患儿伴随脱水症状（如口渴、尿色深）时才考虑添加，否则容易造成电解质紊乱。                                                    |
| 肛门病变可能是什么疾病的症状                                                                     | 痔疮、直肠息肉等。                                                                                                                                                                                                                                                                                                                                                                                           |
| 医生，您好白带增多泛黄怎么回事？                                                                   | 你这种情况考虑可能会是阴道炎症引起的建议平时留意卫生清洁，勤换洗衣物内裤，必要时去正规公立医院妇科复诊做分泌物常规等相关检查明确病因后对症治疗，不同原因所致的疾病在临床表现上也存在差异性，如果患者出现了类似于的症状应当及早到专业、权威的医院进行详细的检验和救治                                                                                                                                                                                                                                                                  |
| 药流后有巧克力白带是没流干净吗                                                                    | 在日常生活中，我们要或多或少了解一些常见疾病的相关知识，不仅能起到增加知识水平的作用，更能让自己更好地预防此类疾病的产生。很多女性都会出现月经紊乱、腹痛等症状，这时候就需要考虑妇科炎症可能，比如说宫颈糜烂和盆腔积液。那么下面小编来给大家介绍一种情况：药物流产后白带呈巧克力色怎么回事？<br/>1、首先，对于药物流产之后发现阴道分泌物异常，应该及时去医院做检查确诊具体原因，然后再根据结果进行治疗调理。<br/>2、其次，由于每个人身体素质以及患病程度存在差别，所以采取的措施也不尽相同，但无论选择哪种方式处理，都必须严格按照规定剂量服用，并且注意观察副作用反映情况，若出现明显不适感觉，则应立即停止使用。<br/>3、最后，建议你保持外阴清洁卫生，勤换内裤，避免吃辛辣刺激食品，而且还要禁忌房事，否则容易引起交叉感染影响恢复健康。                                |
| 维胺酯维E乳膏能治理什么疾病                                                                     | 皮肤淀粉样变性；脂溢性角化症                                                                                                                                                                                                                                                                                                                                                                                      |
| 痤疮是什么原因导致的?                                                                        | 痤疮，又称青春痘、粉刺。它主要发生在面部和胸背部位，表现为黑头粉刺、白头粉刺、丘疹、脓疱、结节等多种形式。其中以黑头粉刺最常见，约占全身皮肤病患者的70%左右。那么，痤疮是什么原因引起的呢？下面就给大家介绍一些相关内容。<br/>1、雄性激素分泌过旺：男女都会长痘痘，但是女孩子比较喜欢用化妆品来掩盖脸上的问题，而且很多人不知道这样做反而加重了自己的毛囊炎症状，所以说平时应该少使用含有雌激素类的产品。<br/>2、饮食习惯差：经常吃辛辣油腻的东西或者甜食也可能造成体内毒素堆积，从而诱发痤疮。<br/>3、压力太大：如果你总是处于紧张焦虑的情绪之中，同时还没有及时调整好心态，那么很容易出现内分泌失调的情况，进而影响到肌肤健康。<br/>4、睡眠质量差：熬夜对我们的身体伤害非常严重，尤其是对于年轻人群更甚，他们需要保证充足的睡眠才行，否则很容易让肌肤变得暗沉无光泽。                  |
| 脸上长青春痘怎么办,能吃啥药,需要注意啥                                                               | 你好，根据您的描述考虑是痤疮。建议外用维胺酯乳膏或者阿达帕林凝胶进行治疗看看，平时禁忌辛辣和刺激性食物，如海鲜、狗肉、羊肉等“发”动品，以及葱蒜韭姜等刺激性调料。多喝水，少吃油腻高热量食物。                                                                                                                                                                                                                                                                                                     |
| 由于输精管堵塞会不会导致男性不育？，医生您好，我同事这几天不开心，我问她怎么了，她说她老公输精管堵塞，所以一直没孩子，不是不想要，请问输精管堵塞会不会导致男性不育？ | 你好！根据你的叙述及症状推测：如果有输精管阻塞或者狭窄等情况都可能会引来无法排出来精液而影响怀上。建议到正规专业的快孕医院实施进行咨询和治疗为宜。对于输精管患者来说，及时救治是可以减低病发，如果不及时救治会影响到受孕情况。因此患者要及时药物救治，并且日常生活中要多看重自身护理                                                                                                                                                                                                                                                          |
| 心肌梗塞能治疗吗？                                                                          | 心脏病是一种危害性极大的疾病，我们要对它有足够多的了解。在生活中很多人都会患上这个疾病，而且还不知道怎么去预防和护理，所以就导致自己的身体越来越差，那么下面小编给你介绍几点关于心脏病的常识吧！<br/>1、冠状动脉粥样硬化性心脏病：又称为冠心病或缺血型心脏病（IHD）,是世界各地最重要的死亡原因之一。<br/>2、风湿热：主要侵犯心脏瓣膜，可引起二尖瓣狭窄及其他类型的心脏瓣膜病变；<br/>3、先天性心脏病：包括房间隔缺损、室间隔缺损等；<br/>4、高脂蛋白血症：由于胆固醇与低密度脂蛋白升高，使得动脉壁增厚，管腔缩小，易形成斑块并脱落造成急性栓塞；<br/>5、肥胖：过量摄入食物，特别是含糖饮料，加速脂肪堆积，促进动脉粥样硬化发展；<br/>6、吸烟：尼古丁刺激交感神经兴奋，使外周阻力增强，收缩压升高，舒张期减短，从而降低左右心室射血分数，诱发心律失常；<br/>7、酗酒：长期嗜酒者，肝功受到影响，代谢紊乱，容易出现脂质代谢异常。 |
| 怎么让自己精力充沛，列5点建议                                                                    | 1、保证睡眠质量：每天晚上7-8小时的高质量睡眠是提升身体和大脑活跃度最有效的方法。<br/>2、合理饮食：均衡营养摄入对于维持人体正常运转至关重要，多吃蔬菜水果，少油腻辛辣刺激性食物。<br/>3、适当锻炼：坚持一定强度的锻炼可以促进血液循环，增加肌肉耐力，改善心肺功能等。<br/>4、学习新知识：不断接触新事物，开拓视野，思考问题，都会使得我们更具创造力与灵感。<br/>5、放松心情：通过冥想或者瑜伽来缓解压力，也可以听音乐或看电影来调节情绪。                                                                                                                                                                 |
| 写个春天的诗                                                                             | 春风吹拂，花儿绽放。<br/>阳光明媚，鸟语鸣叫。<br/>树木抽芽，草地变绿。<br/>万物复苏，生机勃发。                                                                                                                                                                                                                                                                                                                                                   |
| 南极洲为啥没有北极熊?                                                                        |南极是世界上最大的冰盖，覆盖着98％以上的地球表面积。它位于南纬60度至74度之间，环绕在海洋中心周围。由于其独特的气候条件和生态系统，南极被称为“真正的荒野”。南极洲拥有丰富多样的动植物资源，但这里并不存在北极熊。 <br/>那么，为什么南极洲没有北极熊呢？<br/>首先，我们需要了解一下两者栖息地的差异。北极熊主要分布在北半球寒冷的高纬度地区，如加拿大、格陵兰岛等地。而南极洲则处于南半球，属于温带或寒带气候类型。虽然南极洲也有很低的平均气温（-25℃左右），但与北极相比还是太热了。因此，即使北极熊能够适应更严酷的气候条件，它们也无法在南极找到合适的栖息地。<br/>另外，南极洲缺乏陆地哺乳动物食物来源，包括鱼类、鲸鱼和企鹅等。尽管南极洲的水域中也有各种鱼类，但数量远少于北极圈内。<br/>同时，南极洲的土著居民——企鹅群体繁殖季节期间会消耗掉大部分可用的食物资源，导致当地的鱼类数量减少甚至枯竭。|

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
- 6千条多语言ShareGPT GPT4多轮对话数据集：[shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) [本项目支持格式]
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
