<div align="center">
  <a href="https://github.com/shibing624/MedicalGPT">
    <img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/logo.png" width="100" alt="Logo">
  </a>
</div>

-----------------

# MedicalGPT: Training Your Own Medical GPT Model with ChatGPT Training Pipeline
[![PyPI version](https://badge.fury.io/py/MedicalGPT.svg)](https://badge.fury.io/py/MedicalGPT)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/MedicalGPT.svg)](https://github.com/shibing624/MedicalGPT/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

## 📖 Introduction

**MedicalGPT** training medical GPT model with ChatGPT training pipeline, implemantation of Pretraining, 
Supervised Finetuning, Reward Modeling and Reinforcement Learning.

<img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/GPT_Training.jpg" width="600" />


## 🚀 Getting Started
#### Stage 1: Pretraining

#### Stage 2: Supervised FineTuning


#### Stage 3: Reward Modeling

#### Stage 4: Reinforcement Learning


## 📚 Dataset 

- 240万条中文医疗数据集(包括预训练指令微调和奖励数据集)：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)

## ✅ Todo

1. [ ] 新增多轮对话数据微调方法
2. [ ] add reward model finetuning
3. [ ] add rl finetuning
4. [ ] add medical reward dataset
5. [ ] add llama in4 training
6. [ ] add all training and predict demo in colab

## ☎️ Contact

- Issue(建议)
  ：[![GitHub issues](https://img.shields.io/github/issues/shibing624/MedicalGPT.svg)](https://github.com/shibing624/MedicalGPT/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我： 加我*微信号：xuming624, 备注：姓名-公司名-NLP* 进NLP交流群。

<img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/wechat.jpeg" width="200" />

## 😇 Citation

如果你在研究中使用了textgen，请按如下格式引用：

```latex
@misc{MedicalGPT,
  title={MedicalGPT: Text Generation Tool},
  author={Xu Ming},
  year={2021},
  howpublished={\url{https://github.com/shibing624/MedicalGPT}},
}
```

## 🤗 License

授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加textgen的链接和授权协议。

## 😍 Contribute

项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

- 在`tests`添加相应的单元测试
- 使用`python -m pytest`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

## 💕 Acknowledgements 


- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/finetune.py)

Thanks for their great work!
