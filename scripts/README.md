# Training Detail


### Stage 1: Continue Pretraining
第一阶段：PT(Continue PreTraining)增量预训练

基于llama-7b模型，使用医疗百科类数据继续预训练，期望注入医疗知识到预训练模型，得到llama-7b-pt模型，此步骤可选

Continue pretraining of the base llama-7b model to create llama-7b-pt:

```shell
cd scripts
sh run_pt.sh
```

[训练参数说明wiki](https://github.com/shibing624/MedicalGPT/wiki/%E8%AE%AD%E7%BB%83%E7%BB%86%E8%8A%82%E8%AF%B4%E6%98%8E)

### Stage 2: Supervised FineTuning
第二阶段：SFT(Supervised Fine-tuning)有监督微调

基于llama-7b-pt模型，使用医疗问答类数据进行有监督微调，得到llama-7b-sft模型

Supervised fine-tuning of the base llama-7b-pt model to create llama-7b-sft

```shell
cd scripts
sh run_sft.sh
```

[训练参数说明wiki](https://github.com/shibing624/MedicalGPT/wiki/%E8%AE%AD%E7%BB%83%E7%BB%86%E8%8A%82%E8%AF%B4%E6%98%8E)

### Stage 3: Reward Modeling
第三阶段：RM(Reward Model)奖励模型建模

RM(Reward Model)奖励模型，原则上，我们可以直接用人类标注来对模型做 RLHF 微调。

然而，这将需要我们给人类发送一些样本，在每轮优化后计分。这是贵且慢的，因为收敛需要的训练样本量大，而人类阅读和标注的速度有限。
一个比直接反馈更好的策略是，在进入 RL 循环之前用人类标注集来训练一个奖励模型RM。奖励模型的目的是模拟人类对文本的打分。

构建奖励模型的最佳实践是预测结果的排序，即对每个 prompt (输入文本) 对应的两个结果 (yk, yj)，模型预测人类标注的比分哪个更高。
RM模型是通过人工标注SFT模型的打分结果来训练的，目的是取代人工打分，本质是个回归模型，用来对齐人类偏好，主要是"HHH"原则，具体是"helpful, honest, harmless"。


基于llama-7b-sft模型，使用医疗问答偏好数据训练奖励偏好模型，训练得到llama-7b-reward模型

Reward modeling using dialog pairs from the reward dataset using the llama-7b-sft to create llama-7b-reward:

```shell
cd scripts
sh run_rm.sh
```
[训练参数说明wiki](https://github.com/shibing624/MedicalGPT/wiki/%E8%AE%AD%E7%BB%83%E7%BB%86%E8%8A%82%E8%AF%B4%E6%98%8E)

### Stage 4: Reinforcement Learning
第四阶段：RL(Reinforcement Learning)基于人类反馈的强化学习(RLHF)

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
pip install git+https://github.com/lvwerra/trl
cd scripts
sh run_rl.sh
```

