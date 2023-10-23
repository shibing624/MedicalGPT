# Training Detail


### Stage 1: PT(Continue PreTraining)
第一阶段：PT(Continue PreTraining)增量预训练

使用百科类文档类数据集，用来在领域数据集上增量预训练或二次预训练，期望能把领域知识注入给模型，以医疗领域为例，希望增量预训练，能让模型理解感冒的症状、病因、治疗药品、治疗方法、药品疗效等知识，便于后续的SFT监督微调能激活这些内在知识。

这里说明一点，像GPT3、LLaMA这样的大模型理论上是可以从增量预训练中获益，但增量预训练需要满足两个要求：1）高质量的预训练样本；2）较大的计算资源，显存要求高，即使是用LoRA技术，也要满足block_size=1024或2048长度的文本加载到显存中。

其次，如果你的项目用到的数据是模型预训练中已经使用了的，如维基百科、ArXiv等LLaMA模型预训练用了的，则这些数据是没有必要再喂给LLaMA增量预训练，而且预训练样本的质量如果不够高，也可能会损害原模型的生成能力。

tips：PT阶段是可选项，请慎重处理。

基于llama-7b模型，使用医疗百科类数据继续预训练，期望注入医疗知识到预训练模型，得到llama-7b-pt模型

Continue pretraining of the base llama-7b model to create llama-7b-pt:

```shell
cd scripts
sh run_pt.sh
```

[训练参数说明](https://github.com/shibing624/MedicalGPT/blob/main/docs/training_params.md)
- 如果你的显存不足，可以改小batch_size=1, block_size=512（影响训练的上下文最大长度）;
- 如果你的显存更大，可以改大block_size=2048, 此为llama原始预训练长度，不能更大啦；调大batch_size。

### Stage 2: SFT(Supervised Fine-tuning)
第二阶段：SFT(Supervised Fine-tuning)有监督微调

基于llama-7b-pt模型，使用医疗问答类数据进行有监督微调，得到llama-7b-sft模型

Supervised fine-tuning of the base llama-7b-pt model to create llama-7b-sft

```shell
cd scripts
sh run_sft.sh
```

[训练参数说明](https://github.com/shibing624/MedicalGPT/blob/main/docs/training_params.md)

### Stage 3: RLHF(Reinforcement Learning from Human Feedback)
#### Reward Modeling
RM(Reward Model)奖励模型建模

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
[训练参数说明](https://github.com/shibing624/MedicalGPT/blob/main/docs/training_params.md)

#### Reinforcement Learning
RL(Reinforcement Learning)强化学习

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

### Stage 3: DPO(Direct Preference Optimization)
DPO(Direct Preference Optimization)直接偏好优化

DPO方法可以通过直接优化语言模型来实现对其行为的精确控制，而无需使用复杂的强化学习。

DPO 将奖励函数和最优策略之间的映射联系起来，从而把约束奖励最大化问题转化为一个单阶段的策略训练问题。
这种算法不仅不用拟合奖励模型，还避免了在微调过程中从语言模型中采样或调整超参数的需要。

实验结果表明，DPO 算法可以与现有RLHF方法一样有效地从人类偏好中学习，甚至在某些任务中表现更好，比如情感调节、摘要和单轮对话。

PS: 使用DPO训练LLaMA2-7B在fp16，batch_size为2时，需要70GB显存。

```shell
sh run_dpo.sh
```