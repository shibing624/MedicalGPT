
## 造训练数据

### 数据生成框架
本数据集使用OpenAI API接口生成，流程：

- **种子特征集和基础设定**：
   - 手工编写的种子集包含基本角色特征。
   - LLM从这个种子集生成角色的基础设定。
- **角色设定的进化**：
  - 第二个种子集包含指导角色设定进化的指令Prompt。
  - 这些进化角色的指令Prompt被放到一个指令池中。基于这些进化Prompt，LLM对基础设定实施进化。
- **反馈循环**：
  - 由人类评估者和GPT-4组成的混合评价系统。此系统对进化后的设定给出反馈。
  - 反馈用于迭代更新种子集。如此迭代，我们最终得到一个细致的角色设定数据集。
- **角色扮演和对话生成**：
  - 使用self-instruction框架基于角色设定生成角色的对话数据。


1. 生成角色设定，分别生成护士角色和患者角色
```bash
cd role_play_data

python role_generate.py
```


2. 生成医患之间的多轮对话
LLM选择：分别用gpt-4o的api和豆包的doubao-character-pro-32k的api生成对话
```bash
python roleplay_data_generate_gpt4.py

python roleplay_data_generate_doubao.py
```

