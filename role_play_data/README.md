
## 造训练数据

### 数据生成框架
本数据集使用LLM API接口生成，支持多种LLM提供商（OpenAI、豆包、[MiniMax](https://platform.minimaxi.com/)等），流程：

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
LLM选择：支持gpt-4o、豆包doubao-character-pro-32k、MiniMax-M2.7等多种LLM生成对话
```bash
# 使用OpenAI GPT-4o
python roleplay_data_generate_gpt4.py

# 使用豆包
python roleplay_data_generate_doubao.py

# 使用MiniMax（需要设置 MINIMAX_API_KEY 环境变量）
export MINIMAX_API_KEY="your_api_key"
python roleplay_data_generate_minimax.py

# MiniMax支持自定义参数
python roleplay_data_generate_minimax.py --model MiniMax-M2.5-highspeed --total 500 --rounds 8
```

### 多Provider支持

`llm_client.py` 提供了统一的LLM客户端接口，支持通过环境变量自动检测或手动指定Provider：

| Provider | 环境变量 | 默认模型 | API地址 |
|----------|---------|---------|---------|
| OpenAI | `OPENAI_API_KEY` | gpt-4o | https://api.openai.com/v1 |
| 豆包 | `DOUBAO_API_KEY` | doubao-character-pro-32k | https://ark.cn-beijing.volces.com/api/v3 |
| MiniMax | `MINIMAX_API_KEY` | MiniMax-M2.7 | https://api.minimax.io/v1 |

```python
from llm_client import create_llm_client

# 自动检测（根据环境变量）
client, model = create_llm_client()

# 指定Provider
client, model = create_llm_client(provider="minimax")
```
