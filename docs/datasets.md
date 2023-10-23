# 数据集


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

## 预训练数据集详情

第一阶段预训练，16GB中英文无监督、平行语料，数据集由Linly-AI已经上传到Huggingface datasets：https://huggingface.co/datasets/Linly-AI/Chinese-pretraining-dataset

数据包括：[CLUECorpusSmall、中英文翻译数据、News Commentary v13](https://github.com/dbiir/UER-py/wiki/%E9%A2%84%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE) 和[中文科学文献数据 CSL](https://github.com/ydli-ai/CSL)

#### 中文数据集
数据集 | Disk Size | Link
-- | -- | --
ClueCorpusSmall | 13G | https://github.com/CLUEbenchmark/CLUECorpus2020
中文维基百科 2023 | 2.5G | https://download.wikipedia.com/zhwiki/
CSL | 1.5G | https://github.com/ydli-ai/CSL
news-crawl | 2.3G | https://data.statmt.org/news-crawl/zh/

#### 平行语料


数据集 | Disk Size | Link
-- | -- | --
UNCorpus | 4.3G | https://conferences.unite.un.org/UNCorpus
translation2019zh | 1.3G | https://github.com/brightmart/nlp_chinese_corpus
WikiMatri | 0.6G | http://data.statmt.org/wmt21/translation-task/WikiMatrix/
news-commentry | 67M | http://data.statmt.org/wmt20/translation-task/back-translation/
ParaCrawl v9 | 2.6G | https://paracrawl.eu/

#### 英文数据集


数据集 | Disk Size | Link
-- | -- | --
英文维基百科 2023 | 20G | https://download.wikipedia.com/enwiki/
arxiv | 10G | https://github.com/togethercomputer/RedPajama-Data
GitHub | 10G | 同上
Book | 18G | 同上
stackexchange | 13G | 同上

### 格式
下载语料后，合并到一个 .txt 文件并按行随机打乱，语料格式如下：
```
doc1
doc2
doc3
```
#### jsonl 预训练语料
为了支持代码等包含换行符的数据，预训练数据也可以整理成jsonl格式，格式如下：
```
{"text": "doc1"}
{"text": "doc2"}
{"text": "doc3"}
```


# 数据集格式
#### Pre-training dataset format

Dataset formats used `--train_file_dir` and `--validation_file_dir`

The format of the PT (pre-training) data set is as follows:

text file, one sample per line
```
txt file
```

#### SFT dataset format
The format of the SFT (supervised fine-tuning) dataset is as follows

vicuna dataset format, one sample per line, each sample contains the following fields:

json file, one sample per line, each sample contains the following fields:
```
head -n 1 sharegpt_gpt4.jsonl

{"conversations":[
  {'from': 'human',
   'value': '採用優雅現代中文，用中文繁體字型，回答以下問題。為所有標題或專用字詞提供對應的英語翻譯：Using scholarly style, summarize in detail James Barr\'s book "Semantics of Biblical Language". Provide examples supplied in the book and give corresponding page numbers wherever applicable.'},
  {'from': 'gpt',
   'value': '《聖經語言的語義學》（Semantics of Biblical Language）是詹姆斯·巴爾（James Barr）於1961年出版的一本研究聖經語言的專著。在這部作品中，巴爾對聖經希伯來語和希臘語的語言學特徵進行了深入的探討，他批判了當時盛行的聖經研究方法，主張應該將語言學原理應用於聖經文本的詮釋，以便更準確地理解聖經的意思。\n\n巴爾在《聖經語言的語義學》中提出了幾個重要的觀點。首先，他主張語言學和語言學家在聖經研究中的角色應該受到重視。他認為，應該避免將聖經語言學和其他古典語言學區分開來，因為聖經語言和其他古代語言一樣，都有其固有的語法、語義和詞彙特點（第18-23頁）。\n\n其次，巴爾批評了當時流行的"字根意義論"（root-meaning fallacy），即過分強調單詞的字根在詮釋聖經含義時的作用。他指出，詞義是由語境決定的，而非僅僅由字根決定。巴爾舉了希伯來語單詞 "כפר"（kāpar）為例，它在不同的語境下可以表示"遮蓋"、"贖回"或"和解"，而不是僅僅依賴其字根意義（第183-184頁）。\n\n第三，巴爾強調了詞彙意義在詮釋聖經時的重要性。他認為，了解單詞的意義和用法對於恰當理解聖經文本至關重要。巴爾引用了希伯來語單詞 "ברית"（berîṯ，英文：covenant）和希臘語單詞 "διαθήκη"（diathēkē，英文：testament）為例，解釋了這兩個詞在聖經中的不'},
  {'from': 'human', 'value': '繼續'},
  {'from': 'gpt',
   'value': '同含義和用法。巴爾認為，了解這些詞語在不同語境下的意義對於理解聖經中的盟約和遺囑概念至關重要（第50-53頁）。\n\n第四，巴爾強調語境在詮釋聖經時的重要性。他認為，要理解聖經的含義，就必須考慮到文本的歷史背景、文化環境和語言風格。此外，巴爾提倡從語言學的角度研究聖經，以便更好地理解聖經的結構和風格。他以希伯來詩篇的平行主義（Parallelism）為例，闡述了語言學原理如何有助於揭示詩篇中的意象和修辭結構（第250-252頁）。\n\n總之，《聖經語言的語義學》一書對聖經研究領域產生了深遠的影響。詹姆斯·巴爾通過對聖經語言學的深入研究，提出了許多富有洞察力的觀點，尤其是關於語言學在聖經詮釋中的重要性。這些觀點對後來的聖經研究者產生了很大的啟示，使他們在研究聖經時更加重視語言學原理，從而更加準確地理解聖經的含義。'}
  ]
}
```

#### Reward model dataset format
The format of the Reward (reward model) data set is as follows:
json file, one sample per line, each sample contains the following fields:
```
{"question": "text1", "response_chosen": "text2", "response_rejected": "text3"}
```


#### Reinforcement Learning dataset format
The RL (Reinforcement Learning) dataset format is as follows:
json file, one sample per line, each sample contains the following fields:
```
{"instruction": "text1", "input": "text2", "output": "text3"}
```

SFT datasets can be reused.


Use `--dataset_name` to load HF datasets, format refer to [shibing624/medical](https://huggingface.co/datasets/shibing624/medical)
