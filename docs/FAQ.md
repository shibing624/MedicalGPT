#### 问：报错"NotImplementedError: Cannot copy out of meta tensor; no data!"
答：单卡显存不足，device_map='auto'在gpu占满下，会自动利用cpu加载模型，导致`_move_model_to_device`错误。
解决方法：指定多卡训练，参考`CUDA_VISIBLE_DEVICES=0,1,2,3 python supervised_finetuning.py ...`，把batch size调大，显存打满，跟数据并行一样能最大化利用显卡加速训练。参考[issues 4](https://github.com/shibing624/MedicalGPT/issues/4)


#### 问：chatglm，baichuan模型用LoRA（peft）训练，合并时报错
答：chatglm，baichuan模型的代码跟权重文件放一起了，代码没有合入transformers官方库，merge lora时，需要把原始权重路径下的python文件全部拷贝到merged文件夹下使用，参考[issue 68](https://github.com/shibing624/MedicalGPT/issues/68)

#### 问：chatglm，baichuan无法做RM和RL训练？
答：chatglm不是标准CausalLM，RM阶段需要AutoModelForSequenceClassification，chatglm没有实现；PPO训练需要AutoModelForCausalLMWithValueHead，chatglm也不支持，同样的原因百川模型也无法做RM和RL训练。官方transformers兼容chatglm和baichuan模型后才支持。参考[issue 107](https://github.com/shibing624/MedicalGPT/issues/107)