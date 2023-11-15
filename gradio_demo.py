# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

pip install gradio>=3.50.2
"""
import argparse
import os
from threading import Thread

import gradio as gr
import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)

from supervised_finetuning import get_conv_template

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="vicuna", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan2, chatglm2 etc.")
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--share', action='store_true', help='Share gradio')
    parser.add_argument('--port', default=8081, type=int, help='Port of gradio demo')
    args = parser.parse_args()
    print(args)

    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    base_model = model_class.from_pretrained(
        args.base_model,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True,
    )
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("loaded lora model")
    else:
        model = base_model
    if device == torch.device('cpu'):
        model.float()
    model.eval()
    prompt_template = get_conv_template(args.template_name)
    stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str

    def predict(message, history):
        """Generate answer from prompt with GPT and stream the output"""
        history_messages = history + [[message, ""]]
        prompt = prompt_template.get_prompt(messages=history_messages)
        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        input_ids = tokenizer(prompt).input_ids
        context_len = 2048
        max_new_tokens = 512
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        generation_kwargs = dict(
            input_ids=torch.as_tensor([input_ids]).to(device),
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        partial_message = ""
        for new_token in streamer:
            if new_token != stop_str:
                partial_message += new_token
                yield partial_message

    gr.ChatInterface(
        predict,
        chatbot=gr.Chatbot(),
        textbox=gr.Textbox(placeholder="Ask me question", lines=4, scale=9),
        title="MedicalGPT",
        description="为了促进医疗行业大模型的开放研究，本项目开源了[MedicalGPT](https://github.com/shibing624/MedicalGPT)医疗大模型",
        theme="soft",
    ).queue().launch(share=args.share, inbrowser=True, server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    main()
