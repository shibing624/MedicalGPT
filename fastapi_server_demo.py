# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: api start demo

usage:
CUDA_VISIBLE_DEVICES=0 python fastapi_server_demo.py --base_model bigscience/bloom-560m

curl -X 'POST' 'http://0.0.0.0:8008/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "咏鹅--骆宾王；登黄鹤楼--"
}'
"""

import argparse
import os
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI
from loguru import logger
from peft import PeftModel
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig,
)

from template import get_conv_template


@torch.inference_mode()
def stream_generate_answer(
        model,
        tokenizer,
        prompt,
        device,
        do_print=True,
        max_new_tokens=512,
        repetition_penalty=1.0,
        context_len=2048,
        stop_str="</s>",
):
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0]
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    attention_mask = attention_mask[-max_src_len:]
    generation_kwargs = dict(
        input_ids=input_ids.unsqueeze(0).to(device),
        attention_mask=attention_mask.unsqueeze(0).to(device),
        max_new_tokens=max_new_tokens,
        num_beams=1,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        stop = False
        pos = new_text.find(stop_str)
        if pos != -1:
            new_text = new_text[:pos]
            stop = True
        generated_text += new_text
        if do_print:
            print(new_text, end="", flush=True)
        if stop:
            break
    if do_print:
        print()
    return generated_text


class Item(BaseModel):
    input: str = Field(..., max_length=2048)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="vicuna", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan, chatglm2 etc.")
    parser.add_argument('--system_prompt', default="", type=str)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--port', default=8008, type=int)
    args = parser.parse_args()
    print(args)

    def load_model(args):
        if args.only_cpu is True:
            args.gpus = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        load_type = 'auto'
        if torch.cuda.is_available():
            device = torch.device(0)
        else:
            device = torch.device('cpu')
        if args.tokenizer_path is None:
            args.tokenizer_path = args.base_model

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
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
            print("Loaded lora model")
        else:
            model = base_model
        if device == torch.device('cpu'):
            model.float()
        model.eval()
        print(tokenizer)
        return model, tokenizer, device

    # define the app
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    model, tokenizer, device = load_model(args)
    prompt_template = get_conv_template(args.template_name)
    stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str

    def predict(sentence):
        history = [[sentence, '']]
        prompt = prompt_template.get_prompt(messages=history, system_prompt=args.system_prompt)
        response = stream_generate_answer(
            model,
            tokenizer,
            prompt,
            device,
            do_print=False,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            stop_str=stop_str,
        )
        return response.strip()

    @app.get('/')
    async def index():
        return {"message": "index, docs url: /docs"}

    @app.post('/chat')
    async def chat(item: Item):
        try:
            response = predict(item.input)
            result_dict = {'response': response}
            logger.debug(f"Successfully get result, q:{item.input}")
            return result_dict
        except Exception as e:
            logger.error(e)
            return None

    uvicorn.run(app=app, host='0.0.0.0', port=args.port, workers=1)


if __name__ == '__main__':
    main()
