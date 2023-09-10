# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import json
import os
from threading import Thread

import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
)

from supervised_finetuning import get_conv_template

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


@torch.inference_mode()
def stream_generate_answer(
        model,
        tokenizer,
        prompt,
        device,
        do_print=True,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.0,
        context_len=2048,
        stop_str="</s>",
):
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    input_ids = tokenizer(prompt).input_ids
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    generation_kwargs = dict(
        input_ids=torch.as_tensor([input_ids]).to(device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="vicuna", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan-chat, chatglm2 etc.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument('--data_file', default=None, type=str,
                        help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--interactive', action='store_true', help="run in the instruction mode (single-turn)")
    parser.add_argument('--predictions_file', default='./predictions_result.jsonl', type=str)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
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
        load_in_8bit=False,
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
    # test data
    if args.data_file is None:
        examples = ["介绍下北京", "乙肝和丙肝的区别？"]
    else:
        with open(args.data_file, 'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)

    # Chat
    prompt_template = get_conv_template(args.template_name)
    stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str

    if args.interactive:
        print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")
        history = []
        while True:
            try:
                query = input(f"{prompt_template.roles[0]}: ")
            except UnicodeDecodeError:
                print("Detected decoding error at the inputs, please try again.")
                continue
            except Exception:
                raise
            if query == "":
                print("Please input text, try again.")
                continue
            if query.strip() == "exit":
                print("exit...")
                break
            if query.strip() == "clear":
                history = []
                print("history cleared.")
                continue

            print(f"{prompt_template.roles[1]}: ", end="", flush=True)

            history.append([query, ''])
            prompt = prompt_template.get_prompt(messages=history)
            response = stream_generate_answer(
                model,
                tokenizer,
                prompt,
                device,
                do_print=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                stop_str=stop_str,
            )
            if history:
                history[-1][-1] = response.strip()
    else:
        print("Start inference.")
        results = []
        for index, example in enumerate(examples):
            # Single turn inference
            history = [[example, '']]
            prompt = prompt_template.get_prompt(messages=history)
            response = stream_generate_answer(
                model,
                tokenizer,
                prompt,
                device,
                do_print=False,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                stop_str=stop_str,
            )
            response = response.strip()
            print(f"======={index}=======")
            print(f"Input: {example}\n")
            print(f"Output: {response}\n")
            results.append({"Input": prompt, "Output": response})

        with open(args.predictions_file, 'w', encoding='utf-8') as f:
            for entry in results:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f'save to {args.predictions_file}, size: {len(results)}')


if __name__ == '__main__':
    main()
