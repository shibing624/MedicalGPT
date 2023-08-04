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
)
from transformers.generation import GenerationConfig

from supervised_finetuning import get_conv_template

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


class SimpleChatIO:
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)


@torch.inference_mode()
def generate_answer(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.0,
        context_len=2048
):
    input_ids = tokenizer(prompt).input_ids
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    generation_config = dict(
        input_ids=torch.as_tensor([input_ids]).to(device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    generation_output = model.generate(**generation_config)
    output_ids = generation_output[0]
    output = tokenizer.decode(output_ids, skip_special_tokens=False).strip()
    stop_str = tokenizer.eos_token or "</s>"
    l_prompt = len(tokenizer.decode(input_ids, skip_special_tokens=False))
    pos = output.find(stop_str, l_prompt)
    if pos != -1:
        output = output[l_prompt:pos]
    else:
        output = output[l_prompt:]
    return output


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
        context_len=2048
):
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=False)
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
    stop_str = tokenizer.eos_token or "</s>"
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
    parser.add_argument('--predictions_file', default='./predictions.json', type=str)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--use_stream', action='store_true', help='Whether to use stream generation')
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
    base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)

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

    chatio = SimpleChatIO()

    # Chat
    def new_chat():
        return get_conv_template(args.template_name)

    if args.interactive:
        print("Start inference with interactive mode. command: `clear`, `exit`")
        conv = new_chat()
        while True:
            try:
                inp = chatio.prompt_for_input(conv.roles[0])
            except EOFError:
                inp = ""
            except UnicodeDecodeError:
                print("UnicodeDecodeError, please try again.")
                continue
            if inp == "":
                print("Please input text, try again.")
                continue
            if inp == "exit":
                print("exit...")
                break
            if inp == "clear":
                print("history cleared.")
                conv = new_chat()
                continue

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], '')

            prompt, _ = conv.get_prompt()
            chatio.prompt_for_output(conv.roles[1])
            if args.use_stream:
                response = stream_generate_answer(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    do_print=True,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty
                )
            else:
                response = generate_answer(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty
                )
                print(response.strip(), flush=True)
            # NOTE: strip is important to align with the training data.
            conv.messages[-1][-1] = response.strip()
            # print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    else:
        print("Start inference.")
        results = []
        for index, example in enumerate(examples):
            conv = new_chat()
            conv.append_message(conv.roles[0], example)
            conv.append_message(conv.roles[1], '')

            prompt, _ = conv.get_prompt()
            response = generate_answer(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty
            )
            response = response.strip()
            print(f"======={index}=======")
            print(f"Input: {example}\n")
            print(f"Output: {response}\n")
            results.append({"Input": prompt, "Output": response})

        dirname = os.path.dirname(args.predictions_file)
        os.makedirs(dirname, exist_ok=True)
        with open(args.predictions_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
