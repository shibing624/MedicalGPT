# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Inference script
"""
import argparse
import json
import os
from threading import Thread

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig,
    BitsAndBytesConfig,
)


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
    """Generate answer from prompt with GPT and stream the output"""
    # Streamer for handling token generation
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
        temperature=temperature,
        do_sample=True if temperature > 0.0 else False,
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


@torch.inference_mode()
def batch_generate_answer(
        sentences,
        model,
        tokenizer,
        system_prompt,
        device,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.0,
        stop_str="</s>",
):
    """Generate answers from prompts in batch mode"""
    generated_texts = []
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0.0 else False,
        repetition_penalty=repetition_penalty,
    )

    # Construct batch prompts
    prompts = []
    for s in sentences:
        # Construct dialogue message format
        messages = [{"role": "user", "content": s}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        # Generate prompt using apply_chat_template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    inputs_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs_tokens['input_ids'].to(device)
    attention_mask = inputs_tokens['attention_mask'].to(device)
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
    for gen_sequence in outputs:
        prompt_len = len(input_ids[0])
        gen_sequence = gen_sequence[prompt_len:]
        gen_text = tokenizer.decode(gen_sequence, skip_special_tokens=True)
        pos = gen_text.find(stop_str)
        if pos != -1:
            gen_text = gen_text[:pos]
        gen_text = gen_text.strip()
        generated_texts.append(gen_text)

    return generated_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--system_prompt', default="", type=str, help="System prompt")
    parser.add_argument('--stop_str', default="", type=str)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument('--data_file', default=None, type=str,
                        help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--interactive', action='store_true', help="Run in interactive mode (default multi-turn)")
    parser.add_argument('--single_tune', action='store_true', help='Whether to use single-turn model')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--output_file', default='./predictions_result.jsonl', type=str)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--load_in_8bit', action='store_true', help='Whether to load model in 8-bit mode')
    parser.add_argument('--load_in_4bit', action='store_true', help='Whether to load model in 4-bit mode')
    args = parser.parse_args()
    print(args)
    load_type = 'auto'
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, padding_side='left')
    config_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": load_type,
        "low_cpu_mem_usage": True,
        "device_map": 'auto',
    }
    if args.load_in_8bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=load_type,
        )
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **config_kwargs)
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, using default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resizing model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded LoRA model")
    else:
        model = base_model
    model.eval()
    print(tokenizer)
    # Test data
    if args.data_file is None:
        examples = ["介绍下北京", "乙肝和丙肝的区别？"]
    else:
        with open(args.data_file, 'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("First 10 examples:")
        for example in examples[:10]:
            print(example)

    # Chat
    system_prompt = args.system_prompt
    stop_str = args.stop_str or "</s>"  # Use default stop string

    if args.interactive:
        print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")
        history = []
        while True:
            try:
                query = input(f"User: ")
            except UnicodeDecodeError:
                print("Detected decoding error at the inputs, please try again.")
                continue
            except Exception:
                raise
            if query == "":
                print("Please input text, try again.")
                continue
            if query.strip() == "exit":
                print("Exiting...")
                break
            if query.strip() == "clear":
                history = []
                print("History cleared.")
                continue

            print(f"Assistant: ", end="", flush=True)
            if args.single_tune:
                history = []

            # Construct dialogue messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add historical dialogue
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})

            # Add current user query
            messages.append({"role": "user", "content": query})

            # Generate prompt using apply_chat_template
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            response = stream_generate_answer(
                model,
                tokenizer,
                prompt,
                model.device,
                do_print=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                stop_str=stop_str,
            )
            if history:
                history[-1][-1] = response.strip()
            history.append([query, response.strip()])
    else:
        print("Starting inference.")
        counts = 0
        if os.path.exists(args.output_file):
            os.remove(args.output_file)
        eval_batch_size = args.eval_batch_size
        for batch in tqdm(
                [
                    examples[i: i + eval_batch_size]
                    for i in range(0, len(examples), eval_batch_size)
                ],
                desc="Generating outputs",
        ):
            responses = batch_generate_answer(
                batch,
                model,
                tokenizer,
                system_prompt,
                model.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                stop_str=stop_str,
            )
            results = []
            for example, response in zip(batch, responses):
                print(f"===")
                print(f"Input: {example}")
                print(f"Output: {response}\n")
                results.append({"Input": example, "Output": response})
                counts += 1
            with open(args.output_file, 'a', encoding='utf-8') as f:
                for entry in results:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write('\n')
        print(f'Saved to {args.output_file}, size: {counts}')


if __name__ == '__main__':
    main()

