# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

pip install gradio
pip install mdtex2html
"""
import argparse
import os
from threading import Thread

import gradio as gr
import mdtex2html
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


@torch.inference_mode()
def stream_generate_answer(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.0,
        context_len=2048
):
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    input_ids = tokenizer(prompt).input_ids
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    generation_kwargs = dict(
        input_ids=torch.as_tensor([input_ids]).to(device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    yield from streamer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="vicuna", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan-chat, chatglm2 etc.")
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    args = parser.parse_args()
    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    def postprocess(self, y):
        if y is None:
            return []
        for i, (message, response) in enumerate(y):
            y[i] = (
                None if message is None else mdtex2html.convert((message)),
                None if response is None else mdtex2html.convert(response),
            )
        return y

    gr.Chatbot.postprocess = postprocess

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
        print("loaded lora model")
    else:
        model = base_model
    if device == torch.device('cpu'):
        model.float()

    model.eval()

    def reset_user_input():
        return gr.update(value='')

    def reset_state():
        return [], []

    prompt_template = get_conv_template(args.template_name)
    history = []

    def predict(
            input,
            chatbot,
            history,
            max_new_tokens,
            temperature,
            top_p
    ):
        now_input = input
        chatbot.append((input, ""))
        history = history or []
        history.append([now_input, ''])

        prompt = prompt_template.get_prompt(messages=history)
        response = ""
        for new_text in stream_generate_answer(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
        ):
            response += new_text
            new_history = history + [(now_input, response)]
            chatbot[-1] = (now_input, response)
            yield chatbot, new_history

    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">MedicalGPT</h1>""")
        gr.Markdown(
            "> 为了促进医疗行业大模型的开放研究，本项目开源了MedicalGPT医疗大模型")
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                        container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(
                    0, 4096, value=512, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01,
                                  label="Top P", interactive=True)
                temperature = gr.Slider(
                    0, 1, value=0.7, step=0.01, label="Temperature", interactive=True)
        history = gr.State([])
        submitBtn.click(predict, [user_input, chatbot, history, max_length, temperature, top_p], [chatbot, history],
                        show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])
        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)
    demo.queue().launch(share=False, inbrowser=True, server_name='0.0.0.0', server_port=8082)


if __name__ == '__main__':
    main()
