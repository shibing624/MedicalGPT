# -*- coding: utf-8 -*-
"""
@description: Generate medical roleplay dialogue training data using MiniMax API.

MiniMax provides an OpenAI-compatible API at https://api.minimax.io/v1
with models like MiniMax-M2.7 (1M context) and MiniMax-M2.5-highspeed (204K context).

Usage:
    export MINIMAX_API_KEY="your_api_key"
    python roleplay_data_generate_minimax.py

    # Or specify a different model
    python roleplay_data_generate_minimax.py --model MiniMax-M2.5-highspeed

For an API key, visit: https://platform.minimaxi.com/
"""

import argparse
import json
import random

from tqdm import tqdm

from llm_client import create_llm_client


def generate(client, model, prompt, system_prompt=''):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=3048,
        temperature=0.9,
    )
    response = completion.choices[0].message.content
    return response


def main():
    parser = argparse.ArgumentParser(description="Generate roleplay data with MiniMax")
    parser.add_argument("--model", type=str, default=None,
                        help="MiniMax model to use (default: MiniMax-M2.7)")
    parser.add_argument("--total", type=int, default=1000,
                        help="Number of conversations to generate")
    parser.add_argument("--output", type=str, default="roleplay_train_data_minimax.jsonl",
                        help="Output file path")
    parser.add_argument("--max_history", type=int, default=10,
                        help="Max history turns to include in prompt")
    parser.add_argument("--rounds", type=int, default=6,
                        help="Number of dialogue rounds per conversation")
    args = parser.parse_args()

    client, model = create_llm_client(provider="minimax", model=args.model)
    print(f"Using MiniMax model: {model}")

    file_role1 = "seed_nurse_role.jsonl"
    file_role2 = "seed_patient_role.jsonl"
    with open(file_role1, "r", encoding="utf-8") as f:
        role1s = f.readlines()
    with open(file_role2, "r", encoding="utf-8") as f:
        role2s = f.readlines()

    with tqdm(total=args.total, desc="生成对话") as pbar:
        while pbar.n < args.total:
            role1 = random.choice(role1s)
            role2 = random.choice(role2s)
            data1 = json.loads(role1.strip())['system_prompt']
            data2 = json.loads(role2.strip())['system_prompt']
            p = "你是护士，跟患者对话。\n\n护士角色：" + data1 + '\n患者角色：' + data2
            conversation = {"id": str(pbar.n), "system_prompt": p, "conversations": []}

            system_prompt = f"护士角色：{data1}\n患者角色：{data2}\n"
            history = []

            for i in range(args.rounds):
                patient_prompt = "要求你扮演患者，并且根据角色的设定内容模仿角色相应的对话口吻和风格。你说一句话，完成本轮对话即可。"
                for history_turn in history[-args.max_history:]:
                    patient_prompt += history_turn + '\n'
                patient_prompt += "患者:"

                patient_response = generate(client, model, patient_prompt, system_prompt)
                conversation["conversations"].append(
                    {"from": "human", "value": patient_response.strip()}
                )
                history.append("患者:" + patient_response.strip())

                nurse_prompt = "要求你扮演护士，并且根据角色的设定内容模仿角色相应的对话口吻和风格。你说一句话,完成本轮对话即可。\n"
                for history_turn in history[-args.max_history:]:
                    nurse_prompt += history_turn + '\n'
                nurse_prompt += "护士:"

                nurse_response = generate(client, model, nurse_prompt, system_prompt)
                conversation["conversations"].append(
                    {"from": "gpt", "value": nurse_response.strip()}
                )
                history.append("护士: " + nurse_response.strip())

            with open(args.output, 'a', encoding='utf-8') as f:
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

            pbar.update(1)
            if pbar.n >= args.total:
                break


if __name__ == "__main__":
    main()
