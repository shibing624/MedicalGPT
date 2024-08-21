import json
import random

from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key="xxx",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)
print(client)


def generate(prompt, system_prompt=''):
    print('提示：', prompt)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        #pro-32k: ep-20240623141021-r77gl
        #lite-4k:ep-20240623140948-92n2g
        model="ep-20240623141021-r77gl",  # your model endpoint ID
        messages=messages,
        max_tokens=3048,
    )
    response = completion.choices[0].message.content
    print("生成的对话：", response)
    return response


file_role1 = "seed_nurse_role.jsonl"
file_role2 = "seed_patient_role.jsonl"
with open(file_role1, "r", encoding="utf-8") as file:
    role1s = file.readlines()
with open(file_role2, "r", encoding="utf-8") as file:
    role2s = file.readlines()

save_file = "roleplay_train_data_v2.jsonl"
total_lines = 1000  # 10000
max_history_len = 10

with tqdm(total=total_lines, desc="生成对话") as pbar:
    while pbar.n < total_lines:
        role1 = random.choice(role1s)
        role2 = random.choice(role2s)
        data1 = json.loads(role1.strip())['system_prompt']
        data2 = json.loads(role2.strip())['system_prompt']
        p = "你是护士，跟患者对话。\n\n护士角色：" + data1 + '\n患者角色：' + data2
        conversation = {"id": str(pbar.n), "system_prompt": p, "conversations": []}

        system_prompt = f"护士角色：{data1}\n患者角色：{data2}\n"
        print('------' * 10)
        print('system_prompt:', system_prompt)
        history = []

        for i in range(6):
            patient_prompt = f"要求你扮演患者，并且根据角色的设定内容模仿 角色相应的对话口吻和风格。你说一句话，完成本轮对话即可。"
            for history_turn in history[-max_history_len:]:
                patient_prompt += history_turn + '\n'
            patient_prompt += "患者:"

            patient_response = generate(patient_prompt, system_prompt)
            conversation["conversations"].append({"from": "human", "value": patient_response.strip()})
            history.append("患者:" + patient_response.strip())

            nurse_prompt = f"要求你扮演护士，并且根据角色的设定内容模仿 角色相应的对话口吻和风格。你说一句话,完成本轮对话即可。\n"
            for history_turn in history[-max_history_len:]:
                nurse_prompt += history_turn + '\n'
            nurse_prompt += "护士:"

            nurse_response = generate(nurse_prompt, system_prompt)
            conversation["conversations"].append({"from": "gpt", "value": nurse_response.strip()})
            history.append("护士: " + nurse_response.strip())

        with open(save_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

        pbar.update(1)
        if pbar.n >= total_lines:
            break
