import json
import random

from openai import OpenAI
from tqdm import tqdm

client = OpenAI()
print(client)


def generate(prompt):
    print('提示：', prompt)
    messages = [
        {"role": "user", "content": prompt}
    ]
    r = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        temperature=1,
        max_tokens=3048,  # 增加max_tokens以生成更长的对话
    )
    response = r.choices[0].message.content
    print("生成的对话：", response)
    return response


file_role1 = "seed_nurse_role.jsonl"
file_role2 = "seed_patient_role.jsonl"
with open(file_role1, "r", encoding="utf-8") as file:
    role1s = file.readlines()
with open(file_role2, "r", encoding="utf-8") as file:
    role2s = file.readlines()

save_file = "roleplay_train_data_v1.jsonl"
total_lines = 1000

with tqdm(total=total_lines, desc="生成对话") as pbar:
    while pbar.n < total_lines:
        role1 = random.choice(role1s)
        role2 = random.choice(role2s)
        data1 = json.loads(role1.strip())['system_prompt']
        data2 = json.loads(role2.strip())['system_prompt']
        p = "你是护士，跟患者对话。\n\n护士角色：" + data1 + '\n患者角色：' + data2
        conversation = {"id": str(pbar.n), "system_prompt": p, "conversations": []}

        combined_prompt = f"你扮演一个护士，以下对话是你和患者之间的对话。\n护士角色：{data1}\n患者角色：{data2}\n"
        combined_prompt += "进行多轮问答（6轮以上）。患者说话以`患者:`开头，护士说话以`护士:`开头。患者先提问。\n"

        prompt = combined_prompt + "\n对话开始：\n "
        response = generate(prompt)

        # 解析生成的多轮对话
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith("患者"):
                conversation["conversations"].append({"from": "human", "value": line.split("患者")[1].strip()[1:]})
            elif line.startswith("护士"):
                conversation["conversations"].append({"from": "gpt", "value": line.split("护士")[1].strip()[1:]})

        with open(save_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

        pbar.update(1)
        if pbar.n >= total_lines:
            break
