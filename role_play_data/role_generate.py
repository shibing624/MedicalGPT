import json
import random

from openai import OpenAI
from tqdm import tqdm

client = OpenAI()
print(client)


def generate(prompt):
    print(prompt)
    messages = [
        {"role": "user", "content": prompt}
    ]
    r = client.chat.completions.create(
        model='gpt-4o',
        temperature=1,
        messages=messages, )
    response = r.choices[0].message.content
    print("回答：", response)
    return response


def generate_role(input_file, save_file, total_lines):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    with tqdm(total=total_lines, desc="指令进度") as pbar:
        while pbar.n < total_lines:
            random.shuffle(lines)
            i = 0
            sum_str = ""
            for line in lines:
                i += 1
                try:
                    data = json.loads(line.strip())
                except:
                    print("error:", line.strip())
                    continue
                question = data["system_prompt"]

                sum_str += f"{i}.{question}\n\n"

                if i == 5:
                    res = generate(f'请续写下面内容，不少于10条，增加些多样性。\n\n{sum_str}')
                    res = res.split("\n\n")
                    for result in res:
                        result = result.strip()
                        prefix_length = len(result.split(".", 1)[0]) + 1  # 获取前缀数字的长度，包括后面的点号
                        result = result[prefix_length:]
                        if result == "":
                            continue
                        json_data = {'system_prompt': result}
                        # 将数据写入文件
                        with open(save_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(json_data, ensure_ascii=False) + '\n')

                        pbar.update(1)
            if pbar.n >= total_lines:
                break


if __name__ == '__main__':
    total_lines = 50
    input_file = "seed_nurse_role.jsonl"
    save_file = "seed_nurse_role_output.jsonl"
    generate_role(input_file, save_file, total_lines)

    total_lines = 50
    input_file = "seed_patient_role.jsonl"
    save_file = "seed_patient_role_output.jsonl"
    generate_role(input_file, save_file, total_lines)
