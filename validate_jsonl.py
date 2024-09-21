# -*- coding: utf-8 -*-
"""
@author:ZhuangXialie(1832963123@qq.com)
@description: validata the dataset
"""
import json

import argparse


def validate_jsonl(file_path):
    print("开始验证 JSONL 文件格式...\n")

    with open(file_path, 'r', encoding='utf-8') as file:
        line_number = 0
        valid_lines = 0
        total_lines = 0
        for line in file:
            total_lines += 1
            line_number += 1
            try:
                # 尝试解析JSON
                data = json.loads(line)

                # 检查是否包含 'conversations' 键
                if 'conversations' not in data:
                    print(f"第 {line_number} 行: 缺少 'conversations' 键，请检查格式。\n")
                    continue

                # 检查 'conversations' 是否为列表
                conversations = data['conversations']
                if not isinstance(conversations, list):
                    print(f"第 {line_number} 行: 'conversations' 应为列表格式，请检查。\n")
                    continue

                # 检查每个对话是否包含 'from' 和 'value' 键
                conversation_valid = True
                for conv in conversations:
                    if 'from' not in conv or 'value' not in conv:
                        print(f"第 {line_number} 行: 缺少 'from' 或 'value' 键，请检查对话格式。\n")
                        conversation_valid = False
                        continue

                    # 检查 'from' 字段的值是否为 'human' 或 'gpt'
                    if conv['from'] not in ['system', 'human', 'gpt']:
                        print(f"第 {line_number} 行: 'from' 字段的值无效，应为 'human' 或 'gpt'。\n")
                        conversation_valid = False

                if conversation_valid:
                    valid_lines += 1

            except json.JSONDecodeError:
                print(f"第 {line_number} 行: JSON 格式无效，请确保格式正确。\n")

    print(f"验证完成！\n总行数: {total_lines} 行")
    print(f"有效的行数: {valid_lines} 行")
    print(f"无效行数: {total_lines - valid_lines} 行\n")

    if valid_lines == total_lines:
        print("恭喜！所有行的格式都正确。")
    else:
        print("请根据提示检查并修复无效的行。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate JSONL file format.')
    parser.add_argument('--file_path', type=str, help='Path to JSONL file',
                        default="./data/finetune/sharegpt_zh_1K_format.jsonl")
    args = parser.parse_args()
    file_path = args.file_path
    print(f"正在检查文件: {file_path}")

    validate_jsonl(file_path)
