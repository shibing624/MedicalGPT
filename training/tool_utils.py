# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Tuple, Union

from datasets import DatasetDict, concatenate_datasets, load_dataset


class FunctionCall(NamedTuple):
    name: str
    arguments: str


DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}])\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```)\n"""
    "```\n"
)

GLM4_TOOL_PROMPT = (
    "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，"
    "你的任务是针对用户的问题和要求提供适当的答复和支持。# 可用工具{tool_text}"
)

LLAMA3_TOOL_PROMPT = (
    "Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n"
    "You have access to the following functions. To call a function, please respond with JSON for a function call. "
    """Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}. """
    "Do not use variables.\n\n{tool_text}"
)

QWEN_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    """<tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, """
    """"arguments": <args-json-object>}}\n</tool_call>"""
)


@dataclass
class ToolUtils(ABC):
    """
    Base class for tool utilities.
    """

    @staticmethod
    @abstractmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        r"""
        Generates the system message describing all the available tools.
        """
        ...

    @staticmethod
    @abstractmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        r"""
        Generates the assistant message including all the tool calls.
        """
        ...

    @staticmethod
    @abstractmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        r"""
        Extracts all the function calls from the assistant message.

        It should be an inverse function of `function_formatter`.
        """
        ...


class DefaultToolUtils(ToolUtils):
    r"""
    Default tool using template.
    """

    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        tool_names = []
        for tool in tools:
            param_text = ""
            for name, param in tool["parameters"]["properties"].items():
                required, enum, items = "", "", ""
                if name in tool["parameters"].get("required", []):
                    required = ", required"

                if param.get("enum", None):
                    enum = ", should be one of [{}]".format(", ".join(param["enum"]))

                if param.get("items", None):
                    items = ", where each item should be {}".format(param["items"].get("type", ""))

                param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                    name=name,
                    type=param.get("type", ""),
                    required=required,
                    desc=param.get("description", ""),
                    enum=enum,
                    items=items,
                )

            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=tool["name"], desc=tool.get("description", ""), args=param_text
            )
            tool_names.append(tool["name"])

        return DEFAULT_TOOL_PROMPT.format(tool_text=tool_text, tool_names=", ".join(tool_names))

    @staticmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        function_text = ""
        for name, arguments in functions:
            function_text += f"Action: {name}\nAction Input: {arguments}\n"

        return function_text

    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(.+?)(?=\s*Action:|\s*$)", re.DOTALL)
        action_match: List[Tuple[str, str]] = re.findall(regex, content)
        if not action_match:
            return content

        results = []
        for match in action_match:
            tool_name = match[0].strip()
            tool_input = match[1].strip().strip('"').strip("```")
            try:
                arguments = json.loads(tool_input)
                results.append(FunctionCall(tool_name, json.dumps(arguments, ensure_ascii=False)))
            except json.JSONDecodeError:
                return content

        return results


class GLM4ToolUtils(ToolUtils):
    r"""
    GLM-4 tool using template.
    """

    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            tool_text += "\n\n## {name}\n\n{body}\n在调用上述函数时，请使用 Json 格式表示调用的参数。".format(
                name=tool["name"], body=json.dumps(tool, indent=4, ensure_ascii=False)
            )

        return GLM4_TOOL_PROMPT.format(tool_text=tool_text)

    @staticmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        if len(functions) > 1:
            raise ValueError("GLM-4 does not support parallel functions.")

        return f"{functions[0].name}\n{functions[0].arguments}"

    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        if "\n" not in content:
            return content

        tool_name, tool_input = content.split("\n", maxsplit=1)
        try:
            arguments = json.loads(tool_input.strip())
        except json.JSONDecodeError:
            return content

        return [FunctionCall(tool_name, json.dumps(arguments, ensure_ascii=False))]


class Llama3ToolUtils(ToolUtils):
    r"""
    Llama 3.x tool using template with `tools_in_user_message=False`.

    Reference: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling
    """

    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        date = datetime.now().strftime("%d %b %Y")
        tool_text = ""
        for tool in tools:
            wrapped_tool = {"type": "function", "function": tool}
            tool_text += json.dumps(wrapped_tool, indent=4, ensure_ascii=False) + "\n\n"

        return LLAMA3_TOOL_PROMPT.format(date=date, tool_text=tool_text)

    @staticmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        if len(functions) > 1:
            raise ValueError("Llama-3 does not support parallel functions.")

        return f'{{"name": "{functions[0].name}", "parameters": {functions[0].arguments}}}'

    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        try:
            tool = json.loads(content.strip())
        except json.JSONDecodeError:
            return content

        if "name" not in tool or "parameters" not in tool:
            return content

        return [FunctionCall(tool["name"], json.dumps(tool["parameters"], ensure_ascii=False))]


class MistralToolUtils(ToolUtils):
    r"""
    Mistral v0.3 tool using template.
    """

    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        wrapped_tools = []
        for tool in tools:
            wrapped_tools.append({"type": "function", "function": tool})

        return "[AVAILABLE_TOOLS] " + json.dumps(wrapped_tools, ensure_ascii=False) + "[/AVAILABLE_TOOLS]"

    @staticmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        function_texts = []
        for name, arguments in functions:
            function_texts.append(f'{{"name": "{name}", "arguments": {arguments}}}')

        return "[" + ", ".join(function_texts) + "]"

    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        try:
            tools = json.loads(content.strip())
        except json.JSONDecodeError:
            return content

        if not isinstance(tools, list):
            tools = [tools]

        results = []
        for tool in tools:
            if "name" not in tool or "arguments" not in tool:
                return content

            results.append(FunctionCall(tool["name"], json.dumps(tool["arguments"], ensure_ascii=False)))

        return results


class QwenToolUtils(ToolUtils):
    r"""
    Qwen tool using template. Compatible with Qwen2.5, Qwen3 and Qwen3.5.
    """

    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return QWEN_TOOL_PROMPT.format(tool_text=tool_text)

    @staticmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        function_texts = []
        for name, arguments in functions:
            function_texts.append(
                "<tool_call>\n" + f'{{"name": "{name}", "arguments": {arguments}}}' + "\n</tool_call>"
            )

        return "\n".join(function_texts)

    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        regex = re.compile(r"<tool_call>(.+?)</tool_call>(?=\s*<tool_call>|\s*$)", re.DOTALL)
        tool_match: List[str] = re.findall(regex, content)
        if not tool_match:
            return content

        results = []
        for tool in tool_match:
            try:
                tool = json.loads(tool.strip())
            except json.JSONDecodeError:
                return content

            if "name" not in tool or "arguments" not in tool:
                return content

            results.append(FunctionCall(tool["name"], json.dumps(tool["arguments"], ensure_ascii=False)))

        return results


TOOLS = {
    "default": DefaultToolUtils(),
    "glm4": GLM4ToolUtils(),
    "llama3": Llama3ToolUtils(),
    "mistral": MistralToolUtils(),
    "qwen": QwenToolUtils(),
    "qwen3": QwenToolUtils(),
    "qwen3.5": QwenToolUtils(),
}


def get_tool_utils(name: str) -> "ToolUtils":
    tool_utils = TOOLS.get(name, None)
    if tool_utils is None:
        raise ValueError(f"Tool utils `{name}` not found.")

    return tool_utils


def load_local_json_datasets(data_files, cache_dir=None):
    raw_datasets = DatasetDict()
    for split_name, split_files in data_files.items():
        if not split_files:
            continue
        split_datasets = []
        feature_by_column = {}
        column_order = []
        for file_path in split_files:
            split_dataset = load_dataset(
                "json",
                data_files=file_path,
                split="train",
                cache_dir=cache_dir,
            )
            split_datasets.append(split_dataset)
            for column_name in split_dataset.column_names:
                if column_name not in feature_by_column:
                    feature_by_column[column_name] = split_dataset.features[column_name]
                    column_order.append(column_name)
        aligned_datasets = []
        for split_dataset in split_datasets:
            for column_name in column_order:
                if column_name not in split_dataset.column_names:
                    split_dataset = split_dataset.add_column(
                        column_name,
                        [None] * len(split_dataset),
                        feature=feature_by_column[column_name],
                    )
            aligned_datasets.append(split_dataset.select_columns(column_order))
        raw_datasets[split_name] = concatenate_datasets(aligned_datasets)
    return raw_datasets
