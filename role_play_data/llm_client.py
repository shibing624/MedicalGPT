# -*- coding: utf-8 -*-
"""
@description: Multi-provider LLM client for medical data generation.

Supports OpenAI, Doubao (ByteDance), and MiniMax via OpenAI-compatible API.

Usage:
    from llm_client import create_llm_client

    # Auto-detect provider from environment variables
    client, model = create_llm_client()

    # Or specify provider explicitly
    client, model = create_llm_client(provider="minimax")
    client, model = create_llm_client(provider="openai")
    client, model = create_llm_client(provider="doubao")
"""

import os

from openai import OpenAI

# Provider configurations
PROVIDER_CONFIGS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,  # Uses default OpenAI endpoint
        "default_model": "gpt-4o",
    },
    "doubao": {
        "env_key": "DOUBAO_API_KEY",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "ep-20240623141021-r77gl",
    },
    "minimax": {
        "env_key": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.io/v1",
        "default_model": "MiniMax-M2.7",
    },
}

# Detection order for auto-detection
_DETECTION_ORDER = ["openai", "minimax", "doubao"]


def detect_provider():
    """Auto-detect LLM provider from environment variables.

    Returns the first provider whose API key is found in the environment.
    Detection order: openai, minimax, doubao.
    """
    for provider in _DETECTION_ORDER:
        config = PROVIDER_CONFIGS[provider]
        if os.environ.get(config["env_key"]):
            return provider
    return None


def create_llm_client(provider=None, api_key=None, base_url=None, model=None):
    """Create an OpenAI-compatible client for the specified LLM provider.

    Args:
        provider: Provider name ("openai", "doubao", "minimax").
                  If None, auto-detects from environment variables.
        api_key: API key override. If None, reads from environment.
        base_url: Base URL override. If None, uses provider default.
        model: Model name override. If None, uses provider default.

    Returns:
        Tuple of (OpenAI client, model name).

    Raises:
        ValueError: If provider is unknown or no API key is found.
    """
    if provider is None:
        provider = detect_provider()
        if provider is None:
            raise ValueError(
                "No LLM provider detected. Set one of: "
                + ", ".join(c["env_key"] for c in PROVIDER_CONFIGS.values())
            )

    provider = provider.lower()
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: {', '.join(PROVIDER_CONFIGS.keys())}"
        )

    config = PROVIDER_CONFIGS[provider]
    resolved_key = api_key or os.environ.get(config["env_key"])
    if not resolved_key:
        raise ValueError(
            f"API key not found for provider '{provider}'. "
            f"Set {config['env_key']} environment variable or pass api_key parameter."
        )

    resolved_url = base_url or config["base_url"]
    resolved_model = model or config["default_model"]

    kwargs = {"api_key": resolved_key}
    if resolved_url:
        kwargs["base_url"] = resolved_url

    client = OpenAI(**kwargs)
    return client, resolved_model
