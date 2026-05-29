"""
LEAF Credit Agent — LLM Provider Abstraction
Wraps OpenAI and Anthropic behind a single interface.
Switching provider = changing one line in config.

Usage:
    provider = LLMProvider(api_key="sk-...", provider="openai")
    response = provider.complete(system_prompt, user_prompt)
"""

import os
from typing import Literal


class LLMProvider:
    """
    Single interface for LLM providers.
    provider = "openai"    → uses gpt-4o
    provider = "anthropic" → uses claude-sonnet-4-6
    """

    def __init__(
        self,
        api_key: str,
        provider: Literal["openai", "anthropic"] = "openai",
        model: str = None,
    ):
        self.provider = provider
        self.api_key = api_key

        if provider == "openai":
            self.model = model or "gpt-4o"
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("Run: pip install openai")

        elif provider == "anthropic":
            self.model = model or "claude-sonnet-4-6"
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("Run: pip install anthropic")

        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.2,
    ) -> str:
        """
        Send a prompt and return the text response.
        Low temperature (0.2) for consistent, factual explanations.
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()


def get_provider_from_env() -> LLMProvider:
    """
    Load LLM provider from environment variables.
    Set one of:
        OPENAI_API_KEY   → uses OpenAI
        ANTHROPIC_API_KEY → uses Anthropic
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_key:
        return LLMProvider(api_key=openai_key, provider="openai")
    elif anthropic_key:
        return LLMProvider(api_key=anthropic_key, provider="anthropic")
    else:
        raise EnvironmentError(
            "No LLM API key found.\n"
            "Set OPENAI_API_KEY or ANTHROPIC_API_KEY as an environment variable.\n"
            "Example: set OPENAI_API_KEY=sk-... (Windows)\n"
            "         export OPENAI_API_KEY=sk-... (Mac/Linux)"
        )
