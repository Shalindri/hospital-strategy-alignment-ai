"""
Shared configuration constants for the ISPS alignment system.

Centralises thresholds and defaults that are used across multiple modules
so they are defined in exactly one place.

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger("config")

# ---------------------------------------------------------------------------
# Alignment classification thresholds
# ---------------------------------------------------------------------------
# Calibrated against the all-MiniLM-L6-v2 model's typical similarity range
# for hospital-domain documents (unrelated pairs: 0.05–0.20, strongly
# related pairs: 0.45–0.65).

THRESHOLD_EXCELLENT = 0.75   # Near-direct operationalisation of strategy
THRESHOLD_GOOD = 0.60        # Clear strategic support
THRESHOLD_FAIR = 0.45        # Partial or indirect alignment
ORPHAN_THRESHOLD = 0.45      # Below this for ALL objectives → orphan

# ---------------------------------------------------------------------------
# Default hospital name (used when no name is provided in data)
# ---------------------------------------------------------------------------
DEFAULT_HOSPITAL_NAME = "Hospital"

# ---------------------------------------------------------------------------
# LLM Provider Configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")         # "ollama" or "openai"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Shared LLM parameters
LLM_TEMPERATURE = 0.2
LLM_NUM_CTX = 4096

# Agent configuration
MAX_ITERATIONS = 3
MAX_RETRIES = 3
RETRY_DELAYS = [1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

class _OpenAIWrapper:
    """Wraps ChatOpenAI so .invoke(str) returns str (not AIMessage)."""

    def __init__(self, llm):
        self._llm = llm

    def invoke(self, prompt: str) -> str:
        msg = self._llm.invoke(prompt)
        return msg.content


def get_llm(temperature: float = LLM_TEMPERATURE):
    """Return a LangChain-compatible LLM based on LLM_PROVIDER env var.

    Both Ollama and OpenAI objects expose ``.invoke(prompt_str) -> str``.
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        if not OPENAI_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
                "Add it to your .env file."
            )
        logger.info("Using OpenAI model: %s", OPENAI_MODEL)
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        )
        return _OpenAIWrapper(llm)
    else:
        from langchain_ollama import OllamaLLM

        logger.info("Using Ollama model: %s", OLLAMA_MODEL)
        return OllamaLLM(
            model=OLLAMA_MODEL,
            temperature=temperature,
            num_ctx=LLM_NUM_CTX,
        )
