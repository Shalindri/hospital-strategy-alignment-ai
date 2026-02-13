"""
PDF Processor for the ISPS system.

Extracts text from uploaded PDF files and uses Ollama (llama3.1:8b) to
parse the content into the structured JSON schema expected by the
downstream embedding and alignment pipeline.

Two LLM prompts are used:
  1. Strategic plan extraction → objectives with goals, KPIs, etc.
  2. Action plan extraction   → action items with titles, owners, budgets, etc.

Typical usage (from the Streamlit dashboard)::

    from src.pdf_processor import extract_strategic_plan_from_pdf, extract_action_plan_from_pdf

    strategic_data = extract_strategic_plan_from_pdf(uploaded_bytes)
    action_data    = extract_action_plan_from_pdf(uploaded_bytes)

Author : ISPS Team
Created: 2025-01
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pdfplumber

logger = logging.getLogger("pdf_processor")

# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF file given as bytes.

    Uses pdfplumber which handles multi-column layouts, tables, and
    embedded fonts better than simpler libraries.

    Args:
        pdf_bytes: Raw PDF file content.

    Returns:
        Concatenated text from all pages.
    """
    import io
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    full_text = "\n\n".join(text_parts)
    logger.info("Extracted %d characters from PDF (%d pages).",
                len(full_text), len(text_parts))
    return full_text


# ---------------------------------------------------------------------------
# Ollama LLM helper
# ---------------------------------------------------------------------------

def _query_ollama(prompt: str, model: str = "llama3.1:8b") -> str:
    """Send a prompt to the local Ollama API and return the response."""
    import urllib.request

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 4096},
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    return data.get("response", "")


def _extract_json_from_response(text: str) -> dict | list:
    """Find and parse the first JSON object or array in LLM output."""
    # Try to find JSON block in markdown code fences
    match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
    if match:
        return json.loads(match.group(1))

    # Try to find raw JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching closing brace/bracket
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
    raise ValueError("No valid JSON found in LLM response")


# ---------------------------------------------------------------------------
# Strategic plan extraction
# ---------------------------------------------------------------------------

_STRATEGIC_PROMPT = """\
You are an expert at analyzing strategic plans. Read the following document \
text and extract the strategic objectives in JSON format.

Return a JSON object with this EXACT structure:
{{
  "metadata": {{
    "title": "<document title>",
    "period": "<planning period, e.g. 2026-2030>",
    "version": "1.0"
  }},
  "vision": "<vision statement if present, else empty string>",
  "mission": "<mission statement if present, else empty string>",
  "objectives": [
    {{
      "code": "<letter A, B, C, etc.>",
      "name": "<objective name>",
      "goal_statement": "<main goal description>",
      "strategic_goals": [
        {{"id": "<e.g. A1>", "description": "<goal description>"}}
      ],
      "kpis": [
        {{"KPI": "<kpi name>", "Baseline": "<current value>", "Target": "<target value>"}}
      ],
      "keywords": ["<keyword1>", "<keyword2>"]
    }}
  ]
}}

Rules:
- Extract ALL strategic objectives found in the document
- Assign letter codes (A, B, C, ...) if not explicitly labeled
- Extract as many strategic goals and KPIs as mentioned
- Keywords should be 5-10 domain-relevant terms per objective
- Return ONLY valid JSON, no other text

DOCUMENT TEXT:
{text}
"""

_ACTION_PROMPT = """\
You are an expert at analyzing action plans. Read the following document \
text and extract the action items in JSON format.

Return a JSON object with this EXACT structure:
{{
  "metadata": {{
    "title": "<document title>",
    "period": "<action plan period, e.g. 2025>",
    "version": "1.0"
  }},
  "actions": [
    {{
      "action_number": <integer>,
      "title": "<action title>",
      "strategic_objective_code": "<letter matching the strategic objective>",
      "strategic_objective_name": "<objective name>",
      "description": "<full description of the action>",
      "action_owner": "<responsible person/department>",
      "timeline": "<e.g. Q1-Q4 2025>",
      "quarters": ["Q1", "Q2"],
      "budget_lkr_millions": <number or 0>,
      "budget_raw": "<original budget text>",
      "expected_outcome": "<expected result>",
      "kpis": ["<kpi1>", "<kpi2>"],
      "keywords": ["<keyword1>", "<keyword2>"]
    }}
  ]
}}

Rules:
- Extract ALL action items found in the document
- Number them sequentially starting from 1 if not explicitly numbered
- Map each action to its strategic objective using the letter code
- Extract budget as a number in millions if possible (set 0 if not mentioned)
- Extract timeline quarters (Q1, Q2, Q3, Q4) when mentioned
- Keywords should be 5-8 domain-relevant terms per action
- Return ONLY valid JSON, no other text

DOCUMENT TEXT:
{text}
"""


def extract_strategic_plan_from_pdf(
    pdf_bytes: bytes,
    model: str = "llama3.1:8b",
) -> dict[str, Any]:
    """Extract strategic plan structure from a PDF using LLM.

    Args:
        pdf_bytes: Raw PDF file content.
        model:     Ollama model name.

    Returns:
        Structured strategic plan dict matching the schema from
        document_processor.py.

    Raises:
        ConnectionError: If Ollama is not running.
        ValueError:      If LLM output cannot be parsed as JSON.
    """
    text = extract_text_from_pdf(pdf_bytes)
    if not text.strip():
        raise ValueError("PDF appears to be empty or image-only.")

    # Truncate to ~6000 chars to fit model context
    truncated = text[:6000]
    prompt = _STRATEGIC_PROMPT.format(text=truncated)

    logger.info("Sending strategic plan to LLM for parsing (%d chars)...",
                len(truncated))
    response = _query_ollama(prompt, model)
    result = _extract_json_from_response(response)

    # Validate minimum structure
    if isinstance(result, dict) and "objectives" in result:
        n_obj = len(result["objectives"])
        logger.info("LLM extracted %d strategic objectives.", n_obj)
        # Ensure all objectives have required fields
        for obj in result["objectives"]:
            obj.setdefault("strategic_goals", [])
            obj.setdefault("kpis", [])
            obj.setdefault("keywords", [])
            obj.setdefault("goal_statement", obj.get("name", ""))
        return result

    raise ValueError(
        "LLM response does not contain expected 'objectives' key."
    )


def extract_action_plan_from_pdf(
    pdf_bytes: bytes,
    model: str = "llama3.1:8b",
) -> dict[str, Any]:
    """Extract action plan structure from a PDF using LLM.

    Args:
        pdf_bytes: Raw PDF file content.
        model:     Ollama model name.

    Returns:
        Structured action plan dict matching the schema from
        document_processor.py.

    Raises:
        ConnectionError: If Ollama is not running.
        ValueError:      If LLM output cannot be parsed as JSON.
    """
    text = extract_text_from_pdf(pdf_bytes)
    if not text.strip():
        raise ValueError("PDF appears to be empty or image-only.")

    # Truncate to ~6000 chars to fit model context
    truncated = text[:6000]
    prompt = _ACTION_PROMPT.format(text=truncated)

    logger.info("Sending action plan to LLM for parsing (%d chars)...",
                len(truncated))
    response = _query_ollama(prompt, model)
    result = _extract_json_from_response(response)

    # Validate minimum structure
    if isinstance(result, dict) and "actions" in result:
        n_act = len(result["actions"])
        logger.info("LLM extracted %d action items.", n_act)
        # Ensure all actions have required fields
        for act in result["actions"]:
            act.setdefault("description", act.get("title", ""))
            act.setdefault("expected_outcome", "")
            act.setdefault("kpis", [])
            act.setdefault("keywords", [])
            act.setdefault("action_owner", "")
            act.setdefault("timeline", "")
            act.setdefault("quarters", [])
            act.setdefault("budget_lkr_millions", 0.0)
            act.setdefault("budget_raw", "")
        return result

    raise ValueError(
        "LLM response does not contain expected 'actions' key."
    )


def check_ollama_available(model: str = "llama3.1:8b") -> bool:
    """Check if Ollama is running and the model is available."""
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            return any(model in m for m in models)
    except Exception:
        return False
