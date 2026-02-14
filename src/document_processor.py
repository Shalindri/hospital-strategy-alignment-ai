"""
Document Processor for Hospital Strategy–Action Plan Alignment System (ISPS).

This module loads the Nawaloka Hospital Negombo strategic plan and action plan
markdown files, parses their structured content using regex-based section
extraction and spaCy NLP, and outputs clean JSON representations suitable for
downstream embedding, alignment scoring, and knowledge-graph construction.

Typical usage::

    from src.document_processor import DocumentProcessor

    processor = DocumentProcessor()
    strategic = processor.process_strategic_plan()
    actions   = processor.process_action_plan()

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("document_processor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STRATEGIC_PLAN_FILE = DATA_DIR / "strategic_plan.md"
ACTION_PLAN_FILE = DATA_DIR / "action_plan.md"
STRATEGIC_JSON_OUT = DATA_DIR / "strategic_plan.json"
ACTION_JSON_OUT = DATA_DIR / "action_plan.json"

OBJECTIVE_CODES = {
    "A": "Patient Care Excellence",
    "B": "Digital Health Transformation",
    "C": "Research & Innovation",
    "D": "Workforce Development",
    "E": "Community & Regional Health Expansion",
}


# ---------------------------------------------------------------------------
# spaCy loader
# ---------------------------------------------------------------------------

def _load_spacy_model(model_name: str = "en_core_web_sm") -> Language:
    """Load a spaCy language model, downloading it automatically if absent.

    Args:
        model_name: The spaCy model identifier (default ``en_core_web_sm``).

    Returns:
        A loaded spaCy ``Language`` pipeline.

    Raises:
        OSError: If the model cannot be loaded or downloaded.
    """
    try:
        nlp = spacy.load(model_name)
        logger.info("spaCy model '%s' loaded successfully.", model_name)
    except OSError:
        logger.warning(
            "spaCy model '%s' not found. Downloading …", model_name
        )
        from spacy.cli import download  # noqa: WPS433

        download(model_name)
        nlp = spacy.load(model_name)
        logger.info("spaCy model '%s' downloaded and loaded.", model_name)
    return nlp


# ---------------------------------------------------------------------------
# Helper: markdown table parser
# ---------------------------------------------------------------------------

def _parse_markdown_table(text: str) -> list[dict[str, str]]:
    """Parse a markdown pipe-delimited table into a list of row dictionaries.

    The first row is treated as the header.  The separator row
    (``|---|---|``) is skipped.  Leading/trailing pipes and whitespace
    are stripped from each cell.

    Args:
        text: A string containing a single markdown table.

    Returns:
        A list of dictionaries, one per data row, keyed by header values.
        Returns an empty list when *text* contains no valid table.

    Example::

        >>> _parse_markdown_table(
        ...     "| Name | Age |\\n|---|---|\\n| Alice | 30 |"
        ... )
        [{'Name': 'Alice', 'Age': '30'}]
    """
    lines = [
        ln.strip()
        for ln in text.strip().splitlines()
        if ln.strip()
        and "|" in ln
        and not re.match(r"^\|[\s\-:|]+\|$", ln.strip())
    ]
    if len(lines) < 2:
        return []

    def _split_row(row: str) -> list[str]:
        row = row.strip().strip("|")
        return [cell.strip().strip("*").strip() for cell in row.split("|")]

    headers = _split_row(lines[0])
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        cells = _split_row(line)
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


# ---------------------------------------------------------------------------
# Helper: extract key entities via spaCy
# ---------------------------------------------------------------------------

def _extract_entities(nlp: Language, text: str) -> list[dict[str, str]]:
    """Run spaCy NER on *text* and return unique named entities.

    Args:
        nlp:  A loaded spaCy ``Language`` pipeline.
        text: The input text to analyse.

    Returns:
        A deduplicated list of ``{"text": …, "label": …}`` dictionaries
        for every entity recognised by the model.
    """
    doc = nlp(text)
    seen: set[tuple[str, str]] = set()
    entities: list[dict[str, str]] = []
    for ent in doc.ents:
        key = (ent.text.strip(), ent.label_)
        if key not in seen:
            seen.add(key)
            entities.append({"text": ent.text.strip(), "label": ent.label_})
    return entities


def _extract_keywords(nlp: Language, text: str, top_n: int = 15) -> list[str]:
    """Extract salient keywords from *text* using spaCy POS-based filtering.

    Nouns and proper nouns are selected, lemmatised, lowercased, and
    deduplicated.  Stop-words and tokens shorter than 3 characters are
    excluded.

    Args:
        nlp:   A loaded spaCy ``Language`` pipeline.
        text:  The input text to analyse.
        top_n: Maximum number of keywords to return.

    Returns:
        A list of up to *top_n* keyword strings sorted by first
        occurrence.
    """
    doc = nlp(text)
    keywords: list[str] = []
    seen: set[str] = set()
    for token in doc:
        if (
            token.pos_ in {"NOUN", "PROPN"}
            and not token.is_stop
            and len(token.lemma_) >= 3
        ):
            lemma = token.lemma_.lower()
            if lemma not in seen:
                seen.add(lemma)
                keywords.append(lemma)
    return keywords[:top_n]


# ---------------------------------------------------------------------------
# Strategic plan parser
# ---------------------------------------------------------------------------

def _extract_section(text: str, start_heading: str, stop_headings: list[str]) -> str:
    """Slice a markdown document between *start_heading* and the first
    subsequent heading found in *stop_headings*.

    Both ``##`` and ``###`` heading levels are supported.  The match is
    performed with ``re.IGNORECASE``.

    Args:
        text:          Full markdown text.
        start_heading: Regex-safe heading text to begin extraction (inclusive).
        stop_headings: List of regex-safe heading texts. Extraction stops
                       at the first match (exclusive).

    Returns:
        The extracted substring, or an empty string if *start_heading*
        is not found.
    """
    start_pattern = re.compile(
        rf"^#{{1,6}}\s+{start_heading}",
        re.MULTILINE | re.IGNORECASE,
    )
    start_match = start_pattern.search(text)
    if not start_match:
        return ""

    for stop in stop_headings:
        stop_pattern = re.compile(
            rf"^#{{1,6}}\s+{stop}",
            re.MULTILINE | re.IGNORECASE,
        )
        stop_match = stop_pattern.search(text, start_match.end())
        if stop_match:
            return text[start_match.start() : stop_match.start()]

    return text[start_match.start() :]


def _parse_strategic_objective(block: str, nlp: Language) -> dict[str, Any]:
    """Parse a single strategic-objective markdown block into a dictionary.

    The block is expected to contain the subsections **Goal Statement**,
    **Strategic Goals** (numbered list), **KPIs** (table),
    **Timeline** (table), **Responsible Stakeholders** (bullet list),
    and **Risks & Mitigation** (table).

    Args:
        block: The markdown text for one strategic objective (e.g.
               everything under ``### Strategic Objective A: …``).
        nlp:   A loaded spaCy pipeline for NLP enrichment.

    Returns:
        A dictionary with keys ``code``, ``name``, ``goal_statement``,
        ``strategic_goals``, ``kpis``, ``timeline``, ``stakeholders``,
        ``risks_and_mitigation``, ``entities``, and ``keywords``.
    """
    # --- Header -----------------------------------------------------------
    header_match = re.search(
        r"Strategic Objective\s+([A-E]):\s*(.+)", block
    )
    code = header_match.group(1) if header_match else ""
    name = header_match.group(2).strip() if header_match else ""

    # --- Goal statement ---------------------------------------------------
    goal_match = re.search(
        r"\*\*Goal Statement:\*\*\s*(.+?)(?:\n\n|\n####)", block, re.DOTALL
    )
    goal_statement = goal_match.group(1).strip() if goal_match else ""

    # --- Strategic goals --------------------------------------------------
    goals: list[dict[str, str]] = []
    goal_pattern = re.compile(
        r"\d+\.\s+\*\*([A-E]\d+)\*\*\s*—\s*(.+?)(?=\n\d+\.|\n\n|$)",
        re.DOTALL,
    )
    for m in goal_pattern.finditer(block):
        goals.append({"id": m.group(1), "description": m.group(2).strip()})

    # --- KPIs (table) -----------------------------------------------------
    kpi_section = _extract_section(
        block, r"KPIs", [r"Timeline", r"Responsible"]
    )
    kpis = _parse_markdown_table(kpi_section) if kpi_section else []

    # --- Timeline (table) -------------------------------------------------
    timeline_section = _extract_section(
        block, r"Timeline", [r"Responsible", r"Risks"]
    )
    timeline = _parse_markdown_table(timeline_section) if timeline_section else []

    # --- Stakeholders (bullet list) ---------------------------------------
    stakeholder_section = _extract_section(
        block, r"Responsible Stakeholders", [r"Risks"]
    )
    stakeholders: list[dict[str, str]] = []
    for sm in re.finditer(
        r"-\s+\*\*(.+?)\*\*\s*—\s*(.+)", stakeholder_section
    ):
        stakeholders.append(
            {"role": sm.group(1).strip(), "responsibility": sm.group(2).strip()}
        )

    # --- Risks & Mitigation (table) ---------------------------------------
    risk_section = _extract_section(block, r"Risks", [r"---", r"###"])
    risks = _parse_markdown_table(risk_section) if risk_section else []

    # --- NLP enrichment ---------------------------------------------------
    full_text = f"{goal_statement} " + " ".join(
        g["description"] for g in goals
    )
    entities = _extract_entities(nlp, full_text)
    keywords = _extract_keywords(nlp, full_text)

    return {
        "code": code,
        "name": name,
        "goal_statement": goal_statement,
        "strategic_goals": goals,
        "kpis": kpis,
        "timeline": timeline,
        "stakeholders": stakeholders,
        "risks_and_mitigation": risks,
        "entities": entities,
        "keywords": keywords,
    }


def _parse_swot(text: str) -> dict[str, list[str]]:
    """Extract the SWOT analysis from the strategic plan markdown.

    The function looks for the two side-by-side SWOT tables (Strengths /
    Weaknesses and Opportunities / Threats) and returns their cell values.

    Args:
        text: The full strategic plan markdown text.

    Returns:
        A dictionary with keys ``strengths``, ``weaknesses``,
        ``opportunities``, ``threats``, each holding a list of strings.
    """
    swot_section = _extract_section(text, r"2\.3 SWOT", [r"---", r"##\s+3"])
    swot: dict[str, list[str]] = {
        "strengths": [],
        "weaknesses": [],
        "opportunities": [],
        "threats": [],
    }
    if not swot_section:
        return swot

    tables = re.split(r"\n\n+", swot_section)
    for table_text in tables:
        rows = _parse_markdown_table(table_text)
        for row in rows:
            for key, col_header in [
                ("strengths", "Strengths"),
                ("weaknesses", "Weaknesses"),
                ("opportunities", "Opportunities"),
                ("threats", "Threats"),
            ]:
                val = row.get(col_header, "").strip()
                if val:
                    swot[key].append(val)
    return swot


def _parse_financial_targets(text: str) -> dict[str, Any]:
    """Extract Section 6 financial tables from the strategic plan.

    Parses revenue projections, capital investment plan, ROI estimates,
    and service diversification revenue streams.

    Args:
        text: The full strategic plan markdown text.

    Returns:
        A dictionary with keys ``revenue_projections``,
        ``capital_investment``, ``roi_estimates``, and
        ``revenue_streams``, each containing parsed table rows.
    """
    fin_section = _extract_section(
        text, r"6\.\s*Financial", [r"---\s*\n\s*\*\*— End"]
    )
    if not fin_section:
        fin_section = _extract_section(text, r"6\.\s*Financial", [])

    rev_section = _extract_section(
        fin_section, r"6\.1", [r"6\.2", r"6\.3", r"6\.4", r"6\.5"]
    )
    cap_section = _extract_section(
        fin_section, r"6\.2", [r"6\.3", r"6\.4", r"6\.5"]
    )
    roi_section = _extract_section(
        fin_section, r"6\.3", [r"6\.4", r"6\.5"]
    )
    stream_section = _extract_section(
        fin_section, r"6\.4", [r"6\.5"]
    )

    return {
        "revenue_projections": _parse_markdown_table(rev_section),
        "capital_investment": _parse_markdown_table(cap_section),
        "roi_estimates": _parse_markdown_table(roi_section),
        "revenue_streams": _parse_markdown_table(stream_section),
    }


# ---------------------------------------------------------------------------
# Action plan parser
# ---------------------------------------------------------------------------

def _parse_single_action(block: str, nlp: Language) -> dict[str, Any]:
    """Parse one action item block from the action plan markdown.

    Each action block is expected to follow the template::

        #### Action N: <Title>

        **Strategic Objective:** <letter> — <name>
        **Description:** …
        **Action Owner:** …
        **Timeline:** …
        **Budget Allocation:** …
        **Expected Outcome:** …
        **KPIs:**
        - …

    Args:
        block: The markdown text for a single action item.
        nlp:   A loaded spaCy pipeline for NLP enrichment.

    Returns:
        A dictionary containing all parsed fields plus spaCy-derived
        ``entities`` and ``keywords``.
    """

    def _field(label: str) -> str:
        """Extract a bold-labelled field value from the block."""
        pattern = re.compile(
            rf"\*\*{label}:\*\*\s*(.+?)(?=\n\*\*|\n####|\Z)",
            re.DOTALL,
        )
        m = pattern.search(block)
        return m.group(1).strip() if m else ""

    # --- Title & number ---------------------------------------------------
    title_match = re.search(r"####\s*Action\s+(\d+):\s*(.+)", block)
    action_number = int(title_match.group(1)) if title_match else 0
    action_title = title_match.group(2).strip() if title_match else ""

    # --- Labelled fields --------------------------------------------------
    strategic_obj_raw = _field("Strategic Objective")
    description = _field("Description")
    action_owner = _field("Action Owner")
    timeline = _field("Timeline")
    budget_raw = _field("Budget Allocation")
    expected_outcome = _field("Expected Outcome")

    # --- Strategic objective code -----------------------------------------
    obj_match = re.match(r"([A-E])\s*—", strategic_obj_raw)
    objective_code = obj_match.group(1) if obj_match else ""
    objective_name = OBJECTIVE_CODES.get(objective_code, strategic_obj_raw)

    # --- Budget numeric ---------------------------------------------------
    budget_match = re.search(r"LKR\s*([\d,.]+)\s*M", budget_raw)
    budget_lkr_millions = (
        float(budget_match.group(1).replace(",", ""))
        if budget_match
        else 0.0
    )

    # --- KPIs (bullet list) -----------------------------------------------
    kpi_section_match = re.search(
        r"\*\*KPIs?:\*\*\s*\n((?:\s*-\s*.+\n?)+)", block
    )
    kpis: list[str] = []
    if kpi_section_match:
        for line in kpi_section_match.group(1).strip().splitlines():
            cleaned = re.sub(r"^\s*-\s*", "", line).strip()
            if cleaned and not re.match(r"^-+$", cleaned):
                kpis.append(cleaned)

    # --- Timeline quarters ------------------------------------------------
    quarter_matches = re.findall(r"Q([1-4])", timeline)
    quarters = sorted(set(f"Q{q}" for q in quarter_matches))

    # --- NLP enrichment ---------------------------------------------------
    combined_text = f"{action_title}. {description}"
    entities = _extract_entities(nlp, combined_text)
    keywords = _extract_keywords(nlp, combined_text)

    return {
        "action_number": action_number,
        "title": action_title,
        "strategic_objective_code": objective_code,
        "strategic_objective_name": objective_name,
        "description": description,
        "action_owner": action_owner,
        "timeline": timeline,
        "quarters": quarters,
        "budget_lkr_millions": budget_lkr_millions,
        "budget_raw": budget_raw,
        "expected_outcome": expected_outcome,
        "kpis": kpis,
        "entities": entities,
        "keywords": keywords,
    }


# ---------------------------------------------------------------------------
# Main processor class
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """Loads, parses, and structures the hospital strategic and action plans.

    This class is the primary entry point for the document processing
    pipeline.  It reads the markdown source files, extracts structured
    data using regex patterns and spaCy NLP, and writes the results as
    JSON files for consumption by the embedding, alignment, and
    knowledge-graph modules.

    Attributes:
        data_dir:            Path to the directory containing the markdown
                             source files.
        strategic_plan_path: Path to ``strategic_plan.md``.
        action_plan_path:    Path to ``action_plan.md``.
        nlp:                 The loaded spaCy ``Language`` pipeline.

    Example::

        processor = DocumentProcessor()
        strategic_data = processor.process_strategic_plan()
        action_data    = processor.process_action_plan()
        processor.save_all()
    """

    def __init__(
        self,
        data_dir: Path | str = DATA_DIR,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        """Initialise the processor with paths and a spaCy model.

        Args:
            data_dir:     Directory containing ``strategic_plan.md`` and
                          ``action_plan.md``.
            spacy_model:  spaCy model name to load for NLP processing.

        Raises:
            FileNotFoundError: If either markdown file is missing from
                               *data_dir*.
        """
        self.data_dir = Path(data_dir)
        self.strategic_plan_path = self.data_dir / "strategic_plan.md"
        self.action_plan_path = self.data_dir / "action_plan.md"

        for path in (self.strategic_plan_path, self.action_plan_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {path}"
                )

        self.nlp = _load_spacy_model(spacy_model)
        logger.info("DocumentProcessor initialised (data_dir=%s).", self.data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_strategic_plan(self) -> dict[str, Any]:
        """Parse the strategic plan markdown into a structured dictionary.

        The returned dictionary contains the following top-level keys:

        - ``metadata`` – document title, period, version.
        - ``vision`` / ``mission`` – institutional statements.
        - ``strategic_direction`` – the three transformative commitments.
        - ``swot`` – SWOT analysis items.
        - ``objectives`` – a list of five parsed strategic-objective
          dictionaries (see :func:`_parse_strategic_objective`).
        - ``financial_targets`` – revenue, investment, ROI, and
          diversification tables.

        Returns:
            The complete structured representation of the strategic plan.
        """
        logger.info("Processing strategic plan: %s", self.strategic_plan_path)
        text = self.strategic_plan_path.read_text(encoding="utf-8")

        # --- Metadata -----------------------------------------------------
        metadata = {
            "title": "Nawaloka Hospital Negombo — Strategic Plan 2026–2030",
            "period": "2026–2030",
            "version": "1.0",
            "source_file": self.strategic_plan_path.name,
        }

        # --- Vision & Mission ---------------------------------------------
        vision_match = re.search(
            r"### Vision\s+\*(.+?)\*", text, re.DOTALL
        )
        vision = vision_match.group(1).strip() if vision_match else ""

        mission_match = re.search(
            r"### Mission\s+\*(.+?)\*", text, re.DOTALL
        )
        mission = mission_match.group(1).strip() if mission_match else ""

        # --- Strategic direction ------------------------------------------
        direction_items: list[str] = []
        for m in re.finditer(
            r"\d+\.\s+\*\*(.+?)\*\*\s*—\s*(.+?)(?=\n\d+\.|\n\n)", text
        ):
            direction_items.append(f"{m.group(1)}: {m.group(2).strip()}")

        # --- SWOT ---------------------------------------------------------
        swot = _parse_swot(text)

        # --- Objectives ---------------------------------------------------
        objective_blocks = re.split(
            r"(?=### Strategic Objective [A-E]:)", text
        )
        objectives: list[dict[str, Any]] = []
        for ob in objective_blocks:
            if re.match(r"### Strategic Objective [A-E]:", ob.strip()):
                parsed = _parse_strategic_objective(ob, self.nlp)
                objectives.append(parsed)
                logger.info(
                    "  Parsed objective %s: %s (%d goals, %d KPIs)",
                    parsed["code"],
                    parsed["name"],
                    len(parsed["strategic_goals"]),
                    len(parsed["kpis"]),
                )

        # --- Financials ---------------------------------------------------
        financial_targets = _parse_financial_targets(text)

        result = {
            "metadata": metadata,
            "vision": vision,
            "mission": mission,
            "strategic_direction": direction_items[:3],
            "swot": swot,
            "objectives": objectives,
            "financial_targets": financial_targets,
        }
        logger.info(
            "Strategic plan processed: %d objectives extracted.",
            len(objectives),
        )
        return result

    def process_action_plan(self) -> dict[str, Any]:
        """Parse the action plan markdown into a structured dictionary.

        The returned dictionary contains:

        - ``metadata`` – document title, period, version.
        - ``actions`` – a list of parsed action-item dictionaries
          (see :func:`_parse_single_action`).
        - ``summary_budget`` – the summary budget table rows.
        - ``budget_by_objective`` – budget distribution by strategic
          objective.

        Returns:
            The complete structured representation of the action plan.
        """
        logger.info("Processing action plan: %s", self.action_plan_path)
        text = self.action_plan_path.read_text(encoding="utf-8")

        # --- Metadata -----------------------------------------------------
        metadata = {
            "title": "Nawaloka Hospital Negombo — Annual Action Plan 2025",
            "period": "2025",
            "version": "1.0",
            "source_file": self.action_plan_path.name,
        }

        # --- Actions ------------------------------------------------------
        action_blocks = re.split(r"(?=#### Action \d+:)", text)
        actions: list[dict[str, Any]] = []
        for ab in action_blocks:
            if re.match(r"#### Action \d+:", ab.strip()):
                parsed = _parse_single_action(ab, self.nlp)
                actions.append(parsed)
                logger.info(
                    "  Parsed action %d: %s [Obj %s] (%.0fM LKR)",
                    parsed["action_number"],
                    parsed["title"],
                    parsed["strategic_objective_code"],
                    parsed["budget_lkr_millions"],
                )

        # --- Summary budget table -----------------------------------------
        budget_section = _extract_section(
            text, r"3\.\s*Summary Budget", [r"4\.\s*Budget Distribution", r"---"]
        )
        summary_budget = (
            _parse_markdown_table(budget_section) if budget_section else []
        )

        # --- Budget by objective ------------------------------------------
        obj_budget_section = _extract_section(
            text,
            r"4\.\s*Budget Distribution",
            [r"5\.\s*Quarterly", r"---"],
        )
        budget_by_objective = (
            _parse_markdown_table(obj_budget_section)
            if obj_budget_section
            else []
        )

        result = {
            "metadata": metadata,
            "actions": actions,
            "summary_budget": summary_budget,
            "budget_by_objective": budget_by_objective,
        }
        logger.info(
            "Action plan processed: %d actions extracted.", len(actions)
        )
        return result

    def save_json(
        self, data: dict[str, Any], output_path: Path | str
    ) -> Path:
        """Serialise a dictionary to a pretty-printed JSON file.

        Args:
            data:        The dictionary to serialise.
            output_path: Destination file path.

        Returns:
            The resolved ``Path`` of the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.info("Saved JSON: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
        return output_path

    def save_all(self) -> tuple[Path, Path]:
        """Run both parsers and write their outputs as JSON files.

        Convenience method that calls :meth:`process_strategic_plan` and
        :meth:`process_action_plan`, then writes the results to
        ``data/strategic_plan.json`` and ``data/action_plan.json``.

        Returns:
            A tuple of ``(strategic_json_path, action_json_path)``.
        """
        strategic = self.process_strategic_plan()
        action = self.process_action_plan()

        sp = self.save_json(strategic, STRATEGIC_JSON_OUT)
        ap = self.save_json(action, ACTION_JSON_OUT)

        logger.info("All documents processed and saved.")
        return sp, ap


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the document processor from the command line.

    Loads both plan documents, parses them, prints a summary to stdout,
    and writes JSON output files to the ``data/`` directory.
    """
    logger.info("=" * 60)
    logger.info("ISPS Document Processor — Starting")
    logger.info("=" * 60)

    try:
        processor = DocumentProcessor()
    except FileNotFoundError as exc:
        logger.error("Initialisation failed: %s", exc)
        raise SystemExit(1) from exc

    # --- Strategic plan ---------------------------------------------------
    strategic = processor.process_strategic_plan()
    print("\n" + "=" * 60)
    print("STRATEGIC PLAN SUMMARY")
    print("=" * 60)
    print(f"Vision : {strategic['vision'][:80]}…")
    print(f"Mission: {strategic['mission'][:80]}…")
    print(f"\nObjectives extracted: {len(strategic['objectives'])}")
    for obj in strategic["objectives"]:
        print(f"  [{obj['code']}] {obj['name']}")
        print(f"      Goals: {len(obj['strategic_goals'])}  |  "
              f"KPIs: {len(obj['kpis'])}  |  "
              f"Keywords: {', '.join(obj['keywords'][:5])}")

    # --- Action plan ------------------------------------------------------
    action = processor.process_action_plan()
    print("\n" + "=" * 60)
    print("ACTION PLAN SUMMARY")
    print("=" * 60)
    print(f"Actions extracted: {len(action['actions'])}")
    total_budget = sum(a["budget_lkr_millions"] for a in action["actions"])
    print(f"Total action budget: LKR {total_budget:.0f}M\n")
    for act in action["actions"]:
        print(
            f"  [{act['action_number']:>2}] {act['title'][:50]:<52} "
            f"Obj={act['strategic_objective_code']}  "
            f"Budget={act['budget_lkr_millions']:>5.0f}M  "
            f"Q={','.join(act['quarters'])}"
        )

    # --- Save -------------------------------------------------------------
    processor.save_json(strategic, STRATEGIC_JSON_OUT)
    processor.save_json(action, ACTION_JSON_OUT)

    print(f"\nJSON outputs saved to:")
    print(f"  {STRATEGIC_JSON_OUT}")
    print(f"  {ACTION_JSON_OUT}")


if __name__ == "__main__":
    main()
