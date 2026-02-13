# Hospital Strategy–Action Plan Alignment System (ISPS)

An AI-powered system that evaluates the alignment between a hospital's strategic plan and its annual action plan using RAG (Retrieval-Augmented Generation), semantic embeddings, and knowledge graph techniques.

## Problem Statement

Hospitals produce multi-year strategic plans and annual action plans, but ensuring operational actions genuinely support strategic objectives is a manual, subjective process. This system automates alignment detection, identifies misaligned actions, and provides explainable scoring — enabling hospital leadership to make evidence-based planning decisions.

## Case Study

**Nawaloka Hospital Negombo, Sri Lanka**
- Strategic Plan: 2026–2030 (5 strategic objectives, capacity expansion from 75 to 150 beds)
- Action Plan: 2025 (25 operational actions across the five objectives)
- Includes intentionally misaligned actions for evaluation of detection precision/recall

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Strategic   │    │   Action     │    │   Embedding &    │
│  Plan (MD)   │───>│  Plan (MD)   │───>│   Vector Store   │
└──────────────┘    └──────────────┘    │   (ChromaDB)     │
                                        └────────┬─────────┘
                                                 │
                              ┌──────────────────┼──────────────────┐
                              │                  │                  │
                      ┌───────▼──────┐  ┌────────▼───────┐ ┌───────▼───────┐
                      │  Semantic    │  │  Knowledge     │ │  LLM-based    │
                      │  Similarity  │  │  Graph (RDF)   │ │  Reasoning    │
                      │  Scoring     │  │  Analysis      │ │  (Ollama)     │
                      └───────┬──────┘  └────────┬───────┘ └───────┬───────┘
                              │                  │                  │
                              └──────────────────┼──────────────────┘
                                                 │
                                        ┌────────▼─────────┐
                                        │   Alignment      │
                                        │   Dashboard      │
                                        │   (Streamlit)    │
                                        └──────────────────┘
```

## Project Structure

```
hospital-strategy-alignment-ai/
├── data/                  # Strategic plan & action plan documents
│   ├── strategic_plan.md
│   └── action_plan.md
├── src/                   # Core source code
│   ├── __init__.py
│   ├── parser.py          # Document parsing & chunking
│   ├── embeddings.py      # Embedding generation & vector store
│   ├── alignment.py       # Alignment scoring engine
│   ├── knowledge_graph.py # RDF/ontology-based analysis
│   └── llm_chain.py       # LLM reasoning chains (Ollama)
├── models/                # Persisted embeddings & vector DB
├── dashboard/             # Streamlit UI
│   └── app.py
├── tests/                 # Evaluation & test scripts
│   ├── __init__.py
│   └── evaluate.py        # Precision/recall evaluation
├── docs/                  # Report materials & documentation
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM Framework | LangChain |
| Local LLM | Ollama (Llama 3 / Mistral) |
| Embeddings | Sentence-Transformers |
| Vector Store | ChromaDB |
| Knowledge Graph | RDFLib + NetworkX |
| NLP | spaCy |
| Dashboard | Streamlit + Plotly |
| Language | Python 3.10+ |

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd hospital-strategy-alignment-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Pull Ollama model
ollama pull llama3
```

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

## Strategic Objectives Evaluated

| Code | Objective |
|---|---|
| A | Patient Care Excellence |
| B | Digital Health Transformation |
| C | Research & Innovation |
| D | Workforce Development |
| E | Community & Regional Health Expansion |

## Evaluation Methodology

The system evaluates alignment using three complementary approaches:

1. **Semantic Similarity** — Embedding-based cosine similarity between action descriptions and strategic objective text
2. **Knowledge Graph Analysis** — RDF ontology mapping of actions to strategic goals, KPIs, and stakeholders
3. **LLM Reasoning** — Chain-of-thought evaluation using a local LLM to assess logical alignment and provide explanations

Results are aggregated into a composite alignment score (0–1) per action, with classification into:
- **Strong alignment** (≥0.7)
- **Moderate alignment** (0.4–0.7)
- **Weak/misaligned** (<0.4)

## License

This project is developed as part of an MSc Information Retrieval programme. For academic use only.
