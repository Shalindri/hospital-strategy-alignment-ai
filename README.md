# Hospital Strategy–Action Plan Alignment System (ISPS)

An AI-powered system that evaluates the alignment between a hospital's strategic plan and its annual action plan using semantic embeddings, ontology mapping, knowledge graphs, RAG (Retrieval-Augmented Generation), and agentic AI reasoning.

## Problem Statement

Hospitals produce multi-year strategic plans and annual action plans, but ensuring operational actions genuinely support strategic objectives is a manual, subjective process. This system automates alignment detection, identifies misaligned (orphan) actions, and provides explainable scoring — enabling hospital leadership to make evidence-based planning decisions.

## Case Study

**Nawaloka Hospital Negombo, Sri Lanka**
- Strategic Plan: 2026–2030 (5 strategic objectives, capacity expansion from 75 to 150 beds)
- Action Plan: 2025 (25 operational actions across the five objectives)
- Includes intentionally misaligned actions for evaluation of detection precision/recall
- 58-pair human-annotated ground truth dataset for system evaluation

The system is **domain-agnostic** — any hospital or organisation can upload their strategic and action plan PDFs through the dashboard for analysis.

## Architecture

```
┌──────────────┐    ┌──────────────┐
│  Strategic   │    │   Action     │
│  Plan (PDF)  │    │  Plan (PDF)  │
└──────┬───────┘    └──────┬───────┘
       │                   │
       └─────────┬─────────┘
                 ▼
       ┌─────────────────┐
       │  PDF Processor   │  LLM-based extraction (Ollama llama3.1:8b)
       │  → Structured    │
       │    JSON           │
       └────────┬─────────┘
                │
    ┌───────────┼───────────────────────────────┐
    │           │                               │
    ▼           ▼                               ▼
┌────────┐ ┌──────────┐                  ┌────────────┐
│ChromaDB│ │Alignment │                  │  Ontology  │
│Vector  │ │Scoring   │                  │  Mapper    │
│Store   │ │(Cosine)  │                  │  (RDF/OWL) │
└───┬────┘ └────┬─────┘                  └─────┬──────┘
    │           │                               │
    │     ┌─────┴──────┐               ┌────────┴────────┐
    │     │ Knowledge  │               │  Gap Detection  │
    │     │ Graph      │               │                 │
    │     │ (NetworkX) │               └─────────────────┘
    │     └─────┬──────┘
    │           │
    ├───────────┤
    ▼           ▼
┌────────┐ ┌──────────┐
│  RAG   │ │  Agent   │
│Engine  │ │ Reasoner │
│(LLM)  │ │ (LLM)   │
└───┬────┘ └────┬─────┘
    │           │
    └─────┬─────┘
          ▼
  ┌───────────────┐
  │   Streamlit   │
  │   Dashboard   │
  └───────────────┘
```

## 6-Stage Pipeline

| Stage | Module | Description | LLM Required |
|-------|--------|-------------|:------------:|
| 1. Extraction | `pdf_processor.py` | PDF → structured JSON via Ollama LLM | Yes |
| 2. Alignment | `synchronization_analyzer.py` | Cosine similarity matrix (5×25) | No |
| 3. Ontology | `ontology_mapper.py` | RDF/OWL mapping with hybrid scoring | No |
| 4. Knowledge Graph | `knowledge_graph.py` | NetworkX graph with centrality analysis | No |
| 5. RAG | `rag_engine.py` | Context-aware improvement suggestions | Yes |
| 6. Agent | `agent_reasoner.py` | Plan-Act-Reflect diagnostic reasoning | Yes |

## Project Structure

```
hospital-strategy-alignment-ai/
├── src/                       # Core pipeline modules
│   ├── config.py              # Shared configuration constants
│   ├── pdf_processor.py       # PDF extraction with Ollama LLM
│   ├── vector_store.py        # ChromaDB embeddings (all-MiniLM-L6-v2)
│   ├── synchronization_analyzer.py  # Alignment scoring engine
│   ├── dynamic_analyzer.py    # Dynamic analysis for uploaded PDFs
│   ├── ontology_mapper.py     # RDF/OWL ontology mapping
│   ├── knowledge_graph.py     # NetworkX knowledge graph
│   ├── rag_engine.py          # RAG recommendation engine
│   └── agent_reasoner.py      # Agentic AI reasoning (Plan-Act-Reflect)
├── dashboard/                 # Streamlit UI
│   ├── app.py                 # Main dashboard application
│   ├── pipeline_runner.py     # Dynamic pipeline orchestration
│   ├── data_adapter.py        # Data format conversion
│   └── utils.py               # PDF report, charts, exports
├── data/                      # Processed data (JSON)
│   ├── strategic_plan.json
│   ├── action_plan.json
│   └── alignment_report.json
├── tests/                     # Evaluation & ground truth
│   ├── evaluation.py          # P/R/F1/AUC against baselines
│   ├── evaluate_suggestions.py # Suggestion quality metrics
│   ├── create_ground_truth.py # Ground truth labelling tool
│   └── ground_truth.json      # 58-pair human-annotated dataset
├── experiments/               # Parameter tuning notebooks
│   └── parameter_tuning.ipynb
├── outputs/                   # Generated artefacts (ontology, KG, etc.)
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM Framework | LangChain + Ollama |
| Local LLM | Ollama (llama3.1:8b) |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2, 384-dim) |
| Vector Store | ChromaDB (cosine similarity, HNSW indexing) |
| Ontology | RDFLib (Turtle/OWL export) |
| Knowledge Graph | NetworkX (centrality, community detection) |
| Dashboard | Streamlit + Plotly |
| Evaluation | scikit-learn, SciPy |
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

# Pull Ollama model
ollama pull llama3.1:8b
```

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

### Running Evaluation

```bash
python -m tests.evaluation
```

### Parameter Tuning Experiments

```bash
cd experiments
jupyter notebook parameter_tuning.ipynb
```

## Alignment Scoring

The system uses four classification thresholds (configurable in `src/config.py`):

| Range | Classification | Interpretation |
|-------|---------------|----------------|
| >= 0.75 | Excellent | Near-direct operationalisation of strategy |
| 0.60–0.74 | Good | Clear strategic support |
| 0.45–0.59 | Fair | Partial or indirect alignment |
| < 0.45 | Poor / Orphan | Weak or no meaningful alignment |

## Evaluation

The system is evaluated against a **58-pair human-annotated ground truth** dataset using:

- **Classification**: Precision, Recall, F1, Accuracy
- **Regression**: MAE, RMSE, Pearson r, Spearman rho
- **Ranking**: ROC curve, AUC
- **Baselines**: TF-IDF cosine similarity, keyword overlap
- **Statistical**: Paired t-tests for significance

## License

This project is developed as part of an MSc Information Retrieval programme. For academic use only.
