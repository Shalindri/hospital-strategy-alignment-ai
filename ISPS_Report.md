# Intelligent Strategic Plan Synchronization System (ISPS)

## Nawaloka Hospital Negombo — Strategy vs. Action Plan Alignment Analysis

**Module:** MSc in Computer Science — Information Retrieval [Individual]
**Word Count:** ~5,500 (excluding code snippets and screenshots)

---

## 1. Introduction and Background

Modern healthcare institutions rely on multi-year strategic plans and corresponding annual action plans to translate vision into operational reality. However, assessing the degree of synchronization between these documents remains a predominantly manual, subjective exercise — one that is both time-consuming and prone to inconsistency. As hospitals grow in complexity, with dozens of strategic objectives spanning clinical care, digital transformation, workforce development, and infrastructure expansion, the cognitive burden on leadership teams increases proportionally. A strategic objective that lacks supporting operational actions effectively becomes an aspirational statement with no execution pathway, while an action item that cannot be convincingly linked to any strategic goal represents misallocated resources.

This coursework presents the Intelligent Strategic Plan Synchronization System (ISPS), a smart AI-based system designed to analyse the synchronization between a hospital's Strategic Plan and its operational Action Plan. The system was developed for Nawaloka Hospital Negombo, a private healthcare institution in Sri Lanka, using its five-year Strategic Plan (2026–2030) and Annual Action Plan (2025) as the domain-specific input data. The hospital's strategic framework encompasses five strategic objectives — Patient Care Excellence, Digital Health Transformation, Research and Innovation, Workforce Development, and Community and Regional Health Expansion — supported by 25 discrete action items.

ISPS integrates several advanced information retrieval and natural language processing technologies, including sentence-transformer embeddings, vector databases (ChromaDB), ontology-based concept mapping (RDFLib), knowledge graph construction and analysis (NetworkX), Retrieval-Augmented Generation (RAG), and an agentic AI reasoning layer. The system provides a Streamlit-based interactive dashboard that displays synchronization metrics, identifies alignment gaps, generates intelligent improvement suggestions, and offers visual analytics to aid decision-making.

The domain of healthcare was chosen because hospital strategic plans exhibit rich, hierarchical, multi-disciplinary content that exercises the full breadth of information retrieval techniques — from semantic similarity detection across clinical and operational language, to ontology-based reasoning about relationships between specialised medical services and administrative functions.

---

## 2. Literature Review

### 2.1 Information Retrieval and NLP in Strategic Analysis

Information Retrieval (IR) is fundamentally concerned with finding material that satisfies an information need from within large collections (Manning, Raghavan and Schutze, 2008). Traditional IR approaches, such as TF-IDF vectorisation and BM25 scoring, rely on lexical overlap between queries and documents. While effective for keyword-based search, these methods fail to capture semantic meaning — a critical limitation when comparing strategic plans (which use abstract, goal-oriented language) against action plans (which use concrete, operational language). For instance, a strategic objective mentioning "patient care excellence" and an action item describing "cardiac catheterisation laboratory procurement" share no common terms yet are strongly related. This semantic gap motivates the use of dense retrieval methods based on neural language models (Karpukhin et al., 2020).

### 2.2 Sentence Transformers and Semantic Embeddings

Sentence-Transformers (Reimers and Gurevych, 2019) produce fixed-dimensional dense vector representations of text passages that capture semantic meaning. The all-MiniLM-L6-v2 model, a distilled variant of MiniLM (Wang et al., 2020), produces 384-dimensional embeddings and is trained on over one billion sentence pairs for semantic similarity tasks. It achieves approximately 14,000 sentences per second on CPU, making it suitable for interactive applications without GPU infrastructure. Cosine similarity between these embeddings provides a principled measure of semantic relatedness that is robust to differences in document length, as vectors are L2-normalised before comparison.

### 2.3 Vector Databases

Vector databases specialise in storing, indexing, and querying high-dimensional embeddings for approximate nearest-neighbour (ANN) search. ChromaDB, the vector database used in ISPS, employs the HNSW (Hierarchical Navigable Small World) algorithm (Malkov and Yashunin, 2020) for sub-linear query time, making it suitable for interactive dashboard queries. Alternatives such as Pinecone, FAISS (Johnson, Douze and Jegou, 2019), and Weaviate offer similar capabilities with different trade-offs between managed hosting, GPU acceleration, and filtering capabilities. ChromaDB was selected for its lightweight, file-based persistence model that enables local deployment without external infrastructure dependencies — an important consideration for data-sensitive healthcare environments.

### 2.4 Ontologies and Knowledge Representation

Ontologies provide formal, explicit specifications of shared conceptualisations (Gruber, 1993). In the healthcare domain, ontologies have been widely used for clinical coding (SNOMED CT), drug interactions (DrugBank), and administrative classification (ICD-10). ISPS employs a custom lightweight ontology expressed in RDF/OWL using the RDFLib library, defining 10 top-level healthcare strategy concepts (such as PatientCare, DigitalHealth, WorkforceHR, and ClinicalServices) with approximately 40 mid-level sub-concepts. This ontology serves as a bridge between strategic language and operational language, enabling concept-level gap detection that pure embedding similarity cannot achieve.

### 2.5 Knowledge Graphs

Knowledge graphs represent entities and their relationships as directed, labelled graphs (Hogan et al., 2021). NetworkX provides the graph infrastructure for ISPS, enabling structural analysis through centrality measures (degree, betweenness), community detection (greedy modularity), and path analysis. The knowledge graph integrates data from all upstream pipeline stages — parsed documents, alignment scores, and ontology mappings — into a unified relational structure with seven distinct node types and weighted, typed edges.

### 2.6 Large Language Models and Retrieval-Augmented Generation

Large Language Models (LLMs) such as GPT-4 and LLaMA (Touvron et al., 2023) demonstrate impressive text generation capabilities but are prone to hallucination — generating plausible but factually incorrect content. Retrieval-Augmented Generation (RAG) mitigates this by grounding LLM responses in retrieved evidence from a knowledge base (Lewis et al., 2020). ISPS implements RAG using Ollama with the llama3.1:8b model, retrieving relevant strategic context from ChromaDB before prompting the LLM for improvement recommendations. This architecture ensures that generated suggestions are grounded in the hospital's actual strategic data rather than generic healthcare knowledge.

### 2.7 Agentic AI

Agentic AI systems extend LLM capabilities by introducing structured reasoning loops with tool use, self-critique, and iterative refinement (Yao et al., 2023). ISPS implements a Plan-Act-Reflect agent that diagnoses synchronization issues, gathers evidence using four typed tools (vector search, ontology lookup, impact calculation, alignment validation), generates recommendations, self-critiques proposals for feasibility and measurability, and refines them based on critique feedback. This multi-step reasoning produces more grounded, actionable recommendations than single-pass LLM generation.

---

## 3. System Design and Architecture

### 3.1 High-Level Architecture

The ISPS system follows a six-stage pipeline architecture:

**Stage 1 — PDF Extraction:** Uploaded PDF documents are processed using pdfplumber for text extraction, followed by structured JSON parsing via the Ollama llama3.1:8b LLM. The LLM receives carefully engineered prompts that specify the exact JSON schema required, and a robust response parser handles common LLM output issues such as trailing commas, truncated JSON, and inconsistent key names.

**Stage 2 — Alignment Analysis:** Strategic objectives and action items are embedded using the all-MiniLM-L6-v2 sentence transformer into 384-dimensional vectors, stored in a temporary ChromaDB instance. A full pairwise cosine similarity matrix is computed via matrix multiplication (exploiting the fact that L2-normalised vectors yield cosine similarity through dot product), producing an N-objectives x M-actions alignment matrix.

**Stage 3 — Ontology Mapping:** A hybrid scoring approach combines keyword matching (40% weight) with embedding similarity (60% weight) to map each action and strategic goal to the custom healthcare ontology. A mapping threshold of 0.55 determines valid mappings, and gap detection identifies uncovered strategy concepts and weakly aligned actions.

**Stage 4 — Knowledge Graph Construction:** A directed, weighted graph is built using NetworkX, incorporating seven node types (StrategyObjective, StrategyGoal, OntologyConcept, Action, KPI, Stakeholder, TimelineQuarter) with typed edges from alignment scores, ontology mappings, and structural relationships. Graph analytics including degree centrality, betweenness centrality, community detection, bottleneck identification, and critical-path analysis are computed.

**Stage 5 — RAG Recommendations:** Poorly aligned actions (best score below 0.50) receive LLM-generated improvement recommendations, while under-covered objectives (coverage below 25%) receive new action suggestions. Each LLM call is preceded by semantic retrieval of relevant strategic context from ChromaDB, implementing the RAG pattern. Responses are cached using SHA-256 content hashing to avoid redundant inference.

**Stage 6 — Agent Reasoning:** The agentic layer diagnoses synchronization issues across four categories (orphan actions, under-supported strategies, conflicting mappings, KPI mismatches), prioritises them by a composite score of severity, confidence, and business impact, and processes the top issues through a Diagnose-Investigate-Reason-Critique-Refine loop with early stopping when marginal improvement falls below threshold.

### 3.2 Data Flow

The pipeline uses a temporary ChromaDB instance for each analysis session, ensuring that uploaded data does not persist beyond the session and that concurrent users receive isolated analysis environments. Each stage produces structured dictionaries that flow downstream through a data adapter layer, which normalises the output format for dashboard consumption.

### 3.3 Technology Stack

| Component | Technology | Justification |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 | 384-dim, 14K sent/sec on CPU, strong semantic similarity |
| Vector DB | ChromaDB (persistent) | Lightweight, local, HNSW indexing, cosine similarity |
| Ontology | RDFLib (RDF/OWL) | Standards-compliant, Turtle export, SPARQL-ready |
| Knowledge Graph | NetworkX | Rich graph algorithms, community detection, export formats |
| LLM | Ollama (llama3.1:8b) | Local deployment, no API costs, data privacy |
| LLM Framework | LangChain | Prompt templating, output parsing, chain composition |
| Dashboard | Streamlit | Rapid prototyping, native file upload, Plotly integration |
| PDF Extraction | pdfplumber | Multi-column, table-aware, embedded font support |
| Evaluation | scikit-learn, scipy | Precision/Recall/F1, ROC/AUC, statistical tests |

---

## 4. Methodology and Implementation

### 4.1 Input Data Preparation

The Strategic Plan for Nawaloka Hospital Negombo (2026–2030) contains five strategic objectives (coded A through E), each with 3–6 strategic goals, associated KPIs with baselines and targets, and implementation timelines. The Action Plan (2025) contains 25 discrete actions, each linked to a declared strategic objective and specifying an action owner, timeline, budget allocation, expected outcome, and operational KPIs.

Both documents are uploaded as PDFs through the Streamlit interface. The pdf_processor module extracts text using pdfplumber and sends it to the Ollama LLM with structured prompts that enforce the expected JSON schema. A robust JSON extraction pipeline handles markdown code fences, raw JSON detection, and even truncated JSON repair — accounting for the inherent non-determinism of LLM-generated structured output.

### 4.2 Synchronization Assessment (Overall)

The overall synchronization assessment is the core analytical contribution of ISPS. The process works as follows:

1. **Text Composition:** Each strategic objective is represented as a composite text combining its goal statement, individual strategic goal descriptions, and KPI names. Each action item is represented as a composite of its title, description, expected outcome, and KPI text. This composition captures the most semantically meaningful components for alignment matching.

2. **Embedding Generation:** The composite texts are encoded using the all-MiniLM-L6-v2 model with L2 normalisation enabled, producing 384-dimensional unit vectors.

3. **ChromaDB Storage:** Embeddings are stored in two ChromaDB collections — `strategic_objectives` and `action_items` — configured with cosine similarity as the distance metric.

4. **Similarity Matrix Computation:** All embeddings are retrieved from ChromaDB and assembled into numpy arrays. The full pairwise similarity matrix is computed as a single matrix multiplication (`obj_emb @ act_emb.T`), yielding an objectives-by-actions matrix where each cell contains the cosine similarity between one objective and one action.

5. **Score Classification:** Similarity scores are classified into four bands calibrated for the all-MiniLM-L6-v2 model on hospital-domain text: Excellent (>= 0.75), Good (0.60–0.74), Fair (0.45–0.59), and Poor (< 0.45).

6. **Overall Score:** The overall synchronization score is the mean of each action's best-objective similarity — representing how well each action connects to its closest strategic objective, on average.

### 4.3 Strategy-Wise Synchronization

Per-objective analysis computes mean and maximum similarity across all actions, identifies the top-5 aligned actions, counts actions scoring at or above the Fair threshold (0.45), and detects gap actions — actions declared under the objective but scoring below the threshold. Coverage score is calculated as the fraction of total actions scoring above Fair, providing a breadth-of-support metric for each objective.

### 4.4 Ontology-Based Concept Mapping

The ontology mapper implements a hybrid scoring approach that combines two complementary signals:

**Keyword Score (40% weight):** A curated dictionary of domain-specific keywords is maintained for each of the approximately 50 ontology concepts. The keyword score is the fraction of a concept's keywords found in the item text. This provides deterministic, explainable matching that captures exact domain terminology (e.g., "cardiac catheterisation", "haemodialysis", "JCI accreditation").

**Embedding Score (60% weight):** Each concept's description is embedded alongside each item's text, and cosine similarity is computed. This captures semantic relationships that keyword matching misses — for instance, linking an action about "purchasing dialysis equipment" to the "Nephrology" concept even when the exact keyword is absent.

The final score is `0.6 * embedding_score + 0.4 * keyword_score`, and mappings with a final score >= 0.55 are considered valid. Gap detection identifies strategy concepts with no supporting actions (uncovered concepts), actions with no valid concept mapping (weak actions), and actions mapped to multiple unrelated top-level domains (conflicting mappings).

The full ontology is serialised as an RDF/OWL graph in Turtle format, containing class hierarchies, object properties (supportsObjective, implementsGoal, relatedToConcept), and data properties (hasScore, hasBudgetValue), making it interoperable with standard semantic web tools and SPARQL queries.

### 4.5 Knowledge Graph Construction and Analysis

The knowledge graph integrates data from all upstream stages into a unified relational structure. Seven node types are defined with a consistent colour palette for visualisation:

- **StrategyObjective** (blue): Top-level objectives A–E
- **StrategyGoal** (light blue): Individual goals under each objective
- **OntologyConcept** (purple): Mid- and top-level ontology concepts
- **Action** (green): Operational actions 1–25
- **KPI** (yellow): Key performance indicators for both strategies and actions
- **Stakeholder** (orange): Action owners and responsible parties
- **TimelineQuarter** (grey): Q1–Q4 2025 scheduling nodes

Edges are typed and weighted: structural edges (has_goal, has_kpi, has_owner, has_timeline) carry unit weight, alignment edges carry cosine similarity scores (filtered above 0.45), and ontology-mapping edges carry the hybrid final score. Action node sizes are scaled by `alignment_score * log(budget + 1)`, encoding both strategic relevance and resource commitment. Concept node sizes are normalised by degree centrality.

Graph analytics provide structural insights: degree centrality identifies hub nodes, betweenness centrality reveals bridge nodes and bottlenecks, community detection via greedy modularity groups densely interconnected components, and critical-path analysis uses Dijkstra on inverse weights to find the highest-weight paths between strategy and KPI nodes. A novel "suggest new connections" feature identifies ontology concepts covered by strategy goals but lacking action edges, and recommends candidate actions based on alignment scores — providing evidence-based recommendations for closing coverage gaps.

### 4.6 RAG-Based Improvement Suggestions

The RAG engine operates in two modes:

**Improvement Recommendations:** For actions scoring below 0.50, the engine retrieves the top-3 most relevant strategic objectives from ChromaDB, constructs a detailed prompt incorporating the action's metadata, alignment analysis, and retrieved strategic context, and queries the LLM for structured improvements including a modified description, additional KPIs, timeline adjustments, resource reallocation suggestions, strategic linkage explanation, and confidence assessment.

**Gap-Filling Action Suggestions:** For objectives with coverage below 25%, the engine generates 2–3 new concrete actions with titles, descriptions, owners, timelines, budgets, and KPIs. The number of suggestions scales with gap severity (3 for coverage below 15%, 2 otherwise).

All LLM responses are parsed using regex-based extraction of structured sections and cached using SHA-256 content hashing with exponential-backoff retry (3 attempts at 1s, 2s, 4s delays).

### 4.7 Agentic AI Reasoning Layer

The agent reasoner follows a bounded Plan-Act-Reflect loop (maximum 5 iterations) with four typed tools:

1. **vector_search:** Semantic retrieval from ChromaDB across both strategic and action collections
2. **ontology_lookup:** Concept expansion from cached ontology mappings, including related concepts and keywords
3. **calculate_impact:** Rule-based budget/time/risk estimation using objective count, action count, and budget proportionality
4. **validate_alignment:** Real-time similarity scoring of proposed changes against target objectives

Each iteration: (1) selects the highest-priority unaddressed issue, (2) gathers evidence using tools, (3) generates a recommendation via LLM with full context, (4) self-critiques the recommendation for feasibility, alignment strength, KPI measurability, and risks, and (5) refines based on critique feedback. Early stopping triggers when marginal improvement falls below 0.05, preventing wasteful iterations on diminishing-returns issues.

A structured reasoning trace is exported as JSON, providing full transparency into the agent's diagnostic process, evidence gathering, reasoning chain, critique outcomes, and final decisions — supporting explainability requirements in healthcare decision support.

---

## 5. Smart Dashboard

The ISPS dashboard is a single-page Streamlit application with a clean, professional design using a blue-and-white healthcare-appropriate colour palette. The interface is structured as follows:

**Upload Toolbar:** Users upload Strategic Plan and Action Plan PDFs via inline file uploaders. A six-stage progress indicator shows real-time pipeline status as the analysis runs.

**Summary Metrics:** Four prominent metric cards display the overall synchronization score, classification, action distribution across bands, and key alerts (orphan count, mismatch count).

**Alignment Visualisations:** Multiple interactive Plotly charts including a heatmap of the full similarity matrix, a radar chart of per-objective mean similarity, bar charts of per-action best scores colour-coded by classification, and budget-vs-alignment scatter plots.

**Strategy-Wise Analysis:** Expandable sections for each objective showing top aligned actions, coverage metrics, gap actions, and ontology concept mappings with evidence keywords.

**Knowledge Graph:** An interactive network visualisation with colour-coded node types, degree-proportional sizing, and tooltips showing node metadata. Bottleneck analysis and connection suggestions are displayed alongside.

**RAG Recommendations:** Improvement suggestions are displayed with confidence badges, proposed KPIs, timeline adjustments, and strategic linkage explanations. New action suggestions for under-covered objectives are shown with full operational detail.

**Agent Reasoning Trace:** The agent's iteration-by-iteration trace is displayed showing diagnosed issues, evidence gathered, recommendations made, critique results, and final decisions — providing full transparency into the AI reasoning process.

**PDF Report Generation:** A downloadable PDF report summarising all analysis results is generated using the ReportLab library, providing an offline-compatible deliverable for stakeholders.

---

## 6. Hosting Architecture and Security

### 6.1 Architecture

ISPS is deployed as a Streamlit application with the following architecture considerations:

- **LLM Integration:** Ollama runs locally, keeping all data on-premise and avoiding transmission of sensitive hospital documents to external APIs. The llama3.1:8b model provides adequate performance for structured extraction and recommendation generation while being small enough for single-machine deployment.
- **Vector Database:** ChromaDB uses file-based persistent storage, eliminating the need for a separate database server. Temporary instances are created per analysis session and cleaned up upon completion.
- **Data Privacy:** No uploaded documents are persisted beyond the session. Temporary ChromaDB directories are explicitly removed in `finally` blocks. The system processes data entirely on the local machine.

### 6.2 Security Considerations

- **Data Protection:** All processing occurs locally; no data is transmitted to cloud APIs. The Ollama LLM runs on localhost (port 11434).
- **Input Validation:** PDF processing includes size limits and content validation. LLM prompts are template-based with parameterised inputs, mitigating prompt injection risks.
- **GDPR Compliance:** The system does not store personal data. Hospital strategic and action plans contain institutional (not patient) data. Session-scoped processing ensures no data persistence.
- **Dependency Security:** All dependencies are pinned to specific versions, and the system uses well-maintained open-source libraries (ChromaDB, sentence-transformers, RDFLib, NetworkX).

---

## 7. Testing and Evaluation

### 7.1 Evaluation Methodology

The ISPS system is evaluated against a 58-pair human-annotated ground truth dataset where each pair (objective, action) receives a manual alignment score on a four-point scale: 1.0 (strongly aligned), 0.7 (moderately aligned), 0.4 (weakly aligned), and 0.0 (not aligned). The system's embedding-based cosine similarity scores are compared against two baselines:

1. **TF-IDF Baseline:** TF-IDF vectorisation with cosine similarity — a classical IR approach based on term frequency.
2. **Keyword Overlap Baseline:** Jaccard coefficient between keyword sets — a simple lexical overlap measure.

### 7.2 Evaluation Metrics

**Classification Metrics:** Precision, Recall, F1-score, and Accuracy are computed using a binary classification threshold (scores >= 0.5 are "aligned"). Confusion matrices visualise true/false positive/negative distributions.

**Regression Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Pearson correlation coefficient (r), and Spearman rank correlation (rho) measure the agreement between predicted similarity scores and ground-truth annotations.

**Ranking Metrics:** ROC curves and Area Under the Curve (AUC) evaluate the system's ability to rank aligned pairs above non-aligned pairs across all threshold values.

**Statistical Significance:** Paired t-tests compare the ISPS system against each baseline to determine whether performance differences are statistically significant (p < 0.05).

### 7.3 Suggestion Quality Evaluation

A separate evaluation module assesses the quality of LLM-generated improvement suggestions across four dimensions:

1. **Relevance:** Cosine similarity between the suggestion text and the target strategic objective embedding.
2. **Specificity:** 1 minus the similarity to a generic template, penalising boilerplate suggestions.
3. **Actionability:** Heuristic score based on presence of concrete indicators — KPIs, timelines, budget figures, named owners, and measurable targets.
4. **Coherence:** LLM-as-judge evaluation where the Ollama model rates each suggestion on a 1–5 scale for logical consistency and strategic fit.

These metrics are computed for three ISPS sources (RAG improvements, RAG new-action suggestions, and Agent recommendations) and compared against generic baseline templates to quantify the value added by the retrieval-augmented and agentic approaches.

---

## 8. Results, Discussion, and Critical Analysis

The ISPS system produces a comprehensive synchronization assessment across multiple analytical dimensions. The overall synchronization score provides a single, interpretable metric that summarises the alignment quality of the entire action plan. The distribution of actions across classification bands (Excellent, Good, Fair, Poor) reveals the shape of alignment quality — whether the plan has a few strong anchors or broad moderate support.

The strategy-wise analysis reveals differential coverage across objectives. Objectives with high coverage scores and strong top-action alignments indicate well-operationalised strategic areas, while those with low coverage or numerous gap actions highlight areas requiring management attention. The ontology mapping adds a concept-level dimension that pure embedding similarity cannot capture, identifying cases where an action is semantically related to an objective but operates through a different ontological pathway than expected.

The knowledge graph provides structural insights that complement the pairwise similarity analysis. Betweenness centrality identifies bottleneck nodes — stakeholders who own multiple critical actions, or ontology concepts that bridge otherwise disconnected strategic areas. Community detection reveals natural clusters of strategy-action-concept alignment, while the connection suggestion feature provides evidence-based recommendations for closing coverage gaps.

The RAG recommendations demonstrate the value of grounding LLM generation in retrieved context. By providing the LLM with the action's metadata, alignment analysis, and relevant strategic objective text retrieved from ChromaDB, the system generates suggestions that reference specific KPIs, timelines, and stakeholders from the hospital's own planning documents rather than producing generic healthcare advice.

The agentic reasoning layer adds diagnostic depth by categorising issues into distinct types (orphan actions, under-supported strategies, conflicting mappings, KPI mismatches), prioritising them by composite impact scores, and self-critiquing its proposals. The structured trace provides full explainability — a critical requirement for AI systems in healthcare decision support where stakeholders need to understand and trust the reasoning behind recommendations.

---

## 9. Conclusions and Future Work

The ISPS system demonstrates that modern information retrieval techniques — semantic embeddings, vector databases, ontology mapping, knowledge graphs, RAG, and agentic AI — can be effectively integrated to provide intelligent, explainable analysis of strategic-operational alignment in the healthcare domain. The system moves beyond simple keyword matching to capture semantic relationships, provides multi-dimensional analysis through complementary techniques, and generates grounded, actionable improvement suggestions.

**Limitations and Future Work:**

1. **Embedding Model:** The all-MiniLM-L6-v2 model, while efficient, is a general-purpose model. Fine-tuning on healthcare strategic planning corpora could improve domain-specific similarity detection.
2. **LLM Scale:** The llama3.1:8b model, while privacy-preserving through local deployment, has limited capacity compared to larger models. Future work could explore distilled domain-specific models or secure API integration with larger cloud models.
3. **Longitudinal Analysis:** The current system analyses a single snapshot. Extending to track alignment changes over time would provide trend analysis and early warning of strategic drift.
4. **Multi-stakeholder Input:** Incorporating stakeholder feedback loops where domain experts validate and refine AI suggestions would improve recommendation quality and build organisational trust.
5. **Scalability:** While the current architecture handles single-hospital analysis efficiently, multi-facility or health-system-wide analysis would require distributed vector databases and parallelised pipeline stages.
6. **Evaluation Enhancement:** Expanding the ground truth dataset and incorporating domain-expert validation of AI recommendations would strengthen the evaluation methodology.

---

## References

Gruber, T.R. (1993) 'A translation approach to portable ontology specifications', *Knowledge Acquisition*, 5(2), pp. 199–220.

Hogan, A. et al. (2021) 'Knowledge graphs', *ACM Computing Surveys*, 54(4), pp. 1–37.

Johnson, J., Douze, M. and Jegou, H. (2019) 'Billion-scale similarity search with GPUs', *IEEE Transactions on Big Data*, 7(3), pp. 535–547.

Karpukhin, V. et al. (2020) 'Dense passage retrieval for open-domain question answering', in *Proceedings of EMNLP 2020*, pp. 6769–6781.

Lewis, P. et al. (2020) 'Retrieval-augmented generation for knowledge-intensive NLP tasks', in *Advances in Neural Information Processing Systems*, 33, pp. 9459–9474.

Malkov, Y.A. and Yashunin, D.A. (2020) 'Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs', *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), pp. 824–836.

Manning, C.D., Raghavan, P. and Schutze, H. (2008) *Introduction to Information Retrieval*. Cambridge: Cambridge University Press.

Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: sentence embeddings using Siamese BERT-Networks', in *Proceedings of EMNLP-IJCNLP 2019*, pp. 3982–3992.

Touvron, H. et al. (2023) 'LLaMA: open and efficient foundation language models', *arXiv preprint arXiv:2302.13971*.

Wang, W. et al. (2020) 'MiniLM: deep self-attention distillation for task-agnostic compression of pre-trained transformers', in *Advances in Neural Information Processing Systems*, 33, pp. 5776–5788.

Yao, S. et al. (2023) 'ReAct: synergizing reasoning and acting in language models', in *Proceedings of ICLR 2023*.
