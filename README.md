# **Finance QA Chatbot (Neural Pipeline MVP)**

> 🧠 A natural language interface for financial data.
> Ask questions like *“What was the total revenue in 2023?”* and get instant, data-driven answers powered by semantic search, Text-to-SQL, and an LLM.


## **1. Overview**

This project builds an intelligent chatbot that enables **non-technical users** to query financial data (e.g., trial balances) in plain English.
It integrates data preprocessing, semantic entity linking, text-to-SQL generation (via BERT + PICARD), and natural-language answer generation — all in a single neural pipeline.

**Architecture (purple-only version):**

1. Query normalization and topic routing
2. Semantic entity linking using embeddings
3. Text-to-SQL model with PICARD validation
4. DuckDB for execution and data storage
5. LLM-based answer generation and summarization
6. Optional RAG path for conceptual questions


## **2. Features**

✅ Natural language question parsing

✅ Automatic SQL query generation and execution

✅ Semantic matching for financial account names

✅ Interpretable numeric and trend responses

✅ FastAPI backend + simple chat UI

✅ Evaluation metrics for accuracy, latency, and response quality


## **3. Folder Structure**

```bash
Finance-QA-Chatbot/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── orchestrator.py         # Core pipeline logic
│   ├── sql_executor.py         # DuckDB query runner
│   ├── text_to_sql.py          # BERT + PICARD wrapper
│   ├── embeddings/
│   │   ├── linker.py
│   │   └── vector_db.py
│   ├── utils/
│   │   ├── time_parser.py
│   │   └── query_router.py
│   └── config/
│       ├── settings.yaml
│       └── roles.yaml
├── frontend/
│   └── app.py                  # Streamlit or React frontend
├── data/
│   └── (ignored via .gitignore)
├── tests/
│   ├── qa_set.json             # Evaluation queries and ground truth
│   └── test_utils.py
├── reports/
│   ├── final_metrics.md
│   └── project_report.docx
├── /docs/
│   ├── tech_stack.md
│   ├── architecture_diagram.png
│   ├── design_decisions.md
│   └── evaluation_plan.md
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```


## **4. Installation**

```bash
# Clone repo
git clone https://github.com/Pauwels-Xander/finance-qa-chatbot.git
cd finance-qa-chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Setup project dependencies and environment
make setup
```

*(Ensure DuckDB, FAISS/Chroma, and FastAPI dependencies are included in `requirements.txt`.)*


## **5. Quick Start**

```bash

# 1. Start both backend and frontend at once
make start

# 2. Ask a question!
# → "What was the total revenue in 2022?"
# → "How did profit change between 2021 and 2022?"
```


## **6. Evaluation**

Run the evaluation harness on the QA dataset:

```bash
python tests/eval_harness.py
```

Outputs:

* SQL success rate
* Numeric accuracy
* Average response latency
* Qualitative examples


## **7. Team**

| Name       | Role                | Focus Area                            |
| ---------- | ------------------- | ------------------------------------- |
| **Xander** | ML Engineer         | Text-to-SQL, embeddings, model tuning |
| **Anh**    | Data Engineer       | Data ingestion, DuckDB, preprocessing |
| **Fion**   | Backend Developer   | Pipeline orchestration, FastAPI       |
| **Josijah** | Research & Frontend | Evaluation, UI, documentation         |


## **8. License**

MIT License — free to use for educational purposes.

## **9. Acknowledgments**

* Ontario Energy Board (OEB) — *Trial Balance Open Data*
* Hugging Face, DuckDB, LangChain, and OpenAI ecosystems
