# 100 Days of AI/ML 🤖

Building AI/ML from scratch — one day, one concept, one commit.

**Co-founder going deep on the full AI/ML stack. Building in public daily.**
No fluff. No certificates. Just real code, real output, real GitHub commits.

---

## 🗺️ Roadmap

| Phase | Days | Topic | Status |
|-------|------|-------|--------|
| Phase 1 | 1–15 | Math & Data Foundations | ✅ Done |
| Phase 2 | 16–25 | Classical ML Algorithms | ✅ Done |
| Phase 3 | 26–40 | LLM APIs, Embeddings, RAG, LangChain | ✅ Done |
| Phase 4 | 41–60 | AI Agents, Tools, LangChain Agents | 🔄 In Progress |
| Phase 5 | 61–80 | Deploy, FastAPI, Fine-tuning | ⏳ Upcoming |
| Phase 6 | 81–100 | System Design + Job Push | ⏳ Upcoming |

---

## 📁 Structure
100-days-of-aiml/
├── phase1-foundations/           # Days 1–15
├── phase2-classical-ml/          # Days 16–25
│   ├── day16_linear_regression/
│   ├── day17_logistic_regression/
│   ├── day18_decision_trees/
│   ├── day19_random_forest/
│   ├── day20_evaluation/
│   ├── day21_cross_validation/
│   ├── day22_knn_naive_bayes/
│   ├── day23_svm/
│   ├── day24_xgboost/
│   └── day25_churn_prediction/
├── phase3-llm-rag/               # Days 26–40
│   ├── day26_llm_apis/
│   ├── day27_prompt_engineering/
│   ├── day28_embeddings/
│   ├── day29_vector_db/
│   ├── day30_rag_pipeline/
│   ├── day31_langchain_basics/
│   ├── day32_document_loaders/
│   ├── day33_retrieval_chain/
│   ├── day34_memory/
│   ├── day35_project1_rag_bot/
│   ├── day36_output_parsers/
│   ├── day37_ragas_eval/
│   ├── day38_fastapi/
│   ├── day39_langsmith/
│   └── day40_project1_final/
├── phase4-agents/                # Days 41–60
│   ├── day41_tools/
│   ├── day42_react_agent/
│   ├── day43_langchain_agents/
│   ├── day44_multi_tool_agent/
│   ├── day45_agent_memory/
│   ├── day46_agent_debugging/
│   ├── day47_project2_start/
│   ├── day48_project2_tools/
│   ├── day49_project2_ui/
│   └── day50_project2_final/
└── projects/
├── project1-rag-bot/
└── project3-churn-prediction
---

## 📅 Daily Log

| Day | Topic | Key Output |
|-----|-------|------------|
| 01 | NumPy | Arrays, broadcasting, shapes |
| 02 | Pandas | DataFrames, groupby, cleaning |
| 03 | Visualization | Matplotlib + Seaborn EDA |
| 04 | Statistics | Normal distribution, 68-95-99.7 rule |
| 05 | Linear Algebra | Matrix multiply, forward pass |
| 06 | Probability | Bayes theorem, distributions |
| 07 | EDA | Full exploratory analysis — Titanic |
| 08 | Data Cleaning | Nulls, outliers, encoding |
| 09 | Feature Engineering | New features, scaling |
| 10 | Pipeline | End-to-end data pipeline |
| 11–15 | Scikit-learn Basics | Train/test, first models |
| 16 | Linear Regression | R²=0.55, coefficient analysis |
| 17 | Logistic Regression | Acc=98.25%, ROC-AUC=0.9954 |
| 18 | Decision Trees | Tree viz, bias-variance tradeoff |
| 19 | Random Forest | OOB=0.95, feature importance |
| 20 | Model Evaluation | ROC curve, precision, recall, F1 |
| 21 | Cross-Validation | 5-fold CV, confidence intervals |
| 22 | KNN + Naive Bayes | K=10, F1=0.9726 |
| 23 | SVM | 97 support vectors, GridSearchCV |
| 24 | XGBoost | F1=0.9785, boosting vs bagging |
| 25 | Churn Prediction | Full pipeline + predict() function |
| 26 | LLM APIs | Anthropic API, JSON output, few-shot |
| 27 | Prompt Engineering | Zero-shot, few-shot, chain-of-thought |
| 28 | Embeddings | Cosine similarity, semantic search |
| 29 | Vector DB | ChromaDB store and retrieve |
| 30 | RAG Pipeline | Full retrieve + generate from scratch |
| 31 | LangChain Basics | LCEL pipe syntax, sequential chains |
| 32 | Document Loaders | Text splitter, chunk size comparison |
| 33 | Retrieval Chain | Full LangChain RAG in 30 lines |
| 34 | Memory | Multi-turn chat, session isolation |
| 35 | Project 1 — RAG Bot | Streamlit UI + ChromaDB + Claude |
| 36 | Output Parsers | Pydantic parser, structured extraction |
| 37 | RAG Evaluation | 5/5 questions, avg score 0.78 |
| 38 | FastAPI | REST API — POST /ask endpoint |
| 39 | LangSmith | Traces, token usage, latency per step |
| 40 | Project 1 Final | README + architecture diagram |
| 41 | Tools | 5 tools built, LLM tool selection |
| 42 | ReAct Agent | Reason + Act loop, 3 tasks solved |
| 43 | LangChain Agents | Shopping agent, tool routing |
| 44 | Multi-Tool Agent | Market research chaining |
| 45 | Agent Memory | MemorySaver, persistent context |
| 46 | Agent Debugging | Failure modes, graceful handling |
| 47 | Project 2 Start | Market research tools + agent |
| 48 | Project 2 Tools | Competitor analysis, report writer |
| 49 | Project 2 UI | Streamlit agent interface |
| 50 | Project 2 Final | README, Phase 4 midpoint recap |

---

## 🏗️ Projects

### Project 1 — Ecommerce RAG Support Bot
**Stack:** LangChain + ChromaDB + Claude + FastAPI + Streamlit + LangSmith
**What it does:** Answers customer support questions from a FAQ knowledge base. Grounded responses only — no hallucination.
**Folder:** `phase3-llm-rag/day35_project1_rag_bot/`

### Project 2 — Ecommerce Market Research Agent
**Stack:** LangChain Agents + LangGraph + Claude + Streamlit
**What it does:** Autonomously researches ecommerce markets, analyzes competitors, and generates structured reports.
**Tools:** Product search, price segments, market trends, review analysis, competitor analysis, report writing
**Folder:** `phase4-agents/day49_project2_ui/`

### Project 3 — Churn Prediction Pipeline
**Stack:** Scikit-learn + XGBoost + FastAPI + Pickle
**What it does:** Predicts customer churn from raw customer data. XGBoost — F1: 0.631, ROC-AUC: 0.873
**Folder:** `projects/project3-churn-prediction/`

---

## 🛠️ Tech Stack

**Phase 1–2:** Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost

**Phase 3:** Anthropic Claude, LangChain, ChromaDB, HuggingFace, FastAPI, Streamlit, LangSmith

**Phase 4:** LangGraph, LangChain Agents, ReAct, MemorySaver, Custom Tools

---

## 📌 Connect

- **LinkedIn:** linkedin.com/in/aryaraut13
- **GitHub:** github.com/aryaraut13

---

*Posting every day on LinkedIn. Follow along.*