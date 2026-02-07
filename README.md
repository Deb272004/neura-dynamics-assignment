📚 PolicyQA: Retrieval-Augmented Generation (RAG) System

This repository contains a specialized RAG system designed to answer questions based on company policy documents. The system focuses on grounded accuracy, hallucination avoidance, and prompt engineering.

## 📁 Project Structure
``` text
└── 📁company_policy_chat
    └── 📁__pycache__
        ├── __init__.cpython-312.pyc
    └── 📁config
        ├── __init__.py
        ├── config.yaml
    └── 📁logger
        └── 📁__pycache__
            ├── __init__.cpython-312.pyc
            ├── custom_logger.cpython-312.pyc
        ├── __init__.py
        ├── custom_logger.py
    └── 📁prompts
        └── 📁__pycache__
            ├── __init__.cpython-312.pyc
            ├── prompts_library.cpython-312.pyc
        ├── __init__.py
        ├── prompts_library.py
    └── 📁src
        └── 📁__pycache__
            ├── __init__.cpython-312.pyc
        └── 📁document_ingestion
            └── 📁__init__.py
            └── 📁__pycache__
                ├── ingestion.cpython-312.pyc
            ├── ingestion.py
        └── 📁document_retrieval
            └── 📁__pycache__
                ├── __init__.cpython-312.pyc
                ├── retrieval.cpython-312.pyc
            ├── __init__.py
            ├── retrieval.py
        ├── __init__.py
    └── 📁utils
        └── 📁__pycache__
            ├── __init__.cpython-312.pyc
            ├── config_loader.cpython-312.pyc
            ├── file_utils.cpython-312.pyc
            ├── model_loader.cpython-312.pyc
        ├── __init__.py
        ├── config_loader.py
        ├── file_utils.py
        ├── model_loader.py
    └── __init__.py
```

## 🏗️ Architecture Overview

    Language Model: GROQ : llama-3.1-8b-instant

    Vector Database: FAISS for semantic storage.

    Retrieval Strategy: MMR (Maximal Marginal Relevance) to balance relevance with information diversity.

    Core Logic: Built with Python using a modular RAG pipeline.

## 📝 Prompt Engineering & Iteration

The system utilizes structured prompting to ensure the model remains grounded in the provided context.
Initial Prompt

The baseline prompt focused on simple instruction following:

    "CONTEXT_QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant designed to answer questions using the provided context only. "
        "If the answer is not present in the context, reply with: 'I don't have context about it.' "
        "Keep the answer concise (maximum three sentences unless the user asks otherwise).\n\n"
        "{context}"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])".

<p align="center">
  <img src="/home/deblina/Documents/projects/neura dynamics assignment/screenshots/Screenshot from 2026-02-07 12-55-21.png" width="90%" alt="Trace Details">
  <br>
  <em>Detailed trace showing 1.0 correctness on policy queries.</em>
</p>

Improved Iteration

The prompt was updated to include specific formatting constraints and "graceful failure" instructions:

    "
CONTEXT_QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "### 1. ROLE\n"
        "You are a Senior HR Policy Advisor. Your responsibility is to provide employees"
        "with clear, professional guidance on company policies and expectations, ensuring they understand procedures and best practices.\n\n"

        "### 2. GOAL\n"
        "Your primary objective is to assist new employees and answer frequently asked questions."
        "Emphasize that the employee handbook outlines the company’s rights and expectations, serving as a reference rather than a contractual guarantee.\n\n"

        "### 3. TEMPLATE & STRUCTURE\n"
        "Base your responses on the sections of the Model Employee Handbook, using them as a framework for clarity:\n"
        "- Workplace Commitments: Equal Opportunity, Non-Harassment, and Drug-Free policies.\n"
        "- Procedures: Professional conduct, Dress Code, Payday, and Personnel Files.\n"
        "- Leave Policies: Vacation, Sick Leave, FMLA, Jury Duty, and Voting.\n"
        "- Discipline & Safety: Grounds for disciplinary action, termination procedures, and emergency safety protocols.\n\n"

        "### 4. CONTEXT RULES\n"
        "- Use only the information provided in the context. If an answer is not present, respond: 'I don't have context about it.'\n"
        "- If any discrepancy exists between the handbook and current company policy, advise employees to follow the current policy.\n"
        "- Make it clear that the company may revise, revoke, or update policies at its discretion.\n\n"

        "### 5. GUARDRAILS & LEGAL ALERTS\n"
        "Ensure your responses respect these critical guidelines:\n"
        "- Employment is at-will: either the employee or the company may terminate employment at any time, with or without reason.\n"
        "- The handbook is not a contract and does not guarantee employment for any specific duration.\n"
        "- You are not providing legal advice. Employees should contact an authorized supervisor or a local employment attorney for legal guidance.\n"
        "- Employees should have no expectation of privacy when using company property, computers, or electronic equipment.\n"
        "- The company prohibits retaliation against anyone reporting harassment or discrimination in good faith.\n\n"

        "CONTEXT FOR REFERENCE:\n{context}"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
.".

<p align="center">
  <img src="/home/deblina/Documents/projects/neura dynamics assignment/screenshots/Screenshot from 2026-02-07 12-57-36.png" width="90%" alt="Trace Details">
  <br>
  <em>Detailed trace showing 1.0 correctness on policy queries.</em>
</p>

Why it changed: The initial prompt sometimes produced conversational filler. The iteration enforced a structured format (headings/bullets) and improved the handling of partially answerable questions.

## 📊 Evaluation Results

The system was evaluated against a "Golden Dataset" of 8 questions, including answerable, partially answerable, and unanswerable edge cases.
Performance Stability (LangSmith Analysis)

Based on comparative experiments with identical MMR settings and model configurations:
Metric	Result
Average Correctness	

0.75 
P50 Latency	6.68s
Error Rate	

0% (Consistent hallucination avoidance) 
Key Insights

    Consistency: The system maintained a stable 75% correctness rate across multiple runs, indicating high reproducibility.

    Systematic Gaps: Failures occurred consistently on complex "list-based" queries (e.g., "Under what conditions is an employee eligible..."). This suggests the information might be spread across multiple chunks that MMR is currently deprioritizing.

    Efficiency: Average token usage remained stable (~9,300 tokens), confirming efficient context window management.

<p align="center">
  <img src="/home/deblina/Documents/projects/neura dynamics assignment/screenshots/Screenshot from 2026-02-07 12-21-16.png" width="90%" alt="Trace Details">
  <br>
  <em>Detailed trace showing 1.0 correctness on policy queries.</em>
</p>

## 🛠️ Setup Instructions

    Clone the Repo: git clone [your-repo-link].

    Install Dependencies: pip install -r requirements.txt.

    Environment Variables: Add your GROQ_API_KEY to a .env file.

    Run the System: python test.py.

## 🛠️ Data Preparation & Chunking Strategy
Recursive Character Text Splitting

For this project, I implemented the RecursiveCharacterTextSplitter.

    Why it was chosen: The current policy documents consist of relatively small, well-defined sections. This splitter is ideal as it attempts to keep related pieces of text (like paragraphs and sentences) together by recursing through a list of characters (\n\n, \n,      , ""), maintaining the semantic context of each policy rule.

    Current Performance: Given the small scale of the current "Small Business" policy files, this strategy worked effectively to produce chunks that fit well within the LLM's context window while preserving ground truth.

Future Scalability & Adaptability

I designed the document_ingestion.py module with modularity in mind to handle future growth:

    Structural Diversity: If future files have significantly different structures (e.g., complex tables or nested hierarchies), we can transition to Markdown or HTML-aware loaders to better capture document semantics.

    Specialized Splitting: For highly technical or long-form legal documents, we could implement Semantic Chunking (using embeddings to find natural break points) or Token-based splitting to ensure precise context window management.

## ⚖️ Trade-offs & Future Improvements

Refining a RAG system involves balancing speed, accuracy, and resource constraints. Below are the current trade-offs made during development and the roadmap for future scalability.
Current Trade-offs

    MMR vs. Top-K Retrieval: We utilized Maximal Marginal Relevance (MMR) to reduce redundancy in the context window. While this improved response speed and diversity, it occasionally caused the model to miss granular supporting details required for complex, multi-item lists found in certain policy sections.

    Prompt Specificity: The system uses a structured prompt to enforce hallucination control. The trade-off is a more "rigid" output that may lack conversational flow in favor of strict factual grounding.

Future Work & Scalability
1. Improving Accuracy

    Reranking: To break the current 75% accuracy ceiling observed in LangSmith, we plan to implement a Cross-Encoder reranker. This would ensure that the most relevant documents are prioritized after the initial retrieval step.

    Hybrid Search: By combining Keyword Search (BM25) with Semantic Retrieval, the system can better handle specific policy "section" lookups that rely on exact terminology rather than just conceptual meaning.

2. Infrastructure & Optimization

    Scaling with HNSW: As the policy dataset grows, we will transition to Hierarchical Navigable Small World (HNSW) indexing to maintain fast retrieval speeds at scale.

    Memory Efficiency: For environments with memory constraints, Product Quantization (PQ) will be implemented to compress vector embeddings without significant loss in retrieval precision.

    Advanced Backend: The current CLI-based script can be evolved into a full-scale backend (FastAPI/Docker) to support multiple concurrent users and persistent semantic retrieval.
