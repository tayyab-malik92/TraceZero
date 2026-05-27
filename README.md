
# 🛡️ TraceZero: Autonomous AI Agent for Digital Privacy Enforcement

**TraceZero** is a full-stack, autonomous privacy-enforcement agent designed to counteract the unauthorized scraping, aggregation, and monetization of Personally Identifiable Information (PII) by online data brokers. Operating completely on the live web (without simulated data chains), TraceZero scans public directories, utilizes a multi-layered NLP transformer pipeline to isolate exposed credentials, scores matching confidences to eliminate false positives, and programmatically executes automated legal opt-out flows.

**Developers:**
-Muhammmad Tayyab (24L-0634)
-Abdullah Khan (24L-0772)

## 🏗️ Architecture & Core Workflow

The system is split into three decoupled operational layers overseen by a high-performance asynchronous **FastAPI** backend:

```text
[User Profile Input] 
       │
       ▼
 🔍 [Scraper Layer] ────────► Controls Chromium (Playwright) to query live data brokers
       │                      (Addresses.com, Intelius, ZabaSearch)
       ▼
 🧠 [AI Engine Layer] ──────► 1. RoBERTa-Large: Extracts raw PII using custom NER token classification
       │                      2. DistilBERT: Sequences text to classify exposure risk levels
       │                      3. Sentence-BERT: Generates embeddings to calculate cosine similarity matches
       ▼
 🛡️ [Enforcement Layer] ────► 1. Web Form-Filler: Programmatically submits removal requests
                              2. Jinja2 Templates: Auto-generates GDPR/CCPA legal deletion notices
🧠 Deep Learning & Machine Learning Pipelines
TraceZero implements three distinct, specialized Transformer models to process un-structured data scraped from data broker records:

1. Named Entity Recognition (NER) — RoBERTa-Large
Base Model: Jean-Baptiste/roberta-large-ner-english

Dataset Target: Fine-tuned via train.py on custom data mapping pipelines (such as the AI4Privacy layout) to confidently locate unstructured PII attributes (Name, Email, Phone, SSN, DOB, Address).

2. Risk Evaluation Classification — DistilBERT
Base Model: distilbert-base-uncased

Task: Text classification mapping profiles to dynamic severity metrics (High, Medium, Low) based on exposure density and identity patterns.

3. Identity Match & Similarity Calibration — Sentence-BERT
Base Model: sentence-transformers/all-MiniLM-L6-v2

Task: Extracts high-dimensional semantic embeddings from both the input user profile and the text scraped from the broker. It computes a Cosine Similarity Matrix to distinguish between the actual user and individuals sharing identical names, drastically mitigating false positives.

📁 Repository Directory Structure
Organize your repository matching this production-grade architecture layouts:

Plaintext
tracezero/
├── ai_engine/
│   ├── __init__.py
│   └── ner_pipeline.py          # Multi-model pipelines & similarity processing
├── database/
│   ├── __init__.py
│   └── db.py                    # SQLite engine handling audit logging & status registries
├── enforcement/
│   ├── __init__.py
│   ├── email_sender.py          # Jinja2 legal template compilation engine (GDPR/CCPA)
│   └── form_filler.py           # Batch enforcement orchestration routing logic
├── model_training/
│   └── train.py                 # Core model training, metrics monitoring & GPU acceleration config
├── scraper/
│   ├── __init__.py
│   ├── browser.py               # Playwright automated form-submission handlers
│   └── scraper.py               # Evasive stealth scrapers (anti-bot headers, human delays)
├── main.py                      # Main FastAPI deployment application script
├── requirements.txt             # Direct and pinned dependencies file
└── .gitignore                   # Ignores heavy models (/model, /risk_model), cache, and databases
🛠️ **Installation & Environment Setup**
Follow these steps to spin up the local server, download weights, and initialize the browser instances:

1. Clone the Repository
Bash
git clone [https://github.com/your-username/tracezero.git](https://github.com/your-username/tracezero.git)
cd tracezero
2. Configure Virtual Environment & Dependencies
Bash
# Create environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate
# Activate environment (Mac/Linux)
source venv/bin/activate

# Install required modules
pip install -r requirements.txt
3. Initialize Playwright Drivers
Playwright requires setting up its own localized headless browser binaries:

Bash
playwright install
4. (Optional) Train/Fine-Tune the ML Pipelines Locally
If you want to recreate the weights or execute evaluation reports (which produce semantic matching reports and F1 metrics):

Bash
python model_training/train.py
Note: The script automatically binds to cuda if an NVIDIA GPU is accessible.

🚀 Running TraceZero
Start the REST API Backend
Fire up the server using Uvicorn:

Bash
uvicorn main:app --reload
Once initialized, the service will open up a local port at http://127.0.0.1:8000.

Interacting with API Documentation
Navigate to http://127.0.0.1:8000/docs to view the Swagger Interactive UI. You can run tests against the live endpoints:

POST /scan: Evaluates search inputs on target brokers, running strings down the transformer pipeline to calculate exposure profiles.

GET /records: Returns all previous scans registered in the local SQLite database.

POST /enforce: Hooks into the Playwright automation scripts to trigger a targeted, form-based data deletion trace.

📝 Core Technologies Used
Backend Framework: FastAPI, Pydantic v2

Machine Learning / NLP Stack: PyTorch, Hugging Face Transformers, Sentence-Transformers, Scikit-Learn

Scraping & Interaction Engine: Playwright Async API, BeautifulSoup4

Templating & Utilities: Jinja2 Engine, Pandas, Seaborn/Matplotlib (for evaluation metrics output)

Database: SQLite3
