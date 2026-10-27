# BurnAid AI: Intelligent Burn Depth Diagnosis & Clinical Triage System

An end-to-end intelligent medical decision-support system for pre-hospital thermal burn assessment, automated burn depth classification, lesion segmentation, and verified first-aid protocol guidance.

---

## Key Features

- **2-Stage Computer Vision Pipeline**:
  - **Stage 1 (Classification)**: Deep feature extraction via **ConvNeXtV2** to classify thermal burn wounds into **Degree I (Superficial)**, **Degree II (Partial-Thickness)**, **Degree III (Full-Thickness)**, or **No Burn (Healthy Skin)**.
  - **Stage 2 (Segmentation)**: High-resolution wound delineation via **UNet++** to segment affected lesion pixels and render realistic contour overlays.

- **Standardized Medical Calculation & Triage Rules**:
  - **Estimated TBSA (% Total Body Surface Area)**: Formulated according to **Prof. Le The Trung's 22 Adult Anatomical Regions (1-3-6-9-18 standard)**, adapted for Vietnamese and Asian adult body metrics.
  - **Abbreviated Burn Severity Index (ABSI)**: Automated background calculation of clinical prognosis scores (Score 2 to 16) and survival probability categories for medical records and doctor escalation, while keeping the patient-facing interface clean and panic-free.
  - **Unified Clinical Safety Engine (`clinical_rules.py`)**: Single source of truth across UI, API, and database. Immediate emergency escalations for Degree III burns or suspected smoke inhalation injuries.

- **Verified Clinical Knowledge Base & RAG Chat Assistant**:
  - Curated corpus of **2,274 clinical chunks** extracted from **18 verified burn care monographs** and Vietnam Ministry of Health **Decision 635/QD-BYT**.
  - Powered by high-speed Groq LLM inference with an automated **Offline Clinical RAG Engine fallback** if internet or API keys are unavailable.

- **Production-Ready Relational Database Architecture**:
  - **SQL Server (SSMS) Schema v9.2**: Full DDL and DML scripts with identity insert management, multi-area case modeling, and analytical views.
  - **SQLite Integration**: Fully normalized schema with active foreign keys and strict CHECK constraints (`burnaid.db`).

- **Streamlined, Professional Clinical UI**:
  - Pure English interface, minimalist vector iconography, and responsive mobile-first layout.
  - One-tap Emergency 115 dialing with non-blocking guidance so patients can read life-saving first-aid instructions without obstructive popups.

---

## System Architecture

```
                       [ Patient / Caregiver ]
                                  │
                                  ▼
                     [ burn_aid_prototype.html ]
                   (Responsive Mobile-First UI)
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[ REST API Server (api_server.py) ]           [ Client-Side Engine (JS) ]
  • ConvNeXtV2 Classifier                       • Local State & IndexedDB
  • UNet++ Wound Segmenter                      • GS. Le The Trung Reference
  • Groq LLM / Offline RAG Engine               • Background ABSI Computation
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  ▼
                     [ database.py / SQLite ]
                (BurnAid Schema v9.2 Architecture)
```

---

## Project Structure

```
├── api_server.py                    # FastAPI / Starlette backend server
├── burn_aid_prototype.html          # Single-page interactive clinical web application
├── clinical_rules.py                # Unified Clinical Rule Engine (Single Source of Truth)
├── database.py                      # Database models, ABSI calculator, and SQLite ORM
├── burnaid.db                       # Pre-seeded SQLite database (Schema v9.2)
├── knowledge_base_burns.txt         # Curated clinical RAG corpus (2,274 chunks)
├── models/                          # Trained neural network weights
│   ├── convnextv2_burn_4class_best.pth
│   └── best_random_curriculum_seed2026.pt
├── BurnAid_Full_Database_v9.2_SSMS.sql  # SQL Server (SSMS) complete DDL + DML script
├── .env.example                     # Environment variables template
├── .gitignore                       # Git exclusion rules
└── README.md                        # Documentation and architecture guide
```

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.10 or higher
- Git (optional, for version control)

### 2. Environment Setup
Clone the repository and install required Python packages:

```bash
# Clone the repository
git clone https://github.com/your-username/burnaid-ai.git
cd burnaid-ai

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install torch torchvision opencv-python numpy fastapi uvicorn starlette groq
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and set your Groq API key (optional for online AI chat):

```bash
cp .env.example .env
```

Edit `.env`:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(Note: If no key is set, the system seamlessly operates using its built-in Offline Clinical RAG Engine).*

### 4. Run the Application
Launch the unified server:

```bash
python api_server.py
```

The server will automatically load the models and open your default web browser to:
**`http://127.0.0.1:8000`**

---

## Clinical Safety Disclaimer

> **IMPORTANT CLINICAL NOTICE:** BurnAid AI is designed as a pre-hospital first-aid decision support and educational triage tool. It does not replace professional clinical evaluation, surgical diagnosis, or emergency medical dispatch. For deep (Degree III), circumferential, electrical, or extensive burns exceeding 10% TBSA, immediately call **115 / 911** or seek urgent emergency hospital care.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
