# MedFill — AI-Powered Medical Report Completion

> **Upload any partial blood test report → Get a complete 25-biomarker lab panel in ~15 seconds.**
> Local-first, offline, zero cloud. Your patient data never leaves your machine.

---

## The Problem

Real-world medical reports are **incomplete**. Tests are marked `PENDING`, values are missing due to cost or urgency, and doctors are forced to make decisions with partial data. MedFill fills those gaps instantly using AI.

---

## What It Does

```
📷 Partial blood report image  (Apollo, Thyrocare, any lab)
        ↓
EasyOCR  (GPU, ~5s)         — reads every word on the report
        ↓
Regex Parser                — extracts 12 known biomarker values
        ↓
TabularImputerModel         — predicts all 13 missing values
(Transformer · PyTorch)       R² = 99.7%
        ↓
✅ Complete 25-Feature Lab Panel
   Hemoglobin  : 10.8   g/dL   [extracted]
   MCH         : 27.3   pg     [extracted]
   MCV         : 90.8   fL     [★ AI predicted]
   Creatinine  :  4.6   mg/dL  [★ AI predicted]
   WBC Count   : 9,166  /µL    [★ AI predicted]
   ...
```

---

## Model Performance

| Metric | Value |
|---|---|
| **R² Score** | **0.9968 (99.7%)** |
| **Validation Loss (MSE)** | **0.001632** |
| **MAE (normalized)** | **0.0205** |
| **Training epochs** | 76 |
| **Val patients** | 2,231 |
| **Training patients** | 12,637 |
| **OCR Confidence** | 0.81 (81%) |
| **End-to-end time** | ~15 seconds |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              React Frontend  (localhost:5173)                     │
│   FileUpload → PatientDashboard → Report Analysis Panel          │
└──────────────────────────┬───────────────────────────────────────┘
                           │  POST /api/v1/analyze
┌──────────────────────────▼───────────────────────────────────────┐
│              FastAPI Backend  (localhost:8000)                    │
│                                                                  │
│  ① EasyOCRAgent        GPU OCR → raw text  (~5s, conf=0.81)      │
│  ② direct_parse_ocr()  Regex → 12 extracted features             │
│  ③ impute()            Transformer → 13 predicted features       │
│  ④ Response builder    patient + lab_panels + complete_panel     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│           TabularImputerModel  (ml_pipeline/)                    │
│                                                                  │
│  Input:  25 features (missing → 0)                               │
│  Embed:  Linear(1→128) + Positional Embedding(25,128)            │
│  Encoder: 4× TransformerEncoderLayer                             │
│           (8 heads · dim=128 · FFN=512 · Pre-LN · dropout=0.1)  │
│  Output: Linear(128→1) per feature → 25 reconstructed values    │
│  Params: 796,673                                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Datasets

| Dataset | Source | Patients | Features |
|---|---|---|---|
| **Anemia Dataset** | [Kaggle (CC0)](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset) | 1,421 | Hemoglobin, MCH, MCHC, MCV, Gender |
| **CKD Dataset** | [Kaggle — CKD Disease](https://www.kaggle.com/datasets/mansoordaku/ckdisease) · donated by Apollo Hospitals | 400 | 25 biomarkers (renal, electrolytes, urinalysis) |
| **NHANES** | [CDC NHANES](https://www.cdc.gov/nchs/nhanes/) | 10,816 | Blood count, metabolic, renal panels |
| **Combined** | — | **12,637** | **25 unified features** |

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Train the imputation model

```bash
python -m ml_pipeline.train
# ~30s on GPU · saves ml_pipeline/checkpoints/best_model.pt
```

### 3. Run inference from CLI

```bash
python infer.py path/to/report.jpg
```

### 4. Start the API server

```bash
uvicorn api_gateway.main:app --reload --port 8000
# Swagger docs: http://localhost:8000/docs
```

### 5. Start the React frontend

```bash
cd Fronted_medfill
npm install
npm run dev
# Open: http://localhost:5173
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/analyze` | Upload report image → complete lab panel (OCR + imputation in one call) |
| `GET` | `/health` | Backend status check |
| `GET` | `/docs` | Swagger UI |

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
     -F "file=@report.jpg"
```

**Response includes:**
```json
{
  "patient": { "name": "MR. RAHUL SHARMA", "age_years": 52, "sex": "Male" },
  "lab_panels": [
    {
      "panel_name": "Complete Blood Count (CBC)",
      "results": [
        { "test_name": "Hemoglobin (Hb)", "value": 10.8, "flag": "low", "unit": "g/dL" },
        { "test_name": "MCH", "value": 27.3, "flag": "normal", "unit": "pg" }
      ]
    }
  ],
  "complete_panel": {
    "Hemoglobin": { "value": 10.8, "source": "extracted", "predicted": false },
    "MCV":        { "value": 90.8, "source": "predicted",  "predicted": true  }
  },
  "ocr_confidence": 0.81,
  "n_extracted": 12,
  "n_predicted": 13,
  "elapsed_s": 14.7
}
```

---

## What Reports Work

| Report Type | Result |
|---|---|
| CBC / Blood Count | ✅ Best |
| Renal Function Test | ✅ Great |
| Diabetes / Glucose panel | ✅ Good |
| Apollo / Thyrocare / printed reports | ✅ Full support |
| Blurry / poor lighting | ⚠️ Partial (model predicts more) |
| Handwritten | ⚠️ Reduced accuracy |
| Hindi / regional language | ❌ English OCR only |
| Radiology / MRI | ❌ No numeric biomarkers |

---

## Performance Comparison (OCR Engine)

| Metric | Old (llava:7b VLM) | New (EasyOCR) |
|---|---|---|
| OCR time | 110 seconds | **5.5 seconds** |
| Confidence | 0.50 | **0.81** |
| Hallucinations | Yes | **No** |
| VRAM | 4.7 GB | **~1 GB** |
| Ollama required | Yes | **No** |

---

## Project Structure

```
report classifier/
├── ai_agents/
│   ├── easyocr_agent.py     ← GPU OCR engine (primary)
│   ├── pipeline.py          ← Orchestrator
│   ├── vision_agent.py      ← Legacy VLM OCR
│   ├── structuring_agent.py ← llama3 backup (optional)
│   └── schemas.py           ← Pydantic models
├── api_gateway/
│   └── main.py              ← FastAPI (all endpoints)
├── Fronted_medfill/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   └── components/
│   │       ├── FileUpload.jsx
│   │       ├── PatientDashboard.jsx
│   │       ├── PredictionPanel.jsx
│   │       └── Navbar.jsx
│   └── package.json
├── ml_pipeline/
│   ├── train.py             ← Training loop
│   ├── data/dataset.py      ← Data loading (NHANES + Anemia + CKD)
│   └── checkpoints/         ← best_model.pt (gitignored)
├── data/raw/                ← CSV datasets (gitignored)
├── infer.py                 ← CLI inference + impute()
├── requirements.txt         ← Python dependencies
└── README.md
```

---

## Hardware

| Component | Tested On |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop (6 GB VRAM) |
| RAM | 16 GB |
| Python | 3.11 |
| PyTorch | 2.5.1 + CUDA 12.1 |

---

## Contributors

| Name | Role |
|---|---|
| **Pradyuman** | ML pipeline · Transformer model · FastAPI backend · EasyOCR integration |
| **Zulikha** | Dataset research · Model evaluation · Clinical validation |
| **Kshitij** | React frontend · UI/UX design · API integration |

---

## License

MIT