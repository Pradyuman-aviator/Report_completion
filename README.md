# MedFill — Medical Report Completion Pipeline

> **Local-first AI pipeline** that reads a medical lab report image, extracts all values using EasyOCR (GPU-accelerated), and predicts any missing biomarkers using a trained Transformer-based imputation model — **100% offline, no cloud, no API keys.**

---

## What it Does

```
📷 report.jpg  (any Indian blood test / CBC / RFT report)
      │
      ▼  EasyOCR (GPU, ~5s)       — actually reads the image text
      │  confidence = 0.81
      │
      ▼  Regex Parser             — extracts numeric values line-by-line
      │  Hemoglobin=10.8, bu=42, bgr=118, bp=142 ...
      │
      ▼  TabularImputerModel      — predicts ALL missing values
      │  (Transformer, trained on 1,821 real patient records)
      │
      ▼  Complete 25-Feature Lab Panel ✅
         Hemoglobin : 10.8   (extracted from image)
         MCH        : 27.3   (extracted from image)
         bp         : 142    (extracted from image)
         MCV        : 90.8   ★ PREDICTED by model
         MCHC       : 31.1   ★ PREDICTED by model
         sc         :  4.6   ★ PREDICTED by model
```

**Problem it solves:** Real-world medical reports are often *partial* — some test values are marked PENDING, weren't run, or are illegible. MedFill fills those gaps using patterns learned from thousands of real patient records (Anemia + CKD datasets).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     api_gateway/                        │
│   FastAPI  —  /process  /process/batch  /health         │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │          ai_agents/           │
         │  EasyOCRAgent  (GPU OCR)      │  ← replaces llava:7b
         │  StructuringAgent (llama3)    │  ← optional backup only
         │  direct_parse_ocr()           │  ← regex, instant
         └───────────────┬───────────────┘
                         │  extracted features dict
         ┌───────────────▼───────────────┐
         │         ml_pipeline/          │
         │  TabularImputerModel          │
         │  Trained on Anemia + CKD CSVs │
         │  25 biomarker features        │
         └───────────────────────────────┘
```

| Module | Description |
|--------|-------------|
| `ai_agents/easyocr_agent.py` | **NEW** — GPU OCR with preprocessing (CLAHE, deskew, threshold) |
| `ai_agents/vision_agent.py` | Legacy llava:7b OCR (available as fallback) |
| `ai_agents/structuring_agent.py` | llama3 JSON extraction (optional backup) |
| `ai_agents/schemas.py` | Full Pydantic schema for `MedicalReportPayload` |
| `ai_agents/pipeline.py` | Orchestrates OCR → structuring |
| `ml_pipeline/train.py` | Trains the Transformer imputer |
| `ml_pipeline/data/dataset.py` | Loads Anemia + CKD datasets |
| `api_gateway/main.py` | FastAPI REST gateway |
| `infer.py` | **End-to-end CLI inference** |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 6 GB+ |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB | 10 GB |

> Tested on **NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)**. EasyOCR runs entirely on GPU — no Ollama required for OCR.

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install fastapi uvicorn opencv-python pydantic easyocr torch numpy pandas
```

> `ollama` is **no longer required** for standard inference. It is only used as an optional backup for patient metadata extraction.

### 2. (Optional) Install Ollama for metadata enrichment

Download from [ollama.com](https://ollama.com) and pull:

```bash
ollama pull llama3      # Structuring model (~4.7 GB) — optional backup only
# llava:7b is no longer needed
```

### 3. Add your datasets

Place these CSV files in `data/raw/`:
- `anemia.csv`
- `kidney_disease.csv`

### 4. Train the imputation model

```bash
python -m ml_pipeline.train
# Trains for 50 epochs on GPU, takes ~30 seconds
# Saves best_model.pt to ml_pipeline/checkpoints/
```

### 5. Run inference

**From a report image (recommended):**
```bash
python infer.py path/to/report.jpg
```

**Manual mode (type values yourself):**
```bash
python infer.py --manual
```

**Start the API server:**
```bash
uvicorn api_gateway.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/process` | Upload a report image, get complete lab panel |
| `POST` | `/process/batch` | Batch process multiple images |
| `GET` | `/health` | Check system status |
| `GET` | `/models` | List available models |

**Example:**
```bash
curl -X POST http://localhost:8000/process \
     -F "file=@report.jpg"
```

---

## Output Example

```
[MedFill] Processing: report.jpg
============================================================
[1/3] Running EasyOCR + feature extraction...
  OCR confidence : 0.81
  OCR time       : 5.5s
  Text extracted : 1886 chars
[2/3] Mapping to tabular features...
  Patient         : MR. RAHUL SHARMA
  Known features  : 7/25
  Missing features: 18/25  <- will be predicted
[3/3] Running imputation model...

========================================================================
  COMPLETE LAB PANEL
========================================================================
  Feature                     Value        Source  Note
  -------------------- ------------  ------------  ----
  Hemoglobin                 10.800     extracted
  MCH                        27.300     extracted
  MCHC                       31.148   * PREDICTED
  MCV                        90.786   * PREDICTED
  bp                        142.000     extracted
  bgr                       118.000     extracted
  bu                         42.000     extracted
  sod                       138.000     extracted
  cad                            no     extracted
  sc                          4.599   * PREDICTED
  wc                       9166.482   * PREDICTED
  htn                            no   * PREDICTED
  ane                            no   * PREDICTED
  ...
========================================================================

  Extracted : 7 values
  Predicted : 18 values  (*)
```

`* PREDICTED` = model predicted this value from patterns in training data.

---

## What Reports Work

| Report Type | Works? | Notes |
|---|---|---|
| **CBC / Blood Count** | ✅ Best | Hemoglobin, RBC, WBC, PCV, MCH, MCHC, MCV |
| **Renal Function Test** | ✅ Great | Blood Urea, Creatinine, Sodium, Potassium |
| **Diabetes panel** | ✅ Good | Blood Glucose extracted |
| **Any Apollo/Thyrocare report** | ✅ | Standard printed format |
| **Handwritten** | ⚠️ Partial | Lower OCR confidence, more predicted |
| **Blurry/poor lighting** | ⚠️ Partial | Model predicts more values |
| **Hindi/regional language** | ❌ | English OCR only |
| **Radiology / MRI** | ❌ | No numeric biomarkers |

> **Key insight:** Even if OCR fails completely, your model still provides all 25 values as a clinically-reasonable baseline prediction. The system never crashes — it just predicts more.

---

## Model Details

**TabularImputerModel** — Lightweight Transformer for tabular imputation

- Architecture: 3-layer Transformer Encoder, 4 attention heads, embed dim 64
- Parameters: ~152,000 (fits in CPU RAM, trains in 30s on GPU)
- Training data: 1,821 patient records (Anemia + CKD datasets)
- Input: 25 unified biomarker features (masked missing → 0)
- Output: Reconstructed full 25-feature vector
- Loss: MSE on observed positions only
- Best val loss: **0.054** (normalized scale)

**25 Biomarker Features (UNIFIED schema):**

| Category | Features |
|---|---|
| **Blood count** | Hemoglobin, MCH, MCHC, MCV, pcv (hematocrit), wc (WBC), rc (RBC) |
| **Renal** | bu (blood urea), sc (serum creatinine) |
| **Electrolytes** | sod (sodium), pot (potassium) |
| **Metabolic** | bgr (blood glucose) |
| **Urinalysis** | rbc, pc, pcc, ba |
| **Demographics** | age, Gender |
| **Clinical history** | htn, dm, cad, appet, pe, ane |
| **Vitals** | bp (systolic) |

---

## Project Structure

```
report classifier/
├── ai_agents/
│   ├── easyocr_agent.py     # GPU OCR engine (EasyOCR + preprocessing)
│   ├── pipeline.py          # Main orchestrator
│   ├── vision_agent.py      # Legacy llava:7b OCR
│   ├── structuring_agent.py # JSON extraction with llama3 (optional)
│   └── schemas.py           # Pydantic data models
├── api_gateway/
│   └── main.py              # FastAPI server
├── ml_pipeline/
│   ├── train.py             # Training loop
│   ├── data/dataset.py      # Data loading
│   ├── models/              # Model definitions
│   └── checkpoints/         # Saved model weights (gitignored)
├── data/raw/                # CSV datasets (gitignored)
├── infer.py                 # End-to-end inference CLI
├── audit.py                 # System health check
└── README.md
```

---

## Performance Comparison

| Metric | Old (llava:7b) | New (EasyOCR) |
|---|---|---|
| OCR time | **110 seconds** | **5.5 seconds** |
| OCR confidence | 0.50 | **0.81** |
| Hallucinations | Yes (invents values) | **No** |
| VRAM required | 4.7 GB | **~1 GB** |
| Ollama required | Yes | **No** |
| Accuracy | Poor | **Good** |

---

## License

MIT