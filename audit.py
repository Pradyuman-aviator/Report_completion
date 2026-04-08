"""audit.py — one-shot project health check"""
import os, sys
checks = []

# 1. data pipeline
try:
    from ml_pipeline.data.dataset import build_loaders
    tl, vl, meta = build_loaders("data/raw", batch_size=8)
    import torch
    b = next(iter(tl))
    assert b["masked_x"].shape == (8, meta["num_features"])
    checks.append(("data/dataset.py",        "PASS", f"Train={meta['train_size']} Val={meta['val_size']} F={meta['num_features']}"))
except Exception as e:
    checks.append(("data/dataset.py",        "FAIL", str(e)[:90]))

# 2. ai_agents
try:
    from ai_agents.schemas import MedicalReportPayload
    from ai_agents.vision_agent import VisionAgent
    from ai_agents.structuring_agent import StructuringAgent
    checks.append(("ai_agents/",             "PASS", "schemas + vision + structuring imported"))
except Exception as e:
    checks.append(("ai_agents/",             "FAIL", str(e)[:90]))

# 3. ml_pipeline core modules
try:
    from ml_pipeline.loss import CombinedLoss
    from ml_pipeline.predictor import JEPAPredictor
    checks.append(("ml_pipeline/loss+pred",  "PASS", "CombinedLoss + JEPAPredictor"))
except Exception as e:
    checks.append(("ml_pipeline/loss+pred",  "FAIL", str(e)[:90]))

# 4. CUDA
try:
    import torch
    cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "CPU only"
    checks.append(("CUDA",                   "PASS" if cuda else "WARN", name))
except Exception as e:
    checks.append(("CUDA",                   "FAIL", str(e)[:90]))

# 5. FastAPI
try:
    import fastapi, uvicorn
    checks.append(("fastapi/uvicorn",        "PASS", f"fastapi {fastapi.__version__}"))
except Exception as e:
    checks.append(("fastapi/uvicorn",        "FAIL", str(e)[:90]))

# 6. api_gateway
try:
    content = open("api_gateway/main.py").read().strip()
    checks.append(("api_gateway/main.py",    "EMPTY" if not content else "PASS", f"{len(content)} chars"))
except Exception as e:
    checks.append(("api_gateway/main.py",    "FAIL", str(e)[:90]))

# 7. stray redirect files from bad >= syntax
stray = [f for f in ["2.6.0", "4.38.0"] if os.path.exists(f)]
checks.append(("stray files",              "CLEAN" if not stray else "WARN", str(stray) if stray else "none"))

# 8. README
readme = open("README.md", encoding="utf-8").read().strip()
checks.append(("README.md",               "EMPTY" if not readme else "PASS", f"{len(readme)} chars"))

print()
print("=" * 68)
print("  MEDFILL PROJECT AUDIT")
print("=" * 68)
icons = {"PASS": "V", "FAIL": "X", "WARN": "!", "EMPTY": "?", "CLEAN": "V"}
for name, status, note in checks:
    icon = icons.get(status, "?")
    print(f"  [{icon}] {status:5s}  {name:28s}  {note}")
print("=" * 68)
