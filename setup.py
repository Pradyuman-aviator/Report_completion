# setup.py
# --------
# Minimal shim required for `pip install -e .` compatibility with older pip
# versions (< 21.3) that don't support PEP 660 editable installs natively.
#
# All real project configuration is in pyproject.toml.
# Do NOT add metadata here — it will conflict with pyproject.toml.
#
# Usage:
#   pip install -e .              →  core deps only
#   pip install -e ".[agents]"    →  + Ollama OCR agents
#   pip install -e ".[ml]"        →  + PyTorch training pipeline
#   pip install -e ".[all]"       →  everything

from setuptools import setup

if __name__ == "__main__":
    setup()
