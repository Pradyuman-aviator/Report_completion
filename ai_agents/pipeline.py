"""
ai_agents/pipeline.py
======================
Medical Report Processing Pipeline — Orchestrator
--------------------------------------------------

Chains the two agents into a single callable:

    Raw image path
        ↓
    VisionAgent.process()
        • OpenCV preprocessing (grayscale → CLAHE → denoise → threshold → deskew)
        • Ollama LLaVA / Llama-3.2-Vision OCR
        → VisionAgentResult (raw_text, confidence, steps, angle)
        ↓
    StructuringAgent.structure()
        • Prompt-engineered JSON extraction
        • Pydantic validation with 3-attempt retry loop
        • Soft-fail partial payload on exhaustion
        → MedicalReportPayload (fully typed, validated)
        ↓
    Caller  (api_gateway / CLI / test harness)

Design principles:
    • Single responsibility: pipeline.py only wires agents; no business logic.
    • Short-circuit: if VisionAgent confidence is below a configurable floor,
      log a warning but still attempt structuring (V-JEPA handles missing data).
    • Timing: end-to-end wall-clock time is always logged and added to metadata.
    • Idempotent: calling pipeline.process() twice on the same file is safe.

Usage:
    # Programmatic
    >>> from ai_agents.pipeline import MedicalReportPipeline
    >>> pipe = MedicalReportPipeline()
    >>> payload = pipe.process("scans/report_001.jpg")
    >>> print(payload.model_dump_json(indent=2))

    # CLI
    python -m ai_agents.pipeline scans/report_001.jpg [--out results/]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from ai_agents.schemas          import MedicalReportPayload
from ai_agents.vision_agent     import VisionAgent, VisionAgentResult
from ai_agents.structuring_agent import StructuringAgent

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Pipeline Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_VISION_MODEL      = "llava:7b"
DEFAULT_STRUCTURING_MODEL = "llama3"
DEFAULT_OLLAMA_HOST       = "http://localhost:11434"
LOW_CONFIDENCE_THRESHOLD  = 0.35   # Warn (but don't stop) below this OCR score


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Pipeline Class
# ──────────────────────────────────────────────────────────────────────────────

class MedicalReportPipeline:
    """
    End-to-end pipeline: image file → structured MedicalReportPayload.

    Parameters
    ----------
    vision_model : str
        Ollama tag for the multimodal VLM used in OCR.
        Default: "llama3.2-vision"
    structuring_model : str
        Ollama tag for the text LLM used in JSON structuring.
        Default: "llama3"
    ollama_host : str
        Base URL of the local Ollama HTTP server.
    save_cleaned_images : bool
        Whether VisionAgent should write preprocessed images to disk.
    cleaned_image_dir : str | None
        Output directory for cleaned images. None = sibling 'cleaned/' folder.
    max_structuring_retries : int
        How many Pydantic validation retry attempts the StructuringAgent makes.
    skip_vision_preprocessing : bool
        Pass raw image directly to VLM without OpenCV preprocessing.
        Useful for already-clean digital documents (e.g. exported PDFs).
    """

    def __init__(
        self,
        vision_model:              str   = DEFAULT_VISION_MODEL,
        structuring_model:         str   = DEFAULT_STRUCTURING_MODEL,
        ollama_host:               str   = DEFAULT_OLLAMA_HOST,
        save_cleaned_images:       bool  = True,
        cleaned_image_dir:         Optional[str | Path] = None,
        max_structuring_retries:   int   = 3,
        skip_vision_preprocessing: bool  = False,
        use_easyocr:               bool  = True,
    ) -> None:
        self.skip_vision_preprocessing = skip_vision_preprocessing

        logger.info("[Pipeline] Initializing agents...")

        if use_easyocr:
            from ai_agents.easyocr_agent import EasyOCRAgent
            self._vision = EasyOCRAgent(
                use_gpu            = True,
                save_cleaned       = save_cleaned_images,
                cleaned_output_dir = cleaned_image_dir,
            )
            logger.info("[Pipeline] OCR engine: EasyOCR (GPU)")
        else:
            self._vision = VisionAgent(
                model              = vision_model,
                ollama_host        = ollama_host,
                save_cleaned       = save_cleaned_images,
                cleaned_output_dir = cleaned_image_dir,
            )
            logger.info(f"[Pipeline] OCR engine: llava ({vision_model})")

        self._structuring = StructuringAgent(
            model        = structuring_model,
            ollama_host  = ollama_host,
            max_retries  = max_structuring_retries,
        )

        logger.info(
            f"[Pipeline] Ready  |  LLM={structuring_model}  host={ollama_host}"
        )

    # ── Single image ─────────────────────────────────────────────────────────

    def process(self, image_path: str | Path) -> MedicalReportPayload:
        """
        Process a single scanned medical report image.

        Parameters
        ----------
        image_path : str | Path
            Absolute or relative path to the JPEG / PNG scan.

        Returns
        -------
        MedicalReportPayload
            Fully validated structured payload, or best-effort partial.

        Raises
        ------
        FileNotFoundError  if the image does not exist on disk.
        ConnectionError    if Ollama is not running (raised by agents).
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"[Pipeline] Image not found: {image_path}")

        pipeline_start = time.perf_counter()
        logger.info(f"[Pipeline] ▶  Processing: {image_path.name}")

        # ── Stage 1: Vision (OCR) ────────────────────────────────────────────
        logger.info("[Pipeline] Stage 1/2 — Vision Agent (preprocessing + OCR)…")
        vision_result: VisionAgentResult = self._vision.process(
            image_path,
            skip_preprocessing=self.skip_vision_preprocessing,
        )

        self._log_vision_result(vision_result)

        # Low-confidence advisory (we still proceed — soft-fail philosophy)
        if vision_result.confidence < LOW_CONFIDENCE_THRESHOLD:
            logger.warning(
                f"[Pipeline] ⚠ Low OCR confidence ({vision_result.confidence:.2f}). "
                f"The image may be blurry, poorly lit, or heavily handwritten. "
                f"Proceeding to structuring — V-JEPA will impute missing fields."
            )

        # ── Stage 2: Structuring (JSON extraction + Pydantic validation) ─────
        logger.info("[Pipeline] Stage 2/2 — Structuring Agent (JSON extraction)…")
        payload: MedicalReportPayload = self._structuring.structure(
            raw_ocr_text        = vision_result.raw_text,
            ocr_confidence      = vision_result.confidence,
            preprocessing_steps = vision_result.preprocessing_steps,
            deskew_angle_deg    = vision_result.deskew_angle_deg,
            source_image_path   = str(image_path),
            vision_model        = vision_result.model_used,
        )

        # ── Merge vision warnings into payload metadata ───────────────────────
        if vision_result.warnings:
            existing = payload.metadata.extraction_warnings or []
            payload.metadata.extraction_warnings = existing + [
                f"[vision] {w}" for w in vision_result.warnings
            ]

        total_elapsed = time.perf_counter() - pipeline_start
        logger.info(
            f"[Pipeline] ■  Done in {total_elapsed:.1f}s  |  "
            f"report_type={payload.report_type.value}  |  "
            f"ocr_confidence={payload.metadata.ocr_confidence:.2f}  |  "
            f"lab_panels={len(payload.lab_panels)}  |  "
            f"diagnoses={len(payload.diagnoses)}  |  "
            f"warnings={len(payload.metadata.extraction_warnings)}"
        )

        return payload

    # ── Batch processing ─────────────────────────────────────────────────────

    def process_batch(
        self,
        image_paths:  List[str | Path],
        output_dir:   Optional[str | Path] = None,
        fail_fast:    bool = False,
    ) -> List[MedicalReportPayload]:
        """
        Process a list of image files sequentially.

        Parameters
        ----------
        image_paths : list of image paths to process.
        output_dir  : if provided, each payload is saved as a JSON file here.
        fail_fast   : if True, stop on first error; else collect and continue.

        Returns
        -------
        List of MedicalReportPayload — one per successfully processed image.
        """
        results:  List[MedicalReportPayload] = []
        failures: List[str]                  = []

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        total = len(image_paths)
        for idx, path in enumerate(image_paths, 1):
            logger.info(f"[Pipeline] Batch {idx}/{total}: {Path(path).name}")
            try:
                payload = self.process(path)
                results.append(payload)

                if output_dir:
                    out_stem = Path(path).stem
                    out_file = Path(output_dir) / f"{out_stem}_payload.json"
                    out_file.write_text(
                        payload.model_dump_json(indent=2), encoding="utf-8"
                    )
                    logger.info(f"[Pipeline] Saved → {out_file}")

            except Exception as e:
                logger.error(f"[Pipeline] FAILED: {path}  —  {e}")
                failures.append(f"{path}: {e}")
                if fail_fast:
                    raise

        if failures:
            logger.warning(
                f"[Pipeline] Batch complete with {len(failures)} failure(s):\n"
                + "\n".join(f"  • {f}" for f in failures)
            )

        logger.info(
            f"[Pipeline] Batch finished. "
            f"Success: {len(results)}/{total}. Failures: {len(failures)}/{total}."
        )
        return results

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _log_vision_result(r: VisionAgentResult) -> None:
        logger.info(
            f"[Pipeline]   OCR confidence : {r.confidence:.2f}\n"
            f"[Pipeline]   Deskew angle   : {r.deskew_angle_deg}°\n"
            f"[Pipeline]   Preprocessing  : {r.preprocessing_steps}\n"
            f"[Pipeline]   Text length    : {len(r.raw_text)} chars\n"
            f"[Pipeline]   VLM elapsed    : {r.elapsed_seconds:.1f}s"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Medical Report Pipeline: scan image → structured JSON payload.\n"
            "Requires: ollama serve  (with llama3.2-vision and llama3 pulled)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "images", nargs="+", type=str,
        help="One or more image paths to process.",
    )
    p.add_argument(
        "--out", "-o", type=str, default=None,
        help="Output directory for JSON files. Prints to stdout if omitted.",
    )
    p.add_argument(
        "--vision-model", type=str, default=DEFAULT_VISION_MODEL,
        help=f"Ollama VLM model tag (default: {DEFAULT_VISION_MODEL}).",
    )
    p.add_argument(
        "--structuring-model", type=str, default=DEFAULT_STRUCTURING_MODEL,
        help=f"Ollama LLM model tag (default: {DEFAULT_STRUCTURING_MODEL}).",
    )
    p.add_argument(
        "--host", type=str, default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama host URL (default: {DEFAULT_OLLAMA_HOST}).",
    )
    p.add_argument(
        "--no-preprocess", action="store_true",
        help="Skip OpenCV preprocessing (use for already-clean digital PDFs).",
    )
    p.add_argument(
        "--no-save-cleaned", action="store_true",
        help="Do not write preprocessed images to disk.",
    )
    p.add_argument(
        "--fail-fast", action="store_true",
        help="Stop batch on first error.",
    )
    p.add_argument(
        "--retries", type=int, default=3,
        help="Max structuring retry attempts (default: 3).",
    )
    return p


def main() -> None:
    args = _build_cli().parse_args()

    pipeline = MedicalReportPipeline(
        vision_model              = args.vision_model,
        structuring_model         = args.structuring_model,
        ollama_host               = args.host,
        save_cleaned_images       = not args.no_save_cleaned,
        max_structuring_retries   = args.retries,
        skip_vision_preprocessing = args.no_preprocess,
    )

    image_paths = args.images

    if len(image_paths) == 1:
        # Single-image path: always print to stdout (+ optionally save)
        payload = pipeline.process(image_paths[0])
        json_str = payload.model_dump_json(indent=2)

        if args.out:
            out_path = Path(args.out)
            if out_path.is_dir():
                out_file = out_path / f"{Path(image_paths[0]).stem}_payload.json"
            else:
                out_file = out_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json_str, encoding="utf-8")
            print(f"\nSaved → {out_file}")
        else:
            print("\n" + "=" * 60)
            print("  MedicalReportPayload")
            print("=" * 60)
            print(json_str)

    else:
        # Batch path
        pipeline.process_batch(
            image_paths = image_paths,
            output_dir  = args.out,
            fail_fast   = args.fail_fast,
        )


if __name__ == "__main__":
    main()
