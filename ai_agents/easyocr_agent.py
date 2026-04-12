"""
ai_agents/easyocr_agent.py  v1.0
===========================
EasyOCR-based Vision Agent — drop-in replacement for llava:7b
--------------------------------------------------------------

Uses EasyOCR (CRAFT text detector + CRNN recognizer) on GPU (RTX 4050)
instead of a VLM, which:
  • Actually reads the text rather than hallucinating it
  • Runs in ~3-5s vs 110s for llava:7b
  • Achieves 0.85-0.95 confidence on clean printed lab reports

Pipeline:
    Image
      ↓
    OpenCV Preprocessing
      (grayscale → CLAHE → bilateral denoise → adaptive threshold → deskew)
      ↓
    EasyOCR (GPU)
      ↓
    VisionAgentResult (same schema as before — fully compatible)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ai_agents.vision_agent import preprocess_image, VisionAgentResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Preprocessing helpers (enhanced for lab report tables)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_for_ocr(image_path: Path) -> tuple[np.ndarray, List[str], Optional[float]]:
    """
    Enhanced preprocessing tuned for dense printed lab report tables.
    Returns (processed_ndarray, steps_list, deskew_angle).
    """
    steps: List[str] = []

    # Load
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 1. Upscale if too small — EasyOCR works best at ≥150 DPI equivalent
    h, w = img.shape[:2]
    if max(h, w) < 1500:
        scale = 1500 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        steps.append(f"upscale_{scale:.1f}x")

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps.append("grayscale")

    # 3. CLAHE — boosts local contrast (critical for faded prints)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    steps.append("clahe")

    # 4. Bilateral denoise — removes noise while preserving text edges
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    steps.append("bilateral_denoise")

    # 5. Deskew
    angle = _detect_skew(gray)
    if abs(angle) > 0.3:
        gray = _rotate(gray, angle)
        steps.append(f"deskew_{angle:.1f}deg")
    else:
        angle = 0.0

    # 6. Adaptive threshold — binarise for clean black-on-white
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    steps.append("adaptive_threshold")

    # 7. Morphological opening — remove tiny speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    steps.append("morph_open")

    # Convert back to 3-channel for EasyOCR (it accepts both, but BGR is safer)
    out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return out, steps, float(angle)


def _detect_skew(gray: np.ndarray) -> float:
    """Detect document skew angle via Hough line transform."""
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=gray.shape[1] // 4, maxLineGap=20)
        if lines is None:
            return 0.0
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 20:   # ignore nearly-vertical lines
                angles.append(angle)
        return float(np.median(angles)) if angles else 0.0
    except Exception:
        return 0.0


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  EasyOCR Agent
# ──────────────────────────────────────────────────────────────────────────────

class EasyOCRAgent:
    """
    Drop-in replacement for VisionAgent.
    Uses EasyOCR on GPU instead of llava:7b via Ollama.

    Parameters
    ----------
    use_gpu : bool
        Use CUDA GPU (RTX 4050). Falls back to CPU if CUDA not available.
    languages : list
        EasyOCR language codes. ['en'] is sufficient for Indian lab reports.
    save_cleaned : bool
        Save the preprocessed image for audit.
    cleaned_output_dir : str | Path | None
        Where to save cleaned images.
    """

    def __init__(
        self,
        use_gpu:            bool = True,
        languages:          List[str] = ["en"],
        save_cleaned:       bool = True,
        cleaned_output_dir: Optional[str | Path] = None,
    ) -> None:
        self.use_gpu            = use_gpu
        self.languages          = languages
        self.save_cleaned       = save_cleaned
        self.cleaned_output_dir = cleaned_output_dir
        self._reader            = None   # lazy init — model loads on first use

        logger.info(
            f"[EasyOCRAgent] Initialized  |  GPU={use_gpu}  |  "
            f"languages={languages}"
        )

    def _get_reader(self):
        """Lazy-load EasyOCR reader (warm-up takes ~5s, then fast)."""
        if self._reader is None:
            import easyocr
            logger.info("[EasyOCRAgent] Loading EasyOCR model (first run ~5s)...")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False,
            )
            logger.info("[EasyOCRAgent] EasyOCR model loaded.")
        return self._reader

    def process(
        self,
        image_path: str | Path,
        skip_preprocessing: bool = False,
    ) -> VisionAgentResult:
        """
        Process a single medical report image.
        Returns VisionAgentResult — same schema as VisionAgent.
        """
        image_path = Path(image_path)
        t_start    = time.perf_counter()
        warnings: List[str] = []
        steps:    List[str] = []
        angle:    Optional[float] = None
        cleaned_path: Optional[str] = None

        # ── 1. Preprocessing ─────────────────────────────────────────────────
        if skip_preprocessing:
            img = cv2.imread(str(image_path))
            steps = ["raw_passthrough"]
            angle = 0.0
        else:
            logger.info(f"[EasyOCRAgent] Preprocessing: {image_path.name}")
            try:
                img, steps, angle = preprocess_for_ocr(image_path)
            except Exception as e:
                warnings.append(f"Preprocessing failed: {e}. Using raw image.")
                logger.warning(f"[EasyOCRAgent] Preprocessing error: {e}")
                img   = cv2.imread(str(image_path))
                steps = ["raw_fallback"]
                angle = 0.0

        # Save cleaned image for audit
        if self.save_cleaned and img is not None:
            out_dir = Path(self.cleaned_output_dir) if self.cleaned_output_dir \
                      else image_path.parent / "cleaned"
            out_dir.mkdir(parents=True, exist_ok=True)
            cleaned_path = str(out_dir / f"{image_path.stem}_cleaned.png")
            cv2.imwrite(cleaned_path, img)

        # ── 2. EasyOCR ───────────────────────────────────────────────────────
        logger.info("[EasyOCRAgent] Running EasyOCR...")
        reader  = self._get_reader()

        # EasyOCR accepts numpy array directly
        results = reader.readtext(
            img,
            detail=1,          # return bounding boxes + confidence
            paragraph=False,   # keep individual text boxes for accuracy
            width_ths=0.7,     # merge horizontally within rows
        )

        # ── 3. Reconstruct text preserving table layout ───────────────────────
        raw_text   = self._results_to_text(results)
        confidence = self._mean_confidence(results)

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"[EasyOCRAgent] Done in {elapsed:.1f}s  |  "
            f"confidence={confidence:.2f}  |  "
            f"text_len={len(raw_text)} chars  |  "
            f"boxes={len(results)}"
        )

        if confidence < 0.5:
            warnings.append(
                f"Low OCR confidence ({confidence:.2f}). "
                "Image may be blurry or poorly lit."
            )

        return VisionAgentResult(
            raw_text            = raw_text,
            confidence          = confidence,
            preprocessing_steps = steps,
            deskew_angle_deg    = angle,
            cleaned_image_path  = cleaned_path,
            source_image_path   = str(image_path),
            model_used          = "easyocr",
            elapsed_seconds     = elapsed,
            warnings            = warnings,
        )

    @staticmethod
    def _results_to_text(results: list) -> str:
        """
        Convert EasyOCR result list to structured plain text.
        Sorts bounding boxes top-to-bottom, left-to-right to preserve
        the table reading order of lab reports.
        """
        if not results:
            return ""

        # Each result: (bbox, text, confidence)
        # bbox = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        # Sort by Y centroid (row), then X centroid (column)
        def sort_key(r):
            bbox = r[0]
            ys   = [pt[1] for pt in bbox]
            xs   = [pt[0] for pt in bbox]
            return (round(sum(ys) / len(ys) / 20) * 20,   # row bucket (20px)
                    sum(xs) / len(xs))

        sorted_results = sorted(results, key=sort_key)

        # Group into rows (boxes within 20px of each other vertically)
        rows: List[List[str]] = []
        current_row: List[tuple] = []
        last_y = None

        for bbox, text, _ in sorted_results:
            ys     = [pt[1] for pt in bbox]
            cy     = sum(ys) / len(ys)
            cx     = sum(pt[0] for pt in bbox) / len(bbox)

            if last_y is None or abs(cy - last_y) < 25:
                current_row.append((cx, text))
            else:
                rows.append([t for _, t in sorted(current_row)])
                current_row = [(cx, text)]
            last_y = cy

        if current_row:
            rows.append([t for _, t in sorted(current_row)])

        # Join rows, separate columns with tab-like spacing
        return "\n".join("   ".join(row) for row in rows)

    @staticmethod
    def _mean_confidence(results: list) -> float:
        if not results:
            return 0.0
        confs = [r[2] for r in results if len(r) == 3]
        return float(sum(confs) / len(confs)) if confs else 0.0
