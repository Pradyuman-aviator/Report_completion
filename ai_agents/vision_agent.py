"""
ai_agents/vision_agent.py
==========================
Vision Agent — Image Preprocessing + Local VLM OCR
----------------------------------------------------

Pipeline:
    Raw JPEG/PNG (from Android camera)
        ↓
    [1] OpenCV Preprocessing
        • Grayscale conversion
        • CLAHE contrast enhancement (better than plain histogram eq.)
        • Bilateral denoising (preserves text edges)
        • Adaptive thresholding (handles uneven scan lighting)
        • Deskewing (detects and corrects page rotation via Hough transform)
        • Border cropping (removes camera vignette / dark margins)
        ↓
    [2] Base64-encode cleaned image
        ↓
    [3] Send to local Ollama multimodal model (LLaVA or Llama 3.2-Vision)
        • 100 % on-device, zero cloud calls
        • Model runs via the ollama Python client
        ↓
    VisionAgentResult
        • raw_text: str          — verbatim OCR extraction
        • confidence: float      — model self-reported quality (0.0–1.0)
        • preprocessing_steps    — audit trail of applied transforms
        • deskew_angle_deg       — detected rotation angle
        • cleaned_image_path     — path to the saved preprocessed image

Ollama setup (run once in terminal before using this agent):
    ollama pull llama3.2-vision
    ollama serve   # starts the local inference server on localhost:11434

Requirements:
    pip install opencv-python-headless pillow ollama

Usage:
    >>> from ai_agents.vision_agent import VisionAgent
    >>> agent = VisionAgent()
    >>> result = agent.process("scans/report_001.jpg")
    >>> print(result.raw_text)
"""

from __future__ import annotations

import base64
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Result Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VisionAgentResult:
    """
    Returned by VisionAgent.process().

    Consumed directly by structuring_agent.py.
    """
    raw_text:            str          # Verbatim OCR text from the VLM
    confidence:          float        # 0.0–1.0, parsed from model response
    preprocessing_steps: List[str]   = field(default_factory=list)
    deskew_angle_deg:    Optional[float] = None
    cleaned_image_path:  Optional[str]   = None
    source_image_path:   str          = ""
    model_used:          str          = "llama3.2-vision"
    elapsed_seconds:     float        = 0.0
    warnings:            List[str]    = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  OpenCV Preprocessing Functions
# ──────────────────────────────────────────────────────────────────────────────

def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to greyscale. Handles already-grey inputs."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Superior to plain cv2.equalizeHist() for medical scans because it operates
    on local tiles — compensating for the uneven lighting of a mobile phone
    camera photographing paper under fluorescent lights.

    clip_limit=2.0 and tile_grid_size=(8,8) are empirically good defaults
    for A4/Letter-size report photos at 300-600 DPI equivalent.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _denoise(gray: np.ndarray) -> np.ndarray:
    """
    Bilateral filter: smooths noise while preserving sharp text edges.

    h=10 governs filter strength; d=9 is the pixel neighbourhood diameter.
    Bilateral is slower than Gaussian but produces far fewer artefacts on
    fine text (especially printed in 8-9pt font, common in lab reports).
    """
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)


def _threshold(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding → binary (black text on white background).

    Adaptive (not global) thresholding handles:
      • Yellowed/aged paper
      • Shadow gradients from handheld photography
      • Light variations across a single A4 page

    block_size=31 — large enough to cover the biggest background variation.
    C=10           — subtracted constant to fine-tune binarisation boundary.
    """
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return binary


def _morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """
    Light morphological opening to remove salt-and-pepper noise pixels
    that survive thresholding, without eroding thin text strokes.
    kernel (1×1) is intentionally tiny.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _detect_skew_angle(binary: np.ndarray) -> float:
    """
    Detect the skew angle of a scanned page using the Probabilistic Hough
    Line Transform on Canny edges.

    Returns the dominant angle in degrees (positive = clockwise tilt).
    Returns 0.0 if no reliable lines are found.

    Why Hough over minAreaRect:
      • minAreaRect on all contours is confused by complex report layouts
        (tables, seals, stamps).
      • Hough picks up the dominant horizontal/vertical ruled lines that most
        lab report templates print, giving a reliable reference angle.
    """
    # Work on a down-scaled copy for speed
    scale   = 0.5
    small   = cv2.resize(binary, None, fx=scale, fy=scale)
    edges   = cv2.Canny(small, threshold1=50, threshold2=150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=small.shape[1] // 4,   # at least 25 % of image width
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        logger.debug("Skew detection: no lines found, assuming 0°.")
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue   # vertical line — skip
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        angle_deg = math.degrees(angle_rad)
        # Only consider near-horizontal lines (±15°) as reference
        if abs(angle_deg) <= 15.0:
            angles.append(angle_deg)

    if not angles:
        return 0.0

    # Robust median angle
    median_angle = float(np.median(angles))
    logger.debug(f"Skew detection: median angle = {median_angle:.2f}°")
    return median_angle


def _deskew(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate the image to correct detected skew.

    Uses warpAffine with BORDER_REPLICATE to avoid black corners that would
    confuse the VLM into thinking there is content at the edges.

    Skips rotation for angles smaller than 0.3° (within typical phone capture
    tolerance — correcting tiny angles introduces more interpolation artefacts
    than it removes).
    """
    if abs(angle_deg) < 0.3:
        return image

    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, angle_deg, scale=1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _crop_borders(image: np.ndarray, margin_px: int = 10) -> np.ndarray:
    """
    Remove a fixed-pixel border from all sides.

    Phone photos frequently have dark vignette corners that the VLM may
    misinterpret as content. A 10-pixel safe margin is minimal and risk-free.
    """
    h, w = image.shape[:2]
    m = margin_px
    if h <= 2 * m or w <= 2 * m:
        return image   # image too small to crop safely
    return image[m:h - m, m:w - m]


def _auto_crop_content(
    binary: np.ndarray,
    padding: int = 20,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Automatically detect the document bounding box and crop to content.

    Finds the largest contour (the document/paper boundary inside a
    darker background) and crops to its bounding rect.

    Returns:
        cropped image and (x, y, w, h) bounding box relative to input.
    """
    # Invert binary so the document is white-on-black for contour detection
    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(
        inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return binary, (0, 0, binary.shape[1], binary.shape[0])

    # Pick the largest contour by area (= the document)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add safety padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    x2 = min(binary.shape[1], x + w + 2 * padding)
    y2 = min(binary.shape[0], y + h + 2 * padding)

    return binary[y:y2, x:x2], (x, y, x2 - x, y2 - y)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Full Preprocessing Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(
    image_path:   str | Path,
    save_cleaned: bool = True,
    output_dir:   Optional[str | Path] = None,
) -> Tuple[np.ndarray, List[str], Optional[float], Optional[str]]:
    """
    Apply the full OpenCV preprocessing pipeline to a raw medical scan photo.

    Parameters
    ----------
    image_path    : Path to the source JPEG/PNG image.
    save_cleaned  : If True, write the preprocessed image to `output_dir`.
    output_dir    : Where to save cleaned images. Defaults to same dir as source.

    Returns
    -------
    cleaned_img       : np.ndarray (H, W) — binary, ready for VLM
    steps_applied     : list of step name strings (audit log)
    deskew_angle_deg  : detected rotation angle (None if step skipped)
    cleaned_img_path  : path of saved cleaned image (None if not saved)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    steps: List[str] = []
    angle: Optional[float] = None

    # ── Step 1: Grayscale ────────────────────────────────────────────────────
    gray = _to_grayscale(img)
    steps.append("grayscale")

    # ── Step 2: CLAHE contrast enhancement ──────────────────────────────────
    gray = _enhance_contrast(gray)
    steps.append("clahe_contrast")

    # ── Step 3: Bilateral denoise ────────────────────────────────────────────
    denoised = _denoise(gray)
    steps.append("bilateral_denoise")

    # ── Step 4: Adaptive threshold → binary ─────────────────────────────────
    binary = _threshold(denoised)
    steps.append("adaptive_threshold")

    # ── Step 5: Morphological cleanup ───────────────────────────────────────
    binary = _morphological_cleanup(binary)
    steps.append("morph_open")

    # ── Step 6: Skew detection & deskew ─────────────────────────────────────
    angle = _detect_skew_angle(binary)
    binary = _deskew(binary, angle)
    steps.append(f"deskew({angle:.2f}deg)")

    # ── Step 7: Border crop ──────────────────────────────────────────────────
    binary = _crop_borders(binary, margin_px=10)
    steps.append("border_crop")

    # ── Optional: Auto content-crop ──────────────────────────────────────────
    # Disabled by default — enable when scanning isolated reports on a desk
    # binary, bbox = _auto_crop_content(binary)
    # steps.append("auto_content_crop")

    # ── Save cleaned image ───────────────────────────────────────────────────
    cleaned_path: Optional[str] = None
    if save_cleaned:
        out_dir = Path(output_dir) if output_dir else image_path.parent / "cleaned"
        out_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = str(out_dir / f"{image_path.stem}_cleaned.png")
        cv2.imwrite(cleaned_path, binary)
        logger.info(f"Cleaned image saved → {cleaned_path}")

    logger.info(f"Preprocessing complete. Steps: {steps}  |  Skew: {angle:.2f}°")
    return binary, steps, angle, cleaned_path


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Image Encoding for Ollama
# ──────────────────────────────────────────────────────────────────────────────

def _ndarray_to_base64_png(img: np.ndarray) -> str:
    """
    Encode a numpy array (OpenCV image) to a base64-encoded PNG string.
    Ollama's Python client accepts images as base64 strings in the 'images' field.
    """
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("cv2.imencode failed to encode preprocessed image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _load_raw_base64(image_path: str | Path) -> str:
    """Load a raw image file and base64-encode it (no preprocessing)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  VLM Prompt
# ──────────────────────────────────────────────────────────────────────────────

_OCR_PROMPT = """You are a highly accurate medical document OCR system.
Your task is to extract ALL text from the medical report image provided.

Instructions:
1. Transcribe EVERY piece of text you can see, preserving the original layout
   as closely as possible. Use tabs or spaces to represent table structure.
2. Do NOT interpret, summarise, or add any text not present in the image.
3. If a section is partially obscured or illegible, write [ILLEGIBLE] in place
   of the unreadable portion.
4. Preserve all numbers, units, reference ranges, and date formats EXACTLY
   as they appear — do not convert units or reformat dates.
5. At the very end of your response, on a new line, write:
   CONFIDENCE: <a number from 0.0 to 1.0>
   where 1.0 means you could read every character clearly and 0.0 means
   the image was entirely unreadable. Be honest.

Begin transcription now:"""


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Confidence Parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_confidence(raw_text: str) -> Tuple[str, float]:
    """
    Extract the CONFIDENCE score appended by the model.

    Returns:
        clean_text  — raw_text with the CONFIDENCE line removed
        confidence  — float in [0.0, 1.0]
    """
    pattern = re.compile(
        r"\bCONFIDENCE\s*[:\-=]\s*([0-9]*\.?[0-9]+)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(raw_text)
    if match:
        confidence = min(1.0, max(0.0, float(match.group(1))))
        clean_text = pattern.sub("", raw_text).rstrip()
    else:
        confidence = 0.5   # unknown; assume mediocre quality
        clean_text = raw_text

    return clean_text.strip(), confidence


# ──────────────────────────────────────────────────────────────────────────────
# 6.  VisionAgent Class
# ──────────────────────────────────────────────────────────────────────────────

class VisionAgent:
    """
    Orchestrates image preprocessing and local VLM OCR extraction.

    All inference is handled by a locally running Ollama server.
    Zero data leaves the device.

    Parameters
    ----------
    model : str
        Ollama model tag for the multimodal vision-language model.
        Recommended: "llama3.2-vision"  (Meta's on-device VLM)
        Alternative: "llava:13b"        (LLaVA 1.6 / 13B, heavier)
    ollama_host : str
        Base URL of the local Ollama API. Default: http://localhost:11434
    save_cleaned : bool
        Whether to write the preprocessed image to disk (useful for debugging).
    cleaned_output_dir : str | None
        Directory for cleaned image output. Defaults to a 'cleaned/' subdirectory
        next to each source image.
    max_retries : int
        Number of Ollama call retries on network/timeout error.
    """

    def __init__(
        self,
        model:               str  = "llama3.2-vision",
        ollama_host:         str  = "http://localhost:11434",
        save_cleaned:        bool = True,
        cleaned_output_dir:  Optional[str | Path] = None,
        max_retries:         int  = 2,
    ) -> None:
        self.model              = model
        self.ollama_host        = ollama_host
        self.save_cleaned       = save_cleaned
        self.cleaned_output_dir = cleaned_output_dir
        self.max_retries        = max_retries

        # Configure the ollama client to point at our local server
        try:
            import ollama as _ollama
        except ImportError:
            raise ImportError(
                "The 'ollama' package is required. Install it with:\n"
                "  pip install ollama\n"
                "Then make sure 'ollama serve' is running."
            )
        self._client = _ollama.Client(host=ollama_host)
        self._verify_ollama_connection()

    def _verify_ollama_connection(self) -> None:
        """Fail fast if Ollama is not running or model is not pulled."""
        try:
            models = self._client.list()
            # SDK v0.4+ returns objects with .model attr; older returns dicts with "name"
            raw_list = models.get("models", []) if isinstance(models, dict) else getattr(models, "models", [])
            available = []
            for m in raw_list:
                if isinstance(m, dict):
                    available.append(m.get("name") or m.get("model", ""))
                else:
                    available.append(getattr(m, "model", getattr(m, "name", "")))
            if not any(self.model in m for m in available):
                logger.warning(
                    f"[VisionAgent] Model '{self.model}' not found in Ollama.\n"
                    f"  Available: {available}\n"
                    f"  Run:  ollama pull {self.model}"
                )
        except Exception as e:
            raise ConnectionError(
                f"[VisionAgent] Cannot connect to Ollama at {self.ollama_host}.\n"
                f"  Make sure Ollama is running:  ollama serve\n"
                f"  Error: {e}"
            )

    # ── Core Processing ─────────────────────────────────────────────────────

    def process(
        self,
        image_path: str | Path,
        skip_preprocessing: bool = False,
    ) -> VisionAgentResult:
        """
        Full pipeline: preprocess → base64 encode → VLM OCR → parse.

        Parameters
        ----------
        image_path : path to the raw scanned medical report image.
        skip_preprocessing : if True, sends the raw image directly to the VLM
                             (useful for already-clean digital PDFs).

        Returns
        -------
        VisionAgentResult
        """
        image_path = Path(image_path)
        t_start = time.perf_counter()
        warnings: List[str] = []
        steps: List[str] = []
        angle: Optional[float] = None
        cleaned_path: Optional[str] = None

        # ── 1. Preprocessing ────────────────────────────────────────────────
        if skip_preprocessing:
            logger.info("[VisionAgent] Skipping preprocessing (skip_preprocessing=True).")
            image_b64 = _load_raw_base64(image_path)
            steps = ["raw_passthrough"]
        else:
            logger.info(f"[VisionAgent] Preprocessing: {image_path.name}")
            try:
                cleaned_img, steps, angle, cleaned_path = preprocess_image(
                    image_path,
                    save_cleaned=self.save_cleaned,
                    output_dir=self.cleaned_output_dir,
                )
                image_b64 = _ndarray_to_base64_png(cleaned_img)
            except Exception as e:
                warnings.append(f"Preprocessing failed ({e}). Falling back to raw image.")
                logger.warning(f"[VisionAgent] Preprocessing error: {e}. Using raw image.")
                image_b64 = _load_raw_base64(image_path)
                steps = ["raw_fallback"]

        # ── 2. VLM OCR via Ollama ────────────────────────────────────────────
        logger.info(f"[VisionAgent] Sending to Ollama model: {self.model}")
        raw_response = self._call_ollama_with_retry(image_b64, warnings)

        # ── 3. Parse confidence score ────────────────────────────────────────
        clean_text, confidence = _parse_confidence(raw_response)

        if confidence < 0.4:
            warnings.append(
                f"Low OCR confidence ({confidence:.2f}). "
                "Consider rephotographing under better lighting."
            )

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"[VisionAgent] Done in {elapsed:.1f}s  |  "
            f"confidence={confidence:.2f}  |  "
            f"text_len={len(clean_text)} chars"
        )

        return VisionAgentResult(
            raw_text            = clean_text,
            confidence          = confidence,
            preprocessing_steps = steps,
            deskew_angle_deg    = angle,
            cleaned_image_path  = cleaned_path,
            source_image_path   = str(image_path),
            model_used          = self.model,
            elapsed_seconds     = elapsed,
            warnings            = warnings,
        )

    def _call_ollama_with_retry(
        self,
        image_b64: str,
        warnings:  List[str],
    ) -> str:
        """
        Call the Ollama API with retry logic.

        Ollama on Apple Silicon can occasionally stall on the first inference
        call after a cold start. Retrying once is usually sufficient.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):   # +2 = initial + retries
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": _OCR_PROMPT,
                            "images": [image_b64],
                        }
                    ],
                    options={
                        "temperature": 0.0,    # deterministic OCR — no creativity
                        "num_predict": 4096,   # max tokens for long reports
                    },
                )
                return response["message"]["content"]
            except Exception as e:
                last_exc = e
                wait_s   = 2 ** attempt    # exponential back-off: 2s, 4s, 8s …
                logger.warning(
                    f"[VisionAgent] Ollama call failed (attempt {attempt}): {e}. "
                    f"Retrying in {wait_s}s…"
                )
                warnings.append(f"Ollama retry {attempt}: {e}")
                time.sleep(wait_s)

        raise RuntimeError(
            f"[VisionAgent] Ollama call failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_exc}"
        )

    # ── Convenience Batch Method ─────────────────────────────────────────────

    def process_batch(
        self,
        image_paths: List[str | Path],
    ) -> List[VisionAgentResult]:
        """
        Process a list of images sequentially.

        Sequential (not parallel) because Ollama is already using all available
        Neural Engine / GPU cores for each inference call on Apple Silicon.
        Parallelising would thrash unified memory.
        """
        results = []
        for i, path in enumerate(image_paths, 1):
            logger.info(f"[VisionAgent] Batch {i}/{len(image_paths)}: {Path(path).name}")
            results.append(self.process(path))
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Smoke Test  (python -m ai_agents.vision_agent <image_path>)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ai_agents.vision_agent <path/to/report.jpg>")
        print("\nRunning preprocessing-only demo (no Ollama call)...")

        # Generate a synthetic test image (white page with text)
        demo = np.ones((800, 600), dtype=np.uint8) * 240
        cv2.putText(demo, "PATIENT: Test Kumar", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        cv2.putText(demo, "Haemoglobin: 11.2 g/dL  (L)", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
        cv2.imwrite("/tmp/demo_report.jpg", demo)

        cleaned, steps, angle, path = preprocess_image(
            "/tmp/demo_report.jpg",
            save_cleaned=True,
            output_dir="/tmp/",
        )
        print(f"\nPreprocessing steps : {steps}")
        print(f"Deskew angle        : {angle:.2f}°")
        print(f"Cleaned image saved : {path}")
        print(f"Output shape        : {cleaned.shape}")
        sys.exit(0)

    # Full run (requires: ollama serve && ollama pull llama3.2-vision)
    agent  = VisionAgent(model="llama3.2-vision")
    result = agent.process(sys.argv[1])

    print("\n" + "=" * 60)
    print("  VisionAgent Result")
    print("=" * 60)
    print(f"Model used    : {result.model_used}")
    print(f"Elapsed       : {result.elapsed_seconds:.1f}s")
    print(f"Confidence    : {result.confidence:.2f}")
    print(f"Skew angle    : {result.deskew_angle_deg}°")
    print(f"Steps         : {result.preprocessing_steps}")
    if result.warnings:
        print(f"Warnings      : {result.warnings}")
    print(f"\n{'─'*60}\nExtracted Text:\n{'─'*60}")
    print(result.raw_text[:2000])   # Print first 2000 chars
