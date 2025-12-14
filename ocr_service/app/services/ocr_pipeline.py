# NOTE: paste your working code here:
# - OCRLine dataclass
# - global PADDLE_OCR init
# - preprocess_variants, deskew, etc.
# - extract_fields_v2
# - process_image_fast(image_path) returning dict

# Keep the API as:
# def process_image_fast(image_path: str) -> dict:
#     ...
#     return result_dict







# app/services/ocr_pipeline.py

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------
# Environment tweaks (avoid slow/fragile source checks)
# ---------------------------------------------------------------------
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

# ---------------------------------------------------------------------
# OCR model init (global, created once per process)
# ---------------------------------------------------------------------
from paddleocr import PaddleOCR  # noqa: E402

# NOTE: PaddleOCR 3.x: use_textline_orientation is preferred
PADDLE_OCR = PaddleOCR(use_textline_orientation=True, lang="en")


# ---------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------
@dataclass
class OCRLine:
    text: str
    conf: float  # 0..1
    box: Optional[Any] = None  # quad/poly or None


# ---------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------
def to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def resize_max_side(bgr: np.ndarray, max_side: int = 1100) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / m
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_variants(bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Return a few useful grayscale/binary variants.
    (We generally OCR on orig/contrast; threshold variants are fallbacks.)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # light denoise
    den = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(den)

    # adaptive threshold
    th_adapt = cv2.adaptiveThreshold(
        contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )

    # otsu threshold
    _, th_otsu = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        "gray": gray,
        "contrast": contrast,
        "th_adapt": th_adapt,
        "th_otsu": th_otsu,
    }


def estimate_skew_angle_from_binary(binary_img: np.ndarray) -> float:
    """
    Estimate skew angle using minAreaRect on foreground pixels.
    Input should be binary with white background & dark text.
    """
    inv = 255 - binary_img
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) < 50:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Convert OpenCV angle to rotation angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    return float(angle)


def rotate_image(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(
        bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def deskew(binary_img: np.ndarray) -> np.ndarray:
    angle = estimate_skew_angle_from_binary(binary_img)
    if abs(angle) < 0.05:
        return binary_img

    (h, w) = binary_img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        binary_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


# ---------------------------------------------------------------------
# OCR runner (safe parsing; never crash on weird outputs)
# ---------------------------------------------------------------------
def _ensure_bgr_uint8(img: np.ndarray) -> np.ndarray:
    x = img
    if x.ndim == 2:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def run_paddle_ocr_safe(img_in: np.ndarray) -> List[OCRLine]:
    """
    Uses global PADDLE_OCR and prefers predict() if available.
    Returns OCRLine list or [] (does not throw parsing exceptions).
    """
    img = _ensure_bgr_uint8(img_in)
    lines: List[OCRLine] = []

    ocr = PADDLE_OCR

    # Prefer predict() for PaddleOCR 3.x
    if hasattr(ocr, "predict"):
        try:
            res = ocr.predict(img)
            parsed = res if isinstance(res, list) else [res]

            for r in parsed:
                if not isinstance(r, dict):
                    continue

                texts = r.get("rec_texts") or r.get("texts") or []
                scores = r.get("rec_scores") or r.get("scores") or []
                boxes = r.get("dt_polys") or r.get("boxes") or [None] * len(texts)

                if len(scores) != len(texts):
                    scores = list(scores) + [1.0] * (len(texts) - len(scores))

                for t, s, b in zip(texts, scores, boxes):
                    t = str(t).strip()
                    if not t:
                        continue
                    lines.append(OCRLine(text=t, conf=float(s), box=b))

            return lines
        except Exception:
            # fallback below
            pass

    # Fallback to older .ocr()
    if hasattr(ocr, "ocr"):
        try:
            res = ocr.ocr(img)
            # expected: [[ [box, (text, conf)], ... ]]
            for block in res or []:
                for item in block or []:
                    if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                        continue
                    box = item[0]
                    tconf = item[1]
                    if isinstance(tconf, (list, tuple)) and len(tconf) >= 2:
                        text, conf = tconf[0], tconf[1]
                    else:
                        continue
                    text = str(text).strip()
                    if not text:
                        continue
                    lines.append(OCRLine(text=text, conf=float(conf), box=box))
            return lines
        except Exception:
            return []

    return []


# ---------------------------------------------------------------------
# Extraction logic (v2): token-level code, robust title, date parsing
# ---------------------------------------------------------------------
DATE_RE = re.compile(r"\b(\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2})\b", re.I)
LABEL_RE = re.compile(r"^\s*(code|id|identifier)\s*[:\-]\s*", re.I)

CODE_PATTERNS = [
    re.compile(r"\b[A-Za-z]{1,6}\d{1,6}\b"),  # cd01, A1239
    re.compile(r"\b(?:INV|RX|LOT)[\-_]?[A-Za-z0-9]+\b", re.I),  # INV-123, RX_9999
    re.compile(r"\b[A-Za-z0-9]+[-_][A-Za-z0-9]+\b"),  # AA-12, RX_1234
    re.compile(r"\b[A-Za-z]{1,3}\d{1,4}\b"),  # w35, od04
]


def parse_date_to_iso(s: str) -> Optional[str]:
    s2 = s.replace(".", "/").replace("-", "/")
    try:
        dt = datetime.strptime(s2, "%Y/%m/%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _bbox_height(box: Any) -> float:
    """
    Works with list-of-points or numpy arrays. Returns 0 if unknown.
    """
    if box is None:
        return 0.0
    try:
        arr = np.array(box)
        if arr.ndim >= 2 and arr.shape[-1] >= 2:
            ys = arr[..., 1].astype(float).flatten()
            if ys.size > 0:
                return float(ys.max() - ys.min())
    except Exception:
        return 0.0
    return 0.0


def normalize_line_for_code(line: str) -> str:
    s = line.strip()
    s = LABEL_RE.sub("", s)          # remove "Code:" / "Identifier:" / "ID:"
    s = DATE_RE.sub(" ", s)          # remove date fragments
    s = re.sub(r"[,\;\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def code_token_candidates(line: str) -> List[str]:
    s = normalize_line_for_code(line)
    cands = set()

    for pat in CODE_PATTERNS:
        for m in pat.finditer(s):
            cands.add(m.group(0).strip())

    # token fallback: require at least one digit
    for tok in re.split(r"\s+", s):
        tok = tok.strip().strip(":.-_")
        if tok and any(ch.isdigit() for ch in tok):
            cands.add(tok)

    return list(cands)


def score_code_token(tok: str) -> float:
    t = tok.strip()
    if len(t) < 2 or len(t) > 25:
        return -999.0
    if DATE_RE.search(t):
        return -999.0
    if not any(ch.isdigit() for ch in t):
        return -50.0

    allowed = sum(ch.isalnum() or ch in "-_/" for ch in t)
    ratio = allowed / max(1, len(t))
    if ratio < 0.80:
        return -10.0

    score = 0.0
    if re.fullmatch(r"[A-Za-z]{1,6}\d{1,6}", t):
        score += 5.0
    if re.fullmatch(r"[A-Za-z0-9]+[-_][A-Za-z0-9]+", t):
        score += 3.0
    if re.match(r"^(INV|RX|LOT)", t, re.I):
        score += 2.0

    return score + ratio


def looks_like_code(line: str) -> bool:
    return len(code_token_candidates(line)) > 0


def extract_fields_v2(ocr_lines: List[OCRLine]) -> Dict[str, Any]:
    # normalize
    lines = []
    for ln in ocr_lines:
        text = (ln.text or "").strip()
        if not text:
            continue
        conf = float(ln.conf) if ln.conf is not None else 0.0
        h = _bbox_height(ln.box)
        lines.append({"text": text, "conf": conf, "box": ln.box, "h": h})

    debug = {"date_candidates": [], "code_candidates": [], "title_candidates": []}

    # --- DATE ---
    date_best = None
    date_best_score = -1e9
    for ln in lines:
        for m in DATE_RE.finditer(ln["text"]):
            raw = m.group(1)
            iso = parse_date_to_iso(raw)
            if not iso:
                continue
            score = 10.0 + ln["conf"]
            debug["date_candidates"].append(
                {"raw": raw, "iso": iso, "line": ln["text"], "conf": ln["conf"], "score": score}
            )
            if score > date_best_score:
                date_best_score = score
                date_best = {"raw": raw, "iso": iso, "conf": ln["conf"], "line_text": ln["text"]}

    # --- CODE (token-level) ---
    code_best = None
    code_best_score = -1e9
    for ln in lines:
        for tok in code_token_candidates(ln["text"]):
            sc = score_code_token(tok)
            score = sc + ln["conf"]
            debug["code_candidates"].append(
                {"token": tok, "from_line": ln["text"], "conf": ln["conf"], "score": score}
            )
            if score > code_best_score:
                code_best_score = score
                code_best = {"value": tok, "conf": ln["conf"], "line_text": ln["text"]}

    # --- TITLE ---
    excluded = set()
    if date_best:
        excluded.add(date_best["line_text"])

    title_best = None
    title_best_score = -1e9
    for ln in lines:
        t = ln["text"]
        if t in excluded and len(lines) > 1:
            continue

        base = 0.0
        if DATE_RE.search(t):
            base -= 100.0
        if looks_like_code(t):
            base -= 4.0

        # bbox height is a strong title hint (works well on labels/forms)
        score = base + (ln["h"] * 0.05) + ln["conf"]

        debug["title_candidates"].append(
            {"candidate": t, "height": ln["h"], "conf": ln["conf"], "score": score}
        )
        if score > title_best_score:
            title_best_score = score
            title_best = {"value": t, "conf": ln["conf"], "line_text": t}

    errors = []
    if not title_best:
        errors.append("title not found")
    if not date_best:
        errors.append("date not found")
    if not code_best:
        errors.append("code not found")

    if errors:
        return {
            "status": "failed",
            "error": "Unable to extract required fields: " + ", ".join(errors),
            "debug": debug,
            "raw_lines": [{"text": x["text"], "conf": x["conf"], "h": x["h"]} for x in lines],
        }

    return {
        "status": "success",
        "title": title_best["value"],
        "date": date_best["iso"],
        "code": code_best["value"],
        "confidence": {
            "title": title_best["conf"],
            "date": date_best["conf"],
            "code": code_best["conf"],
        },
        "debug": debug,
        "raw_lines": [{"text": x["text"], "conf": x["conf"], "h": x["h"]} for x in lines],
    }


# ---------------------------------------------------------------------
# End-to-end function (FAST): early exit + limited variants
# ---------------------------------------------------------------------
def _avg_conf(res: dict) -> float:
    c = res.get("confidence", {}) or {}
    return (c.get("title", 0.0) + c.get("code", 0.0) + c.get("date", 0.0)) / 3.0


def process_image_fast(
    image_path: str,
    *,
    max_side: int = 1100,
    conf_stop: float = 0.97,
    try_contrast: bool = True,
    try_th_adapt: bool = True,
    try_th_otsu: bool = False,
) -> Dict[str, Any]:
    """
    Main pipeline used by Celery worker.
    Returns a dict:
      - status: success/failed
      - title/date/code on success
      - error on failure
      - raw_lines/raw_text/variant_used/confidence/debug
    """
    try:
        pil = Image.open(image_path)
        bgr = to_bgr(pil)
        bgr = resize_max_side(bgr, max_side=max_side)

        vars_ = preprocess_variants(bgr)

        # deskew estimation from adaptive threshold, then rotate ORIGINAL
        angle = estimate_skew_angle_from_binary(vars_["th_adapt"])
        bgr_rot = rotate_image(bgr, angle)

        candidates: List[Tuple[str, np.ndarray]] = [("orig_rot", bgr_rot)]

        if try_contrast:
            candidates.append(("contrast", cv2.cvtColor(vars_["contrast"], cv2.COLOR_GRAY2BGR)))
        if try_th_adapt:
            candidates.append(("th_adapt", cv2.cvtColor(deskew(vars_["th_adapt"]), cv2.COLOR_GRAY2BGR)))
        if try_th_otsu:
            candidates.append(("th_otsu", cv2.cvtColor(deskew(vars_["th_otsu"]), cv2.COLOR_GRAY2BGR)))

        best: Optional[Dict[str, Any]] = None
        best_score = -1e9

        for name, img in candidates:
            ocr_lines = run_paddle_ocr_safe(img)
            res = extract_fields_v2(ocr_lines)

            # scoring
            if res.get("status") == "success":
                score = 100.0 + _avg_conf(res) + len(ocr_lines) * 0.02
            else:
                score = len(ocr_lines) * 0.01

            if score > best_score:
                best_score = score
                best = res
                best["ocr_engine"] = "paddleocr"
                best["variant_used"] = name
                best["raw_text"] = "\n".join([ln.text for ln in ocr_lines if ln.text.strip()])
                best["raw_lines"] = [{"text": ln.text, "conf": float(ln.conf)} for ln in ocr_lines]

            # early exit
            if res.get("status") == "success" and _avg_conf(res) >= conf_stop:
                break

        # best should never be None, but guard anyway
        if best is None:
            return {
                "status": "failed",
                "error": "OCR pipeline produced no output.",
                "ocr_engine": "paddleocr",
                "variant_used": None,
                "raw_text": "",
                "raw_lines": [],
            }

        return best

    except Exception as e:
        return {
            "status": "failed",
            "error": f"Unhandled error in OCR pipeline: {repr(e)}",
            "ocr_engine": "paddleocr",
            "variant_used": None,
            "raw_text": "",
            "raw_lines": [],
        }
