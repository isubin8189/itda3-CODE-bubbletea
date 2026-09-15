#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path
import shutil

import torch
from paddleocr import PaddleOCR
from doctr.models import ocr_predictor
import rapidocr_onnxruntime

root = Path("models")
root.mkdir(parents=True, exist_ok=True)

_ = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
    doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
    textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    device="cpu",
)

paddlex_cache = Path.home() / ".paddlex" / "official_models"
paddle_names = [
    "PP-OCRv5_mobile_det",
    "korean_PP-OCRv5_mobile_rec",
    "PP-LCNet_x1_0_doc_ori",
    "PP-LCNet_x1_0_textline_ori",
]

for name in paddle_names:
    src = paddlex_cache / name
    dst = root / name
    if not src.exists():
        raise FileNotFoundError(f"PaddleOCR model not found after download: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

doctr_dir = root / "doctr"
doctr_dir.mkdir(parents=True, exist_ok=True)

doctr_model = ocr_predictor(
    det_arch="db_mobilenet_v3_large",
    reco_arch="crnn_mobilenet_v3_small",
    pretrained=True,
    pretrained_backbone=True,
    assume_straight_pages=True,
    preserve_aspect_ratio=True,
)

torch.save(
    doctr_model.det_predictor.model.state_dict(),
    doctr_dir / "detector.pt",
)
torch.save(
    doctr_model.reco_predictor.model.state_dict(),
    doctr_dir / "recognizer.pt",
)

rapid_pkg = Path(rapidocr_onnxruntime.__file__).resolve().parent
rapid_src = rapid_pkg / "models"
rapid_dst = root / "rapidocr"
rapid_dst.mkdir(parents=True, exist_ok=True)

rapid_names = [
    "ch_PP-OCRv4_det_infer.onnx",
    "ch_PP-OCRv4_rec_infer.onnx",
    "ch_ppocr_mobile_v2.0_cls_infer.onnx",
]

for name in rapid_names:
    src = rapid_src / name
    dst = rapid_dst / name
    if not src.exists():
        raise FileNotFoundError(f"RapidOCR packaged model not found: {src}")
    shutil.copy2(src, dst)

print("Offline weights prepared under ./models")
PY
