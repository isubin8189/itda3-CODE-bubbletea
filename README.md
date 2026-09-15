# CODE-bubbletea: OCR 기반 소비기한 추출

제3회 ITDA 연합학술제 1차 예선 제출 프로젝트입니다. `predict.ipynb`는 상품 이미지에서 소비기한 날짜를 추출하고, 결과를 `submission.csv` 형식으로 저장합니다.

## Custom data

`additional_images.zip`에는 상품 이미지 100장이 포함되어 있습니다. **수집 출처는 집, 올리브영, 편의점**이며, 라벨은 이미지에 표시된 소비기한을 `YYYY-MM-DD` 형식의 연·월·일로 기록합니다. 확장자를 제외한 6자리 숫자 파일명을 `image_id`로 사용합니다.

라벨 이름 : 'additional_answers.csv' 파일 내에 존재

google drive : https://drive.google.com/drive/folders/1QCfGgR7nrNhpi4rBDEm1qZ2EtHY6uh_Y?usp=drive_link 

## Pipeline

노트북은 CPU 환경에서 PaddleOCR → docTR → RapidOCR 순서로 추론합니다. 이전 단계에서 신뢰도가 낮다고 판단한 이미지만 다음 단계에서 다시 처리합니다.

1. **PaddleOCR**
   - 전체 이미지에 PP-OCRv5 mobile 검출 모델과 한국어 인식 모델을 적용합니다.
   - 원본 이미지에서 날짜를 찾지 못하면 어두운 글자/밝은 글자용 적응형 이진화와 방향 보정을 차례로 시도합니다.
   - OCR 신뢰도, 날짜 표기 순서 힌트, 후보 간 일치 여부를 이용해 결과와 저신뢰 이미지를 판정합니다.
2. **docTR**
   - PaddleOCR 저신뢰 이미지만 `db_mobilenet_v3_large`와 `crnn_mobilenet_v3_small`로 재추론합니다.
   - 소비기한·유통기한·EXP 등의 긍정 문맥과 제조일·MFG 등의 부정 문맥을 반영해 날짜 후보를 선택합니다.
3. **RapidOCR**
   - docTR 이후에도 저신뢰로 남은 이미지를 PP-OCRv4 ONNX 모델로 마지막 재추론합니다.
   - 연·월·일 전체 날짜뿐 아니라 월·일만 인식된 결과도 보완합니다.

날짜 후보는 실제 달력 날짜인지 검증하며, 허용하는 연도 범위는 2018년부터 2032년까지입니다. 모든 OCR 엔진은 CPU만 사용합니다(`USE_GPU = False`).

## Repository structure

```text
itda3-CODE-bubbletea/
├── predict.ipynb
├── download_weights.sh
├── requirements.txt
├── README.md
└── models/                         # download_weights.sh 실행 후 생성
    ├── PP-OCRv5_mobile_det/
    ├── korean_PP-OCRv5_mobile_rec/
    ├── PP-LCNet_x1_0_doc_ori/
    ├── PP-LCNet_x1_0_textline_ori/
    ├── doctr/
    │   ├── detector.pt
    │   └── recognizer.pt
    └── rapidocr/
        ├── ch_PP-OCRv4_det_infer.onnx
        ├── ch_PP-OCRv4_rec_infer.onnx
        └── ch_ppocr_mobile_v2.0_cls_infer.onnx
```

## Environment

- Python 3.10.11
- CPU-only inference

가상환경을 만든 뒤 의존성을 설치합니다.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell에서는 활성화 명령으로 `.\.venv\Scripts\Activate.ps1`을 사용합니다. `nbconvert`와 `ipykernel`은 `requirements.txt`에 포함되어 있습니다.

## Weights

Before inference, download/prepare pretrained weights:

```bash
bash download_weights.sh
```

This creates the required local weights under ./models.

최초 실행 시에는 PaddleOCR 및 docTR 사전학습 가중치를 내려받기 위한 인터넷 연결이 필요합니다. 스크립트는 PaddleOCR 모델을 로컬 디렉터리에 복사하고, docTR 가중치를 `.pt` 파일로 저장하며, RapidOCR 패키지에 포함된 ONNX 모델을 복사합니다. 준비가 끝난 뒤 `predict.ipynb`는 로컬 가중치를 사용합니다.

## Inference

입력과 출력 경로는 다음 환경변수로 지정합니다.

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `ITDA_INPUT_DIR` | `./val_images` | 입력 이미지 폴더 |
| `ITDA_OUTPUT_PATH` | `./submission.csv` | 결과 CSV 경로 |

입력 폴더는 하위 디렉터리까지 탐색하며 `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp` 파일을 처리합니다. 이미지 파일의 확장자를 제외한 이름이 `image_id`가 되므로 각 이미지의 stem은 고유해야 합니다.

기본 경로를 사용할 때는 노트북의 모든 셀을 위에서 아래로 실행하면 됩니다.

```bash
jupyter nbconvert --to notebook --execute predict.ipynb \
  --ExecutePreprocessor.timeout=2400 \
  --output executed_predict.ipynb
```

다른 경로를 지정하는 예시는 다음과 같습니다.

```bash
export ITDA_INPUT_DIR=./val_images
export ITDA_OUTPUT_PATH=./submission.csv

jupyter nbconvert --to notebook --execute predict.ipynb \
  --ExecutePreprocessor.timeout=2400 \
  --output executed_predict.ipynb
```

Windows PowerShell에서는 환경변수를 다음과 같이 설정할 수 있습니다.

```powershell
$env:ITDA_INPUT_DIR = ".\val_images"
$env:ITDA_OUTPUT_PATH = ".\submission.csv"
jupyter nbconvert --to notebook --execute predict.ipynb `
  --ExecutePreprocessor.timeout=2400 `
  --output executed_predict.ipynb
```

## Output

추론이 끝나면 `ITDA_OUTPUT_PATH`에 CSV가 생성됩니다. 인덱스는 저장하지 않습니다.

| 컬럼 | 설명 |
| --- | --- |
| `image_id` | 확장자를 제외한 이미지 파일명 |
| `year` | 인식된 4자리 연도 |
| `month` | 두 자리 월 |
| `day` | 두 자리 일 |
| `final_date` | `YYYY-MM-DD` 형식의 날짜 |

인식하지 못한 날짜 요소는 `NONE`으로 저장합니다. 연·월·일을 모두 인식하지 못한 경우 `final_date`는 `NONE`, 일부만 인식한 경우 `NONE-MM-DD` 등 인식된 요소를 유지한 형식으로 저장합니다.
