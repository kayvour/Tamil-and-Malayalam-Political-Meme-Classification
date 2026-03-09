# Multimodal Tamil and Malayalam Political Meme Classification — Complete Documentation

A deep learning system for hierarchical classification of political memes in Tamil and Malayalam using **CLIP**, **Vision Transformer (ViT)**, and **OCR text extraction**. Includes training pipelines, inference scripts, a Flask web application, and exploratory data analysis.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Dataset Setup](#dataset-setup)
7. [Exploratory Data Analysis](#exploratory-data-analysis)
8. [Training the Models](#training-the-models)
9. [Running Inference](#running-inference)
10. [Running the Web Application](#running-the-web-application)
11. [API Reference](#api-reference)
12. [Architecture Details](#architecture-details)
13. [Notebooks](#notebooks)
14. [Troubleshooting](#troubleshooting)
15. [Performance](#performance)
16. [Key Insights](#key-insights)

---

## Overview

Political memes are multimodal — they contain visual elements (faces, symbols, logos), embedded text (often in regional languages), sarcasm, and contextual meaning. Traditional single-modality models fail to capture this.

This project employs a **two-level hierarchical classification** approach:

| Level | Type | Classes |
|-------|------|---------|
| **Level 1** | Binary | `TROLL / OPPOSE`, `SUPPORT` |
| **Level 2** | Multi-class | `Support for Party`, `Support for Person`, `Troll/Oppose Against Party`, `Troll/Oppose Against Person` |

**Models available:**

| Model | Description |
|-------|-------------|
| **Zero-Shot CLIP** | Pre-trained `openai/clip-vit-base-patch32` — works with no training |
| **Fine-tuned CLIP** | CLIP fine-tuned on the meme dataset |
| **Fine-tuned ViT** | `google/vit-base-patch16-224` fine-tuned on the meme dataset |
| **Ensemble (CLIP + ViT)** | Combined model for Level 2 classification |

---

## Features

- Two-level hierarchical classification (binary → multi-class)
- Zero-shot classification using CLIP (no training required)
- Fine-tuned CLIP and ViT models for higher accuracy
- OCR text extraction via Tesseract (Malayalam + English)
- Flask web app with drag-and-drop image upload and model selection
- GPU acceleration (CUDA) with automatic CPU fallback
- Class-weighted loss to handle severe data imbalance
- Stratified train-validation splits (80/20)
- Comprehensive EDA with visualizations

---

## Project Structure

```
Tamil-and-Malayalam-Political-Meme-Classification/
│
├── app.py                          # Flask web application
├── web_requirements.txt            # Python dependencies for the web app
├── README.md                       # Project overview
├── DOCUMENTATION.md                # This file — complete documentation
├── LICENSE                         # License file
│
├── templates/
│   └── index.html                  # Web UI template
├── static/
│   ├── css/style.css               # Frontend styles
│   └── js/main.js                  # Frontend JavaScript
│
├── trained_weights/                # Place trained .pth weight files here
│   ├── clip_level1.pth             #   (generated after training)
│   ├── clip_level2.pth
│   ├── vit_level1.pth
│   └── vit_level2.pth
│
├── Dataset/
│   ├── Train-20260214T175134Z-1-001/Train/
│   │   ├── Train_images/           # Tamil training images
│   │   └── Train_labels.xlsx       # Tamil training labels
│   ├── Train-20260214T175142Z-1-001/Train/
│   │   ├── Train_images/           # Malayalam training images
│   │   └── Malayalam_Train_label.xlsx
│   ├── Test-20260214T175139Z-1-001/Test/
│   │   ├── Test_images/            # Tamil test images
│   │   └── Test_labels.xlsx
│   └── Test-20260214T175144Z-1-001/Test/
│       ├── Test_images/            # Malayalam test images
│       └── Malayalam_Test_label.xlsx
│
├── Malayalam_model/
│   ├── predict_test_complete.py    # End-to-end Level 1 + Level 2 prediction
│   ├── Level_1_classification/
│   │   ├── train_clip.py           # Train CLIP for Level 1 (binary)
│   │   ├── train_vit.py            # Train ViT for Level 1
│   │   ├── inference_clip.py       # Single-image inference (CLIP L1)
│   │   ├── inference_vit.py        # Single-image inference (ViT L1)
│   │   ├── inference_both.py       # Ensemble inference (CLIP + ViT L1)
│   │   └── predict_test.py         # Batch test set prediction (L1)
│   └── Level_2_classification/
│       ├── train_clip.py           # Train CLIP for Level 2 (multi-class)
│       ├── train_vit.py            # Train ViT for Level 2
│       ├── inference_clip.py       # Single-image inference (CLIP L2)
│       ├── inference_vit.py        # Single-image inference (ViT L2)
│       └── predict_test.py         # Batch test set prediction (L1 + L2)
│
├── Malayalam_EDA/
│   ├── malayalam_eda.py            # EDA Python script
│   ├── malayalam_eda.ipynb         # EDA Jupyter notebook
│   ├── summary_statistics.csv      # Generated summary statistics
│   └── README.md                   # EDA documentation
│
├── speech_vilt.ipynb               # ViLT experiment notebook
└── tamil_final.ipynb               # Tamil classification notebook
```

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.9 or higher |
| **CUDA** (optional) | CUDA 11.7+ for GPU acceleration |
| **Tesseract OCR** | Required for text extraction from memes |
| **RAM** | 8 GB minimum, 16 GB recommended |
| **Disk** | ~5 GB for model weights and dataset |

### Install Tesseract OCR

**Windows:**

1. Download the installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
2. During installation, select additional language packs: **Malayalam**, **Tamil**.
3. Add the install directory (e.g., `C:\Program Files\Tesseract-OCR`) to your system `PATH`.
4. Verify:
   ```
   tesseract --version
   ```

**Ubuntu / Debian:**

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-mal tesseract-ocr-tam
```

**macOS:**

```bash
brew install tesseract
brew install tesseract-lang   # includes Malayalam and Tamil
```

**Google Colab:**

```python
!apt-get install tesseract-ocr tesseract-ocr-mal tesseract-ocr-tam -y
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Tamil-and-Malayalam-Political-Meme-Classification.git
cd Tamil-and-Malayalam-Political-Meme-Classification
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

**For the web application only:**

```bash
pip install -r web_requirements.txt
```

**For training and full pipeline:**

```bash
pip install torch torchvision transformers pillow pytesseract pandas openpyxl scikit-learn tqdm matplotlib seaborn flask flask-cors numpy
```

> **GPU support:** Install PyTorch with CUDA first from [pytorch.org](https://pytorch.org/get-started/locally/):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### Complete dependency list

| Package | Purpose |
|---------|---------|
| `flask` >= 3.0.0 | Web application server |
| `flask-cors` >= 4.0.0 | Cross-origin request handling |
| `torch` >= 2.0.0 | Deep learning framework |
| `transformers` >= 4.38.0 | CLIP and ViT model loading (Hugging Face) |
| `pillow` >= 10.0.0 | Image processing |
| `pytesseract` >= 0.3.10 | OCR text extraction |
| `numpy` >= 1.24.0 | Numerical operations |
| `pandas` | Data manipulation (training/inference) |
| `openpyxl` | Reading `.xlsx` label files |
| `scikit-learn` | Metrics, stratified train/test splitting |
| `tqdm` | Progress bars |
| `matplotlib` | EDA visualizations |
| `seaborn` | EDA visualizations |

---

## Dataset Setup

The dataset consists of political meme images with corresponding Excel label files.

### Label file format

Each Excel file must contain:

| Column | Description |
|--------|-------------|
| `Image_id` / `meme_id` | Unique numeric identifier for the image |
| `Image_name` | Filename (e.g., `001.jpg`) |
| `Level1` / `Level 1` | Binary label: `Troll/Oppose` or `Support` |
| `Level2` / `Level 2` | Multi-class label (see below) |

### Level 2 classes

| Class | Description |
|-------|-------------|
| `Support for Party` | Meme supports a political party |
| `Support for Person` | Meme supports a political leader/person |
| `Troll/Oppose Against Party` | Meme trolls or opposes a political party |
| `Troll/Oppose Against Person` | Meme trolls or opposes a political leader/person |

### Expected directory structure

```
Dataset/
├── Train-20260214T175134Z-1-001/Train/
│   ├── Train_images/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── Train_labels.xlsx            # Tamil labels
│
├── Train-20260214T175142Z-1-001/Train/
│   ├── Train_images/
│   │   ├── 1.jpg
│   │   └── ...
│   └── Malayalam_Train_label.xlsx    # Malayalam labels
│
├── Test-20260214T175139Z-1-001/Test/
│   ├── Test_images/
│   └── Test_labels.xlsx             # Tamil test labels
│
└── Test-20260214T175144Z-1-001/Test/
    ├── Test_images/
    └── Malayalam_Test_label.xlsx     # Malayalam test labels
```

---

## Exploratory Data Analysis

Run EDA on the Malayalam dataset before training to understand class distributions and data quality.

```bash
cd Malayalam_EDA

# Option 1: Python script
python malayalam_eda.py

# Option 2: Jupyter notebook
jupyter notebook malayalam_eda.ipynb
```

**Generated outputs:**

| File | Description |
|------|-------------|
| `label_distribution.png` | Bar charts for Level 1 and Level 2 distributions |
| `class_imbalance.png` | Pie charts showing class imbalance |
| `image_properties.png` | Histograms of image dimensions and file sizes |
| `sample_images.png` | Grid of 18 sample memes with labels |
| `label_correlation.png` | Heatmap of Level 1 vs Level 2 correlation |
| `summary_statistics.csv` | Key dataset statistics |

**Key findings from the Malayalam dataset:**

- ~803 total samples
- Severe class imbalance (up to 6:1 ratio)
- Average image size: ~775 x 794 px
- Variable dimensions — images are resized to 224x224 during training

---

## Training the Models

Training follows a **two-level hierarchy**: train Level 1 (binary) first, then Level 2 (multi-class).

### Step 1: Train Level 1 (Binary: Troll/Oppose vs Support)

```bash
cd Malayalam_model/Level_1_classification

# Train with CLIP
python train_clip.py

# OR train with ViT
python train_vit.py
```

**Training configuration (Level 1):**

| Parameter | CLIP | ViT |
|-----------|------|-----|
| Base model | `openai/clip-vit-base-patch32` | `google/vit-base-patch16-224` |
| Batch size | 32 | 8 |
| Learning rate | 1e-5 | 2e-5 |
| Epochs | 15 | 20 |
| Optimizer | AdamW (weight_decay=0.01) | AdamW (weight_decay=0.01) |
| Scheduler | — | CosineAnnealingLR |
| Loss | CrossEntropyLoss (class-weighted) | CrossEntropyLoss (class-weighted) |
| Validation split | 80/20 stratified | 80/20 stratified |

**Output:** `best_clip_model.pth` or `best_vit_model.pth` saved in the same directory.

### Step 2: Train Level 2 (Multi-class: 4-5 classes)

```bash
cd Malayalam_model/Level_2_classification

# Train with CLIP
python train_clip.py

# OR train with ViT
python train_vit.py
```

**Training configuration (Level 2):**

| Parameter | CLIP | ViT |
|-----------|------|-----|
| Base model | `openai/clip-vit-base-patch32` | `google/vit-base-patch16-224` |
| Batch size | 32 | 8 |
| Learning rate | 1e-5 | 2e-5 |
| Epochs | 15 | 20 |
| Num classes | 5 | 5 |
| Loss | CrossEntropyLoss | CrossEntropyLoss |

**Output:** `best_clip_level2_model.pth` or `best_vit_level2_model.pth`.

### Step 3: Copy weights for the web app (optional)

To use trained models in the web application, copy the weight files to `trained_weights/`:

**Windows:**

```powershell
copy Malayalam_model\Level_1_classification\best_clip_model.pth trained_weights\clip_level1.pth
copy Malayalam_model\Level_2_classification\best_clip_level2_model.pth trained_weights\clip_level2.pth
copy Malayalam_model\Level_1_classification\best_vit_model.pth trained_weights\vit_level1.pth
copy Malayalam_model\Level_2_classification\best_vit_level2_model.pth trained_weights\vit_level2.pth
```

**Linux / macOS:**

```bash
cp Malayalam_model/Level_1_classification/best_clip_model.pth trained_weights/clip_level1.pth
cp Malayalam_model/Level_2_classification/best_clip_level2_model.pth trained_weights/clip_level2.pth
cp Malayalam_model/Level_1_classification/best_vit_model.pth trained_weights/vit_level1.pth
cp Malayalam_model/Level_2_classification/best_vit_level2_model.pth trained_weights/vit_level2.pth
```

---

## Running Inference

### Single image inference

```bash
# Level 1 — CLIP
cd Malayalam_model/Level_1_classification
python inference_clip.py

# Level 1 — ViT
python inference_vit.py

# Level 1 — Ensemble (CLIP + ViT)
python inference_both.py
```

```bash
# Level 2 — CLIP
cd Malayalam_model/Level_2_classification
python inference_clip.py

# Level 2 — ViT
python inference_vit.py
```

### Batch prediction on test set

**Level 1 only:**

```bash
cd Malayalam_model/Level_1_classification
python predict_test.py
```

**Level 2 only (also runs Level 1 internally):**

```bash
cd Malayalam_model/Level_2_classification
python predict_test.py
```

**Complete pipeline (Level 1 + Level 2 combined):**

```bash
cd Malayalam_model
python predict_test_complete.py
```

This generates `Malayalam_Test_label_PREDICTED.xlsx` in the test dataset directory with columns:

| Column | Description |
|--------|-------------|
| `meme_id` | Image identifier |
| `Level 1` | Predicted binary label |
| `Level 1 Confidence` | Confidence score for Level 1 |
| `Level 2` | Predicted multi-class label |
| `Level 2 Confidence` | Confidence score for Level 2 |

---

## Running the Web Application

### Quick start (3 commands)

```bash
cd Tamil-and-Malayalam-Political-Meme-Classification
pip install -r web_requirements.txt
python app.py
```

The server starts at **http://localhost:5000**. Open this URL in your browser.

### What you'll see

1. **Model selector** — Choose between Zero-Shot CLIP, Trained CLIP, or Trained ViT.
2. **Image upload** — Drag & drop or browse for a meme image (JPG, PNG, GIF, BMP).
3. **Results panel** — Shows:
   - Level 1 classification (binary) with confidence bar
   - Level 2 classification (multi-class) with confidence bar
   - Extracted OCR text (Malayalam + English)
   - Name of the model used

### Models available in the web app

| Model | Requirements | Description |
|-------|-------------|-------------|
| **Zero-Shot CLIP** | None — always ready | Pre-trained CLIP, works out of the box with lower accuracy |
| **Trained CLIP** | `clip_level1.pth` + `clip_level2.pth` in `trained_weights/` | Fine-tuned CLIP for higher accuracy |
| **Trained ViT** | `vit_level1.pth` + `vit_level2.pth` in `trained_weights/` | Fine-tuned Vision Transformer |

> **Note:** Zero-Shot CLIP requires no training and is always available. Trained models become available after you place the `.pth` weight files in `trained_weights/` and restart the server.

### Changing the port

Edit `app.py`, last line:

```python
app.run(debug=True, host='0.0.0.0', port=5000)
#                                     ^^^^ change this
```

---

## API Reference

The Flask app exposes the following REST API endpoints:

### `POST /api/classify`

Classify a meme image.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File | Yes | Image file to classify |
| `model` | String | No | `zero_shot_clip`, `trained_clip`, or `trained_vit` (default: `zero_shot_clip`) |

**Response (success — 200):**

```json
{
  "success": true,
  "level1": {
    "label": "TROLL / OPPOSE",
    "confidence": 87.34
  },
  "level2": {
    "label": "Troll/Oppose Against Party",
    "confidence": 72.15
  },
  "ocr_text": "Extracted text from the meme...",
  "model_used": "Zero-Shot CLIP"
}
```

**Response (error — 400/500):**

```json
{
  "success": false,
  "error": "No image file provided"
}
```

---

### `GET /api/models`

List all registered models and their status.

**Response:**

```json
{
  "models": {
    "zero_shot_clip": {
      "name": "Zero-Shot CLIP",
      "description": "Pre-trained CLIP model (no fine-tuning).",
      "status": "ready",
      "type": "zero-shot"
    },
    "trained_clip": {
      "name": "Trained CLIP",
      "description": "Fine-tuned CLIP model. Place weight files in trained_weights/",
      "status": "not_trained",
      "type": "trained"
    },
    "trained_vit": {
      "name": "Trained ViT",
      "description": "Fine-tuned ViT model. Place weight files in trained_weights/",
      "status": "not_trained",
      "type": "trained"
    }
  },
  "default": "zero_shot_clip"
}
```

**Model `status` values:**

| Status | Meaning |
|--------|---------|
| `ready` | Model is loaded and can classify images |
| `not_trained` | Weight files are missing |
| `weights_found_restart_needed` | Weights were added at runtime; restart the server to load them |

---

### `GET /api/status`

Check server health.

**Response:**

```json
{
  "status": "online",
  "device": "cuda",
  "models_loaded": true,
  "available_models": 1,
  "total_models": 3
}
```

---

## Architecture Details

### Two-Level Hierarchical Classification

```
                    Input Image
                         |
                    +----------+
                    |  Level 1 |  Binary Classification
                    | (CLIP/ViT)|
                    +----+-----+
                         |
              +----------+----------+
              |                     |
        TROLL/OPPOSE            SUPPORT
              |                     |
         +----+-----+         +----+-----+
         | Level 2  |         | Level 2  |  Multi-class
         |(CLIP/ViT)|         |(CLIP/ViT)|
         +----+-----+         +----+-----+
              |                     |
     +--------+--------+    +------+------+
     |                 |    |              |
  Against           Against  For          For
  Party             Person   Party        Person
```

### CLIP Classifier Head (Level 1 — Binary)

```
CLIP Image Encoder -> 768-dim features
  -> Linear(768, 256) -> ReLU -> Dropout(0.3)
  -> Linear(256, 2) -> Softmax
```

### CLIP Classifier Head (Level 2 — Multi-class)

```
CLIP Image Encoder -> 768-dim features
  -> Linear(768, 256) -> ReLU -> Dropout(0.3)
  -> Linear(256, 5) -> Softmax
```

### Combined CLIP + ViT Model (Level 2 Ensemble)

```
CLIP Image Encoder -> 512-dim
ViT Image Encoder  -> 768-dim
Concatenate -> 1280-dim
  -> Linear(1280, 512) -> BatchNorm -> ReLU -> Dropout(0.4)
  -> Linear(512, 256)  -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(256, 5) -> Softmax
```

### Zero-Shot CLIP Approach

Uses text prompts to classify without any fine-tuning:

**Level 1 prompts:**
- `"a political meme that trolls, opposes, or criticizes"`
- `"a political meme that supports or praises"`

**Level 2 prompts:**
- `"a political meme supporting a political party"`
- `"a political meme supporting a political person or leader"`
- `"a political meme trolling or opposing a political party"`
- `"a political meme trolling or opposing a political person or leader"`

### OCR Pipeline

```
Input Image -> Tesseract OCR (mal+eng languages) -> Extracted Text
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `tamil_final.ipynb` | Main Tamil meme classification experiment — CLIP feature extraction, MLP classifier training, evaluation |
| `speech_vilt.ipynb` | ViLT (Vision-and-Language Transformer) experiment |
| `Malayalam_EDA/malayalam_eda.ipynb` | Interactive EDA for the Malayalam dataset |

Run notebooks with:

```bash
jupyter notebook tamil_final.ipynb
```

Or open them in VS Code, Google Colab, or JupyterLab.

---

## Troubleshooting

### Common issues

| Problem | Solution |
|---------|----------|
| `TesseractNotFoundError` | Install Tesseract OCR and add it to your system PATH |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in training scripts (e.g., set to 8) |
| Trained model shows `not_trained` in web app | Place `.pth` files in `trained_weights/` and restart the Flask server |
| `ModuleNotFoundError: openpyxl` | Run `pip install openpyxl` (needed for `.xlsx` files) |
| Malayalam OCR returns garbage text | Install the Malayalam Tesseract language pack (`tesseract-ocr-mal`) |
| `FileNotFoundError` for images | Verify the dataset directory structure matches the layout above |
| Web app won't start on port 5000 | Another process is using port 5000 — kill it or change the port in `app.py` |
| Low accuracy with Zero-Shot CLIP | Expected — zero-shot is a baseline. Train the models for better results |
| `RuntimeError: weight shape mismatch` | Ensure you're loading weights trained with the same architecture |

### Verifying your setup

```bash
# Check Python version
python --version

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check Tesseract
tesseract --version

# Check all imports work
python -c "import flask, torch, transformers, PIL, pytesseract; print('All imports OK')"

# Check GPU details (if available)
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## Performance

Expected performance varies by dataset and model:

| Metric | Typical Range |
|--------|---------------|
| Training Accuracy | 85-95% |
| Test Accuracy | 80-88% |
| Macro F1 Score | 0.75-0.85 |

Performance depends on:
- OCR quality for text extraction
- Label consistency in the dataset
- Class imbalance severity (up to 20:1 in some splits)
- Model choice (ensemble > single model > zero-shot)

---

## Key Insights

- Multimodal learning (image + text) significantly outperforms single-modality approaches
- Severe class imbalance (up to 20:1) requires class-weighted loss functions
- Oversampling can destabilize small datasets — weighted loss is preferred
- CLIP converges quickly and provides strong baselines even without fine-tuning
- OCR quality directly impacts classification performance
- Ensemble methods (CLIP + ViT) yield the best Level 2 accuracy
- Zero-shot CLIP provides a useful rapid baseline for new datasets

---

## Quick-Start Summary

```bash
# 1. Install
pip install -r web_requirements.txt

# 2. Run web app (zero-shot, no training needed)
python app.py
# -> Open http://localhost:5000 in your browser

# 3. (Optional) Train models for better accuracy
cd Malayalam_model/Level_1_classification
python train_clip.py
cd ../Level_2_classification
python train_clip.py

# 4. Copy weights to web app
cd ../..
copy Malayalam_model\Level_1_classification\best_clip_model.pth trained_weights\clip_level1.pth
copy Malayalam_model\Level_2_classification\best_clip_level2_model.pth trained_weights\clip_level2.pth

# 5. Restart web app with trained models
python app.py
```

---

## License

See the [LICENSE](LICENSE) file for details.
