# 🧠 Multimodal Tamil and Malayalam Political Meme Classification using CLIP and VILT

A deep learning project for classifying political memes using **image + OCR text fusion** with OpenAI CLIP embeddings and a custom neural network classifier.

This system supports:

- ✅ English political memes  
- ✅ Malayalam political memes  
- ✅ Multimodal learning (Image + Text)  
- ✅ Class imbalance handling  
- ✅ GPU acceleration (CUDA)  

---

## 📌 Problem Statement

Political memes are multimodal in nature. They contain:

- Visual elements (faces, symbols, logos)
- Embedded text (requires OCR)
- Sarcasm and contextual meaning

Traditional text-only or image-only models fail to capture the complete semantic intent.

This project leverages **CLIP (Contrastive Language–Image Pretraining)** to extract unified multimodal representations for robust classification.

---

## 🏗 Architecture Overview

```
Image → CLIP Image Encoder → 512-dim
Text  → CLIP Text Encoder  → 512-dim
Concatenate → 1024-dim Feature Vector
→ MLP Classifier → 4 Output Classes
```

### Feature Details

- 512-dim image embedding
- 512-dim text embedding (OCR extracted)
- L2 normalization applied
- Concatenated into 1024-dim feature vector

---

## 📂 Dataset Structure

```
meme/
│
├── Train_images/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
│
└── Train_labels.xlsx
```

### Excel Columns

- `Image_id`
- `Image_name`
- `Level1`
- `Level2` ← Final classification label

---

## 🏷 Classification Labels (Level2)

1. Support for Party  
2. Support for Person  
3. Troll/Oppose Against Party  
4. Troll/Oppose Against Person  

---

## 🔎 OCR Pipeline

Text is extracted from memes using:

- `pytesseract`
- PIL image preprocessing

Malayalam support works because:

- Tesseract includes Malayalam language packs
- CLIP’s text encoder handles multilingual tokens reasonably well

---

## 🧠 Feature Extraction Model

Model Used:

```
openai/clip-vit-base-patch32
```

CLIP provides strong multimodal embeddings without requiring fine-tuning.

---

## 🏋️ Classifier Architecture

Fully Connected Neural Network:

- Linear(1024 → 768)
- BatchNorm + ReLU + Dropout(0.3)
- Linear(768 → 512)
- BatchNorm + ReLU + Dropout(0.3)
- Linear(512 → 256)
- BatchNorm + ReLU + Dropout(0.3)
- Linear(256 → 4)

---

## ⚙️ Training Configuration

- Optimizer: AdamW  
- Learning Rate: 3e-4  
- Weight Decay: 0.01  
- Epochs: 20  
- Batch Size: 32  
- Loss: Weighted CrossEntropy (to handle class imbalance)

---

## 📊 Performance

Expected performance (dataset dependent):

| Metric | Typical Range |
|--------|---------------|
| Training Accuracy | 85–95% |
| Test Accuracy | 80–88% |
| Macro F1 Score | 0.75–0.85 |

Performance depends heavily on:
- OCR quality
- Label consistency
- Class imbalance severity

---

## 🌍 Malayalam Extension

The pipeline was extended to handle **Malayalam political memes**:

- OCR using Tesseract Malayalam model
- Same CLIP architecture
- No structural modifications required

This demonstrates language-agnostic multimodal capability.

---

## 🚀 Installation

### Local Setup

```bash
pip install transformers pytesseract pillow scikit-learn tqdm
sudo apt-get install tesseract-ocr
```

### Google Colab

```python
!pip install transformers pytesseract pillow scikit-learn tqdm
!apt-get install tesseract-ocr -y
```

---

## ▶️ Execution Steps

1. Load dataset from Drive
2. Extract OCR text
3. Generate CLIP embeddings
4. Train classifier
5. Evaluate using classification report

---

## 🔥 Key Insights

- Multimodal learning significantly outperforms single-modality models
- Class imbalance drastically affects minority recall
- Oversampling can destabilize small datasets
- Weighted loss improves minority class detection
- CLIP converges quickly without full fine-tuning
- OCR quality directly influences final performance

---

## 📈 Future Improvements

- Fine-tune CLIP end-to-end
- Use multilingual CLIP variants

---

## �️ Running the Frontend (Web App)

```bash
pip install -r web_requirements.txt
python app.py
```

Open **http://localhost:5000** in your browser. Upload a meme image, pick a model, and hit **Classify**.

> Zero-Shot CLIP works immediately with no training needed.  
> For trained models, place the `.pth` weight files in the `trained_weights/` folder and restart.

---

## �📖 Full Documentation

For complete documentation including installation, dataset setup, training steps, inference, web app usage, API reference, architecture details, and troubleshooting, see **[DOCUMENTATION.md](DOCUMENTATION.md)**.
- Replace concatenation with attention-based fusion
- Experiment with focal loss
- Apply stronger vision backbone (ViT-L/14)

---

## 🧑‍💻 Project Type

Engineering Project  
Deep Learning | NLP | Multimodal AI  

---

If you found this project useful, feel free to ⭐ the repository.
