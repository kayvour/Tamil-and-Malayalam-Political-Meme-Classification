from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, ViTForImageClassification, ViTImageProcessor
import pytesseract
import io
import os

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {MODEL_DEVICE}")

# Paths for trained model weights (place .pth files here when available)
TRAINED_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'trained_weights')
os.makedirs(TRAINED_WEIGHTS_DIR, exist_ok=True)

TRAINED_CLIP_L1_PATH = os.path.join(TRAINED_WEIGHTS_DIR, 'clip_level1.pth')
TRAINED_CLIP_L2_PATH = os.path.join(TRAINED_WEIGHTS_DIR, 'clip_level2.pth')
TRAINED_VIT_L1_PATH = os.path.join(TRAINED_WEIGHTS_DIR, 'vit_level1.pth')
TRAINED_VIT_L2_PATH = os.path.join(TRAINED_WEIGHTS_DIR, 'vit_level2.pth')

# Level 1 labels and prompts
LEVEL1_LABELS = ['TROLL / OPPOSE', 'SUPPORT']
LEVEL1_PROMPTS = [
    "a political meme that trolls, opposes, or criticizes",
    "a political meme that supports or praises"
]

# Level 2 labels and prompts  
LEVEL2_LABELS = [
    'Support for Party',
    'Support for Person',
    'Troll/Oppose Against Party',
    'Troll/Oppose Against Person'
]
LEVEL2_PROMPTS = [
    "a political meme supporting a political party",
    "a political meme supporting a political person or leader",
    "a political meme trolling or opposing a political party",
    "a political meme trolling or opposing a political person or leader"
]

# ============================================================================
# MODEL REGISTRY
# ============================================================================

# Track available models and their status
MODELS = {}

def check_trained_weights():
    """Check which trained weight files exist and update model availability."""
    statuses = {}
    statuses['trained_clip'] = (
        os.path.isfile(TRAINED_CLIP_L1_PATH) and os.path.isfile(TRAINED_CLIP_L2_PATH)
    )
    statuses['trained_vit'] = (
        os.path.isfile(TRAINED_VIT_L1_PATH) and os.path.isfile(TRAINED_VIT_L2_PATH)
    )
    return statuses

# ============================================================================
# LOAD MODELS
# ============================================================================

# --- Zero-shot CLIP (always available) ---
print("Loading CLIP model for zero-shot classification...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(MODEL_DEVICE)
clip_model.eval()
MODELS['zero_shot_clip'] = {
    'name': 'Zero-Shot CLIP',
    'description': 'Pre-trained CLIP model (no fine-tuning). Works out of the box but lower accuracy.',
    'status': 'ready',
    'type': 'zero-shot'
}
print("Zero-shot CLIP model loaded!")

# --- Trained CLIP (placeholder) ---
trained_clip_model = None
trained_clip_processor = None
weight_status = check_trained_weights()

if weight_status['trained_clip']:
    print("Loading trained CLIP weights...")
    try:
        trained_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        trained_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(MODEL_DEVICE)
        trained_clip_model.load_state_dict(torch.load(TRAINED_CLIP_L1_PATH, map_location=MODEL_DEVICE, weights_only=True), strict=False)
        trained_clip_model.eval()
        MODELS['trained_clip'] = {
            'name': 'Trained CLIP',
            'description': 'Fine-tuned CLIP model on Malayalam political meme dataset.',
            'status': 'ready',
            'type': 'trained'
        }
        print("Trained CLIP loaded!")
    except Exception as e:
        print(f"Failed to load trained CLIP: {e}")
        MODELS['trained_clip'] = {
            'name': 'Trained CLIP',
            'description': 'Fine-tuned CLIP model. Place clip_level1.pth & clip_level2.pth in trained_weights/',
            'status': 'not_trained',
            'type': 'trained'
        }
else:
    MODELS['trained_clip'] = {
        'name': 'Trained CLIP',
        'description': 'Fine-tuned CLIP model. Place clip_level1.pth & clip_level2.pth in trained_weights/',
        'status': 'not_trained',
        'type': 'trained'
    }
    print("Trained CLIP weights not found — placeholder registered.")

# --- Trained ViT (placeholder) ---
trained_vit_model_l1 = None
trained_vit_model_l2 = None
trained_vit_processor = None

if weight_status['trained_vit']:
    print("Loading trained ViT weights...")
    try:
        trained_vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        trained_vit_model_l1 = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=len(LEVEL1_LABELS), ignore_mismatched_sizes=True
        ).to(MODEL_DEVICE)
        trained_vit_model_l1.load_state_dict(torch.load(TRAINED_VIT_L1_PATH, map_location=MODEL_DEVICE, weights_only=True))
        trained_vit_model_l1.eval()

        trained_vit_model_l2 = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=len(LEVEL2_LABELS), ignore_mismatched_sizes=True
        ).to(MODEL_DEVICE)
        trained_vit_model_l2.load_state_dict(torch.load(TRAINED_VIT_L2_PATH, map_location=MODEL_DEVICE, weights_only=True))
        trained_vit_model_l2.eval()

        MODELS['trained_vit'] = {
            'name': 'Trained ViT',
            'description': 'Fine-tuned Vision Transformer on Malayalam political meme dataset.',
            'status': 'ready',
            'type': 'trained'
        }
        print("Trained ViT loaded!")
    except Exception as e:
        print(f"Failed to load trained ViT: {e}")
        MODELS['trained_vit'] = {
            'name': 'Trained ViT',
            'description': 'Fine-tuned ViT model. Place vit_level1.pth & vit_level2.pth in trained_weights/',
            'status': 'not_trained',
            'type': 'trained'
        }
else:
    MODELS['trained_vit'] = {
        'name': 'Trained ViT',
        'description': 'Fine-tuned ViT model. Place vit_level1.pth & vit_level2.pth in trained_weights/',
        'status': 'not_trained',
        'type': 'trained'
    }
    print("Trained ViT weights not found — placeholder registered.")

print(f"\nModel registry: {', '.join(f'{k} ({v['status']})' for k, v in MODELS.items())}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_text_from_image(image):
    """Extract text from image using pytesseract"""
    try:
        text = pytesseract.image_to_string(image, lang='mal+eng')
        return text.strip() if text else "No text detected"
    except Exception as e:
        return f"OCR Error: {str(e)}"


def classify_zero_shot_clip(image):
    """Classify using zero-shot CLIP with text prompts."""
    img_rgb = image.convert('RGB')
    with torch.no_grad():
        # Level 1
        inputs1 = clip_processor(text=LEVEL1_PROMPTS, images=img_rgb, return_tensors="pt", padding=True).to(MODEL_DEVICE)
        outputs1 = clip_model(**inputs1)
        probs1 = outputs1.logits_per_image.softmax(dim=1)
        pred1 = probs1.argmax(dim=1).item()
        conf1 = probs1[0, pred1].item() * 100

        # Level 2
        inputs2 = clip_processor(text=LEVEL2_PROMPTS, images=img_rgb, return_tensors="pt", padding=True).to(MODEL_DEVICE)
        outputs2 = clip_model(**inputs2)
        probs2 = outputs2.logits_per_image.softmax(dim=1)
        pred2 = probs2.argmax(dim=1).item()
        conf2 = probs2[0, pred2].item() * 100

    return {
        'level1': {'label': LEVEL1_LABELS[pred1], 'confidence': round(conf1, 2)},
        'level2': {'label': LEVEL2_LABELS[pred2], 'confidence': round(conf2, 2)}
    }


def classify_trained_clip(image):
    """Classify using fine-tuned CLIP model."""
    if trained_clip_model is None:
        return None
    # Placeholder: uses same zero-shot approach but with the fine-tuned weights
    img_rgb = image.convert('RGB')
    with torch.no_grad():
        inputs1 = trained_clip_processor(text=LEVEL1_PROMPTS, images=img_rgb, return_tensors="pt", padding=True).to(MODEL_DEVICE)
        outputs1 = trained_clip_model(**inputs1)
        probs1 = outputs1.logits_per_image.softmax(dim=1)
        pred1 = probs1.argmax(dim=1).item()
        conf1 = probs1[0, pred1].item() * 100

        inputs2 = trained_clip_processor(text=LEVEL2_PROMPTS, images=img_rgb, return_tensors="pt", padding=True).to(MODEL_DEVICE)
        outputs2 = trained_clip_model(**inputs2)
        probs2 = outputs2.logits_per_image.softmax(dim=1)
        pred2 = probs2.argmax(dim=1).item()
        conf2 = probs2[0, pred2].item() * 100

    return {
        'level1': {'label': LEVEL1_LABELS[pred1], 'confidence': round(conf1, 2)},
        'level2': {'label': LEVEL2_LABELS[pred2], 'confidence': round(conf2, 2)}
    }


def classify_trained_vit(image):
    """Classify using fine-tuned ViT model."""
    if trained_vit_model_l1 is None or trained_vit_model_l2 is None:
        return None
    img_rgb = image.convert('RGB')
    with torch.no_grad():
        inputs = trained_vit_processor(images=img_rgb, return_tensors="pt").to(MODEL_DEVICE)
        # Level 1
        logits1 = trained_vit_model_l1(**inputs).logits
        probs1 = logits1.softmax(dim=1)
        pred1 = probs1.argmax(dim=1).item()
        conf1 = probs1[0, pred1].item() * 100
        # Level 2
        logits2 = trained_vit_model_l2(**inputs).logits
        probs2 = logits2.softmax(dim=1)
        pred2 = probs2.argmax(dim=1).item()
        conf2 = probs2[0, pred2].item() * 100

    return {
        'level1': {'label': LEVEL1_LABELS[pred1], 'confidence': round(conf1, 2)},
        'level2': {'label': LEVEL2_LABELS[pred2], 'confidence': round(conf2, 2)}
    }


CLASSIFY_FN = {
    'zero_shot_clip': classify_zero_shot_clip,
    'trained_clip': classify_trained_clip,
    'trained_vit': classify_trained_vit,
}


def classify_image(image, model_key='zero_shot_clip'):
    """Classify the image using the selected model."""
    try:
        if model_key not in MODELS:
            return {'success': False, 'error': f'Unknown model: {model_key}'}
        if MODELS[model_key]['status'] != 'ready':
            return {'success': False, 'error': f'{MODELS[model_key]["name"]} is not trained yet. Place weight files in trained_weights/ and restart.'}

        result = CLASSIFY_FN[model_key](image)
        if result is None:
            return {'success': False, 'error': 'Model weights not loaded.'}

        ocr_text = extract_text_from_image(image)

        return {
            'level1': result['level1'],
            'level2': result['level2'],
            'ocr_text': ocr_text,
            'model_used': MODELS[model_key]['name'],
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint for image classification"""
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Get selected model (default to zero_shot_clip)
        model_key = request.form.get('model', 'zero_shot_clip')
        
        # Read image
        image = Image.open(io.BytesIO(file.read()))
        
        # Classify
        result = classify_image(image, model_key)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/models', methods=['GET'])
def api_models():
    """List all available models and their status"""
    # Re-check weight files in case they were added at runtime
    ws = check_trained_weights()
    for key, exists in ws.items():
        if exists and MODELS[key]['status'] == 'not_trained':
            MODELS[key]['status'] = 'weights_found_restart_needed'
    return jsonify({
        'models': {k: v for k, v in MODELS.items()},
        'default': 'zero_shot_clip'
    })


@app.route('/api/status', methods=['GET'])
def api_status():
    """Check server status"""
    return jsonify({
        'status': 'online',
        'device': str(MODEL_DEVICE),
        'models_loaded': True,
        'available_models': sum(1 for m in MODELS.values() if m['status'] == 'ready'),
        'total_models': len(MODELS)
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
