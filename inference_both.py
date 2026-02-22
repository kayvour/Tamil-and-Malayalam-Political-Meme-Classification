import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
from PIL import Image
import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CLIPMemeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, pixel_values):
        outputs = self.clip.get_image_features(pixel_values=pixel_values)
        image_features = outputs if isinstance(outputs, torch.Tensor) else outputs.last_hidden_state[:, 0, :]
        return self.classifier(image_features)

class ViTMemeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

def predict_both(image_path, clip_model_path='best_clip_model.pth', vit_model_path='best_vit_model.pth'):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    if not os.path.exists(clip_model_path):
        print(f"Error: CLIP model not found at {clip_model_path}")
        sys.exit(1)
    if not os.path.exists(vit_model_path):
        print(f"Error: ViT model not found at {vit_model_path}")
        sys.exit(1)
    
    labels = ['TROLL/OPPOSE', 'SUPPORT']
    image = Image.open(image_path).convert('RGB')
    
    print("\n" + "="*60)
    print("MALAYALAM MEME CLASSIFICATION - DUAL MODEL INFERENCE")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Device: {device}")
    
    # CLIP Prediction
    print("\n[1] CLIP Model Prediction...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPMemeClassifier(num_classes=2).to(device)
    clip_model.load_state_dict(torch.load(clip_model_path, map_location=device))
    clip_model.eval()
    
    clip_inputs = clip_processor(images=image, return_tensors="pt")
    clip_pixels = clip_inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        clip_outputs = clip_model(pixel_values=clip_pixels)
        clip_probs = torch.softmax(clip_outputs, dim=1)
        clip_pred = clip_outputs.argmax(1).item()
    
    print(f"    Predicted: {labels[clip_pred]}")
    print(f"    Confidence: {clip_probs[0][clip_pred].item()*100:.2f}%")
    print(f"    Probabilities: TROLL/OPPOSE={clip_probs[0][0].item()*100:.2f}%, SUPPORT={clip_probs[0][1].item()*100:.2f}%")
    
    # ViT Prediction
    print("\n[2] ViT Model Prediction...")
    vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    vit_model = ViTMemeClassifier(num_classes=2).to(device)
    vit_model.load_state_dict(torch.load(vit_model_path, map_location=device))
    vit_model.eval()
    
    vit_inputs = vit_processor(images=image, return_tensors="pt")
    vit_pixels = vit_inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        vit_outputs = vit_model(pixel_values=vit_pixels)
        vit_probs = torch.softmax(vit_outputs, dim=1)
        vit_pred = vit_outputs.argmax(1).item()
    
    print(f"    Predicted: {labels[vit_pred]}")
    print(f"    Confidence: {vit_probs[0][vit_pred].item()*100:.2f}%")
    print(f"    Probabilities: TROLL/OPPOSE={vit_probs[0][0].item()*100:.2f}%, SUPPORT={vit_probs[0][1].item()*100:.2f}%")
    
    # Ensemble
    print("\n[3] Ensemble Prediction (Average)...")
    ensemble_probs = (clip_probs[0] + vit_probs[0]) / 2
    ensemble_pred = ensemble_probs.argmax().item()
    
    print(f"    Predicted: {labels[ensemble_pred]}")
    print(f"    Confidence: {ensemble_probs[ensemble_pred].item()*100:.2f}%")
    print(f"    Probabilities: TROLL/OPPOSE={ensemble_probs[0].item()*100:.2f}%, SUPPORT={ensemble_probs[1].item()*100:.2f}%")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"CLIP:     {labels[clip_pred]} ({clip_probs[0][clip_pred].item()*100:.2f}%)")
    print(f"ViT:      {labels[vit_pred]} ({vit_probs[0][vit_pred].item()*100:.2f}%)")
    print(f"Ensemble: {labels[ensemble_pred]} ({ensemble_probs[ensemble_pred].item()*100:.2f}%)")
    
    if clip_pred == vit_pred:
        print(f"\n✓ Both models agree: {labels[clip_pred]}")
    else:
        print(f"\n⚠ Models disagree! CLIP: {labels[clip_pred]}, ViT: {labels[vit_pred]}")
    
    print("="*60 + "\n")
    
    return {
        'clip': (clip_pred, clip_probs[0].cpu().numpy()),
        'vit': (vit_pred, vit_probs[0].cpu().numpy()),
        'ensemble': (ensemble_pred, ensemble_probs.cpu().numpy())
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_both.py \"<image_path>\"")
        print('Example: python inference_both.py "D:\\path\\to\\image.jpg"')
        sys.exit(1)
    
    image_path = ' '.join(sys.argv[1:])
    predict_both(image_path)
