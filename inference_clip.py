import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
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

def predict_image(image_path, model_path='best_clip_model.pth'):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading CLIP model from {model_path}...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPMemeClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(1).item()
    
    labels = ['TROLL/OPPOSE', 'SUPPORT']
    print("\n" + "="*50)
    print("CLIP MODEL PREDICTION")
    print("="*50)
    print(f"Predicted Class: {labels[pred_class]}")
    print(f"Confidence: {probs[0][pred_class].item()*100:.2f}%")
    print(f"\nProbabilities:")
    print(f"  TROLL/OPPOSE: {probs[0][0].item()*100:.2f}%")
    print(f"  SUPPORT: {probs[0][1].item()*100:.2f}%")
    print("="*50)
    
    return pred_class, probs[0].cpu().numpy()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_clip.py \"<image_path>\"")
        print('Example: python inference_clip.py "D:\\path\\to\\image.jpg"')
        sys.exit(1)
    
    image_path = ' '.join(sys.argv[1:])
    predict_image(image_path)
