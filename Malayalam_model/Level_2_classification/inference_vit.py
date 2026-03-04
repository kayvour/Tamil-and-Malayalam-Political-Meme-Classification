import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_image(image_path, model_path='best_vit_level2_model.pth'):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading ViT Level 2 model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    label_map = checkpoint['label_map']
    reverse_map = {v: k for k, v in label_map.items()}
    
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=len(label_map),
        ignore_mismatched_sizes=True
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values).logits
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(1).item()
    
    print("\n" + "="*50)
    print("ViT LEVEL 2 MODEL PREDICTION")
    print("="*50)
    print(f"Predicted Class: {reverse_map[pred_class]}")
    print(f"Confidence: {probs[0][pred_class].item()*100:.2f}%")
    print(f"\nProbabilities:")
    for idx, label in sorted(reverse_map.items()):
        print(f"  {label}: {probs[0][idx].item()*100:.2f}%")
    print("="*50)
    
    return pred_class, probs[0].cpu().numpy()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_vit.py \"<image_path>\"")
        print('Example: python inference_vit.py "path/to/image.jpg"')
        sys.exit(1)
    
    image_path = ' '.join(sys.argv[1:])
    predict_image(image_path)
