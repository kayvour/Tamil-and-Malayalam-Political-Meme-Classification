import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = os.path.join(BASE_DIR, 'Dataset', 'Test-20260214T175144Z-1-001', 'Test', 'Test_images')
TEST_LABELS = os.path.join(BASE_DIR, 'Dataset', 'Test-20260214T175144Z-1-001', 'Test', 'Malayalam_Test_label.xlsx')
MODEL_PATH = 'best_clip_model.pth'

class CLIPMemeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, pixel_values):
        outputs = self.clip.get_image_features(pixel_values=pixel_values)
        image_features = outputs if isinstance(outputs, torch.Tensor) else outputs.last_hidden_state[:, 0, :]
        return self.classifier(image_features)

class TestDataset(Dataset):
    def __init__(self, df, img_dir, processor):
        self.df = df
        self.img_dir = img_dir
        self.processor = processor
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['meme_id']}.jpg")
        image = Image.open(img_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), row['meme_id']

print("="*80)
print("MALAYALAM LEVEL 1 - TEST PREDICTION")
print("="*80)

print("\n[1] Loading test dataset...")
df = pd.read_excel(TEST_LABELS)
print(f"Test samples: {len(df)}")

print("\n[2] Loading model...")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPMemeClassifier(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded successfully!")

print("\n[3] Running predictions...")
test_dataset = TestDataset(df, TEST_DIR, processor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

predictions = []
meme_ids = []

with torch.no_grad():
    for images, ids in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)
        outputs = model(pixel_values=images)
        _, predicted = outputs.max(1)
        predictions.extend(predicted.cpu().numpy())
        meme_ids.extend(ids.numpy())

label_map = {0: 'TROLL/ OPPOSE', 1: 'SUPPORT'}
df['Level 1'] = [label_map[p] for p in predictions]

output_path = os.path.join(BASE_DIR, 'Dataset', 'Test-20260214T175144Z-1-001', 'Test', 'Malayalam_Test_label_Level1_predicted.xlsx')
df.to_excel(output_path, index=False)

print(f"\n[4] Predictions saved to: {output_path}")
print(f"\nPrediction distribution:")
print(df['Level 1'].value_counts())
print("\n" + "="*80)
print("PREDICTION COMPLETE!")
print("="*80)
