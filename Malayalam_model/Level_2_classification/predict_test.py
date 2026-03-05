import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
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
TRAIN_LABELS = os.path.join(BASE_DIR, 'Dataset', 'Train-20260214T175142Z-1-001', 'Train', 'Malayalam_Train_label.xlsx')

class CombinedModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, clip_inputs, vit_inputs):
        clip_features = self.clip.get_image_features(pixel_values=clip_inputs)
        if not isinstance(clip_features, torch.Tensor):
            clip_features = clip_features.pooler_output
        vit_features = self.vit(pixel_values=vit_inputs).last_hidden_state[:, 0, :]
        combined = torch.cat([clip_features, vit_features], dim=1)
        return self.classifier(combined)

class TestDataset(Dataset):
    def __init__(self, df, img_dir, clip_processor, vit_processor):
        self.df = df
        self.img_dir = img_dir
        self.clip_processor = clip_processor
        self.vit_processor = vit_processor
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['meme_id']}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        vit_inputs = self.vit_processor(images=image, return_tensors="pt")
        
        return clip_inputs['pixel_values'].squeeze(0), vit_inputs['pixel_values'].squeeze(0), row['meme_id']

print("="*80)
print("MALAYALAM LEVEL 2 - TEST PREDICTION")
print("="*80)

print("\n[1] Loading datasets...")
test_df = pd.read_excel(TEST_LABELS)
train_df = pd.read_excel(TRAIN_LABELS)
print(f"Test samples: {len(test_df)}")

# Get label mapping from training data
label_mapping = {label: idx for idx, label in enumerate(sorted(train_df['Level 2'].unique()))}
reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
print(f"\nLabel mapping: {label_mapping}")

print("\n[2] Loading model...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

model = CombinedModel(num_classes=len(label_mapping)).to(device)

# Try to load the best model
model_paths = ['best_combined_model.pth', 'best_model.pth']
model_loaded = False
for model_path in model_paths:
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")
        model_loaded = True
        break

if not model_loaded:
    print("ERROR: No trained model found! Please train Level 2 model first.")
    exit(1)

model.eval()

print("\n[3] Running predictions...")
test_dataset = TestDataset(test_df, TEST_DIR, clip_processor, vit_processor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

predictions = []
meme_ids = []

with torch.no_grad():
    for clip_imgs, vit_imgs, ids in tqdm(test_loader, desc="Predicting"):
        clip_imgs = clip_imgs.to(device)
        vit_imgs = vit_imgs.to(device)
        outputs = model(clip_imgs, vit_imgs)
        _, predicted = outputs.max(1)
        predictions.extend(predicted.cpu().numpy())
        meme_ids.extend(ids.numpy())

test_df['Level 2'] = [reverse_label_mapping[p] for p in predictions]

output_path = os.path.join(BASE_DIR, 'Dataset', 'Test-20260214T175144Z-1-001', 'Test', 'Malayalam_Test_label_Level2_predicted.xlsx')
test_df.to_excel(output_path, index=False)

print(f"\n[4] Predictions saved to: {output_path}")
print(f"\nPrediction distribution:")
print(test_df['Level 2'].value_counts())
print("\n" + "="*80)
print("PREDICTION COMPLETE!")
print("="*80)
