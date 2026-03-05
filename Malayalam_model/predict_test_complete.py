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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, 'Dataset', 'Test-20260214T175144Z-1-001', 'Test', 'Test_images')
TEST_LABELS = os.path.join(BASE_DIR, 'Dataset', 'Test-20260214T175144Z-1-001', 'Test', 'Malayalam_Test_label.xlsx')
TRAIN_LABELS = os.path.join(BASE_DIR, 'Dataset', 'Train-20260214T175142Z-1-001', 'Train', 'Malayalam_Train_label.xlsx')

# Level 1 Model
class CLIPLevel1Classifier(nn.Module):
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

# Level 2 Model
class CombinedLevel2Model(nn.Module):
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
        meme_id = int(row['meme_id'])
        img_path = os.path.join(self.img_dir, f"{meme_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        vit_inputs = self.vit_processor(images=image, return_tensors="pt")
        
        return clip_inputs['pixel_values'].squeeze(0), vit_inputs['pixel_values'].squeeze(0), meme_id

print("="*80)
print("MALAYALAM COMPLETE TEST PREDICTION - LEVEL 1 & 2")
print("="*80)

print("\n[1] Loading datasets...")
test_df = pd.read_excel(TEST_LABELS)
train_df = pd.read_excel(TRAIN_LABELS)
print(f"Test samples: {len(test_df)}")

print("\n[2] Loading processors...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# LEVEL 1 PREDICTION
print("\n[3] Loading Level 1 model...")
level1_model = CLIPLevel1Classifier(num_classes=2).to(device)
level1_path = os.path.join(BASE_DIR, 'Malayalam_model', 'Level_1_classification', 'best_clip_model.pth')
if not os.path.exists(level1_path):
    level1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Level_1_classification', 'best_clip_model.pth')
if os.path.exists(level1_path):
    level1_model.load_state_dict(torch.load(level1_path, map_location=device))
    print("Level 1 model loaded!")
else:
    print(f"ERROR: Level 1 model not found. Please train the model first:")
    print(f"  cd Level_1_classification")
    print(f"  python train_clip.py")
    exit(1)

level1_model.eval()

print("\n[4] Running Level 1 predictions...")
test_dataset = TestDataset(test_df, TEST_DIR, clip_processor, vit_processor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

level1_predictions = []
level1_confidences = []
with torch.no_grad():
    for clip_imgs, _, ids in tqdm(test_loader, desc="Level 1"):
        clip_imgs = clip_imgs.to(device)
        outputs = level1_model(pixel_values=clip_imgs)
        probs = torch.softmax(outputs, dim=1)
        confidences, predicted = probs.max(1)
        level1_predictions.extend(predicted.cpu().numpy())
        level1_confidences.extend(confidences.cpu().numpy())

level1_map = {0: 'TROLL/ OPPOSE', 1: 'SUPPORT'}
test_df['Level 1'] = [level1_map[p] for p in level1_predictions]
test_df['Level 1 Confidence'] = [f"{c:.4f}" for c in level1_confidences]

print(f"\nLevel 1 distribution:")
print(test_df['Level 1'].value_counts())

# LEVEL 2 PREDICTION
print("\n[5] Loading Level 2 model...")
label_mapping = {label: idx for idx, label in enumerate(sorted(train_df['Level 2'].unique()))}
reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}

level2_model = CombinedLevel2Model(num_classes=len(label_mapping)).to(device)
level2_paths = [
    os.path.join(BASE_DIR, 'Malayalam_model', 'Level_2_classification', 'best_combined_model.pth'),
    os.path.join(BASE_DIR, 'Malayalam_model', 'Level_2_classification', 'best_model.pth'),
    os.path.join(BASE_DIR, 'Malayalam_model', 'Level_2_classification', 'best_vit_level2_model.pth'),
    os.path.join(BASE_DIR, 'Malayalam_model', 'Level_2_classification', 'best_clip_level2_model.pth'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Level_2_classification', 'best_combined_model.pth'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Level_2_classification', 'best_vit_level2_model.pth')
]

level2_loaded = False
for level2_path in level2_paths:
    if os.path.exists(level2_path):
        try:
            checkpoint = torch.load(level2_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                level2_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                level2_model.load_state_dict(checkpoint, strict=False)
            print(f"Level 2 model loaded from {level2_path}")
            level2_loaded = True
            break
        except Exception as e:
            print(f"Failed to load {level2_path}: {e}")
            continue

if not level2_loaded:
    print("WARNING: Level 2 model not found. Filling with default values.")
    test_df['Level 2'] = 'Intersection'
else:
    level2_model.eval()
    
    print("\n[6] Running Level 2 predictions...")
    level2_predictions = []
    level2_confidences = []
    with torch.no_grad():
        for clip_imgs, vit_imgs, ids in tqdm(test_loader, desc="Level 2"):
            clip_imgs = clip_imgs.to(device)
            vit_imgs = vit_imgs.to(device)
            outputs = level2_model(clip_imgs, vit_imgs)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)
            level2_predictions.extend(predicted.cpu().numpy())
            level2_confidences.extend(confidences.cpu().numpy())
    
    test_df['Level 2'] = [reverse_label_mapping[p] for p in level2_predictions]
    test_df['Level 2 Confidence'] = [f"{c:.4f}" for c in level2_confidences]
    
    print(f"\nLevel 2 distribution:")
    print(test_df['Level 2'].value_counts())

# Save results
output_path = os.path.join(BASE_DIR, 'Dataset', 'Test-20260214T175144Z-1-001', 'Test', 'Malayalam_Test_label_PREDICTED.xlsx')
test_df.to_excel(output_path, index=False)

print(f"\n[7] Complete predictions saved to:")
print(f"    {output_path}")
print("\n" + "="*80)
print("PREDICTION COMPLETE!")
print("="*80)
