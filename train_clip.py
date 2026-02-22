import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

# Paths
TRAIN_DIR = r'D:\Data\Speech and Language Processing Project\Train-20260214T175134Z-1-001\Train\Train_images'
TRAIN_LABELS = r'D:\Data\Speech and Language Processing Project\Train-20260214T175134Z-1-001\Train\Train_labels.xlsx'

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-5
NUM_WORKERS = 0

class CLIPMemeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        print("Loading CLIP base model...")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        print("Building classifier head...")
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

class MemeDataset(Dataset):
    def __init__(self, df, img_dir, processor):
        self.df = df
        self.img_dir = img_dir
        self.processor = processor
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image_name'])
        image = Image.open(img_path).convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        label = 0 if row['Level1'] == 'Troll/Oppose' else 1
        return pixel_values, label

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(pixel_values=images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels

def main():
    print("="*80)
    print("MALAYALAM MEME CLASSIFICATION - CLIP TRAINING")
    print("="*80)
    
    # Load data
    print("\n[1] Loading training dataset...")
    df = pd.read_excel(TRAIN_LABELS)
    print(f"Total samples: {len(df)}")
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Level1'], random_state=42)
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}")
    
    # Load CLIP
    print("\n[2] Loading CLIP model...")
    print("Downloading CLIP model (this may take a few minutes on first run)...")
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", resume_download=True)
        model = CLIPMemeClassifier(num_classes=2).to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Try running again or check your internet connection.")
        return
    
    # Create datasets
    print("\n[3] Creating data loaders...")
    train_dataset = MemeDataset(train_df, TRAIN_DIR, processor)
    val_dataset = MemeDataset(val_df, TRAIN_DIR, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=NUM_WORKERS, pin_memory=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    print("\n[4] Training model...")
    best_acc = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_clip_model.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    # Final evaluation
    print("\n[5] Final Evaluation")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=['TROLL/OPPOSE', 'SUPPORT']))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
