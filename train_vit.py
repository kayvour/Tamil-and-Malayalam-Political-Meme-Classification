import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input size

# Paths
TRAIN_DIR = r'D:\Data\Speech and Language Processing Project\Train-20260214T175134Z-1-001\Train\Train_images'
TRAIN_LABELS = r'D:\Data\Speech and Language Processing Project\Train-20260214T175134Z-1-001\Train\Train_labels.xlsx'
TEST_DIR = r'D:\Data\Speech and Language Processing Project\Test-20260214T175144Z-1-001\Test\Test_images'

# Hyperparameters
BATCH_SIZE = 8  # Reduced for 8GB GPU
NUM_EPOCHS = 5 if not torch.cuda.is_available() else 20
LEARNING_RATE = 2e-5
IMG_SIZE = 224
NUM_WORKERS = 0 if not torch.cuda.is_available() else 2  # Reduced workers

class MemeDataset(Dataset):
    def __init__(self, df, img_dir, processor, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.processor = processor
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        if self.is_test:
            return pixel_values, row['Image_name']
        
        # Level 1: Binary classification
        label = 0 if row['Level1'] == 'Troll/Oppose' else 1
        return pixel_values, label

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
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
            
            outputs = model(pixel_values=images).logits
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
    print("MALAYALAM MEME CLASSIFICATION - ViT TRAINING")
    print("="*80)
    
    # Load data
    print("\n[1] Loading training dataset...")
    df = pd.read_excel(TRAIN_LABELS)
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['Level1'].value_counts()}")
    
    # Train-validation split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Level1'], random_state=42)
    print(f"\nTrain: {len(train_df)}, Validation: {len(val_df)}")
    
    # Load ViT model
    print("\n[2] Loading Vision Transformer model...")
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    
    # Enable mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler()
    
    # Create datasets
    print("\n[3] Creating data loaders...")
    train_dataset = MemeDataset(train_df, TRAIN_DIR, processor)
    val_dataset = MemeDataset(val_df, TRAIN_DIR, processor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True  # Faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
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
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_vit_model.pth')
            print(f"[SAVED] Best model (Val Acc: {val_acc:.2f}%)")
    
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
