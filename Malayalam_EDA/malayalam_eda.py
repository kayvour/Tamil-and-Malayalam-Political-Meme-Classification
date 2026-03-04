import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from pathlib import Path

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'Dataset', 'Train-20260214T175134Z-1-001', 'Train', 'Train_images')
TRAIN_LABELS = os.path.join(BASE_DIR, 'Dataset', 'Train-20260214T175134Z-1-001', 'Train', 'Train_labels.xlsx')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Malayalam_EDA')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("MALAYALAM MEME DATASET - EXPLORATORY DATA ANALYSIS")
print("="*80)

# 1. Load Data
print("\n[1] LOADING DATA")
df = pd.read_excel(TRAIN_LABELS)
print(f"Total samples: {len(df)}")
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# 2. Label Distribution
print("\n[2] LABEL DISTRIBUTION ANALYSIS")
print(f"\nLevel 1 Distribution:\n{df['Level1'].value_counts()}")
print(f"\nLevel 2 Distribution:\n{df['Level2'].value_counts()}")

# Plot distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['Level1'].value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#3498db'])
axes[0].set_title('Level 1 Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(df['Level1'].value_counts()):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

df['Level2'].value_counts().plot(kind='bar', ax=axes[1], color=sns.color_palette("husl", 5))
axes[1].set_title('Level 2 Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(df['Level2'].value_counts()):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'label_distribution.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: label_distribution.png")
plt.close()

# 3. Class Imbalance
print("\n[3] CLASS IMBALANCE ANALYSIS")
level1_counts = df['Level1'].value_counts()
imbalance_ratio = level1_counts.max() / level1_counts.min()
print(f"Imbalance Ratio (Level 1): {imbalance_ratio:.2f}:1")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors1 = ['#e74c3c', '#3498db']
axes[0].pie(level1_counts, labels=level1_counts.index, autopct='%1.1f%%', 
            colors=colors1, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0].set_title('Level 1 Class Distribution', fontsize=14, fontweight='bold')

colors2 = sns.color_palette("husl", len(df['Level2'].unique()))
axes[1].pie(df['Level2'].value_counts(), labels=df['Level2'].value_counts().index, 
            autopct='%1.1f%%', colors=colors2, startangle=90, textprops={'fontsize': 10})
axes[1].set_title('Level 2 Class Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'class_imbalance.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: class_imbalance.png")
plt.close()

# 4. Image Properties
print("\n[4] IMAGE PROPERTIES ANALYSIS")
widths, heights, sizes, aspect_ratios = [], [], [], []

for img_name in df['Image_name']:
    img_path = os.path.join(TRAIN_DIR, img_name)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        widths.append(img.width)
        heights.append(img.height)
        sizes.append(os.path.getsize(img_path) / 1024)  # KB
        aspect_ratios.append(img.width / img.height)

print(f"Average Width: {np.mean(widths):.2f} px")
print(f"Average Height: {np.mean(heights):.2f} px")
print(f"Average File Size: {np.mean(sizes):.2f} KB")
print(f"Average Aspect Ratio: {np.mean(aspect_ratios):.2f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].hist(widths, bins=30, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Image Width Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Width (px)')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(heights, bins=30, color='lightcoral', edgecolor='black')
axes[0, 1].set_title('Image Height Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Height (px)')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(sizes, bins=30, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('File Size Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Size (KB)')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].scatter(widths, heights, alpha=0.5, c=aspect_ratios, cmap='viridis')
axes[1, 1].set_title('Width vs Height', fontweight='bold')
axes[1, 1].set_xlabel('Width (px)')
axes[1, 1].set_ylabel('Height (px)')
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Aspect Ratio')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'image_properties.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: image_properties.png")
plt.close()

# 5. Sample Images
print("\n[5] SAMPLE IMAGES VISUALIZATION")
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for i, ax in enumerate(axes.flat):
    if i < len(df):
        img_path = os.path.join(TRAIN_DIR, df.iloc[i]['Image_name'])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"{df.iloc[i]['Level1']}\n{df.iloc[i]['Level2']}", fontsize=8)
            ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: sample_images.png")
plt.close()

# 6. Label Correlation
print("\n[6] LABEL CORRELATION ANALYSIS")
correlation_matrix = pd.crosstab(df['Level1'], df['Level2'])
print(f"\nCorrelation Matrix:\n{correlation_matrix}")

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
plt.title('Level 1 vs Level 2 Correlation', fontsize=14, fontweight='bold')
plt.xlabel('Level 2')
plt.ylabel('Level 1')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'label_correlation.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: label_correlation.png")
plt.close()

# 7. Data Quality Check
print("\n[7] DATA QUALITY CHECK")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Check if all images exist
missing_images = []
for img_name in df['Image_name']:
    if not os.path.exists(os.path.join(TRAIN_DIR, img_name)):
        missing_images.append(img_name)
print(f"Missing images: {len(missing_images)}")

# 8. Summary Statistics
print("\n[8] SUMMARY STATISTICS")
summary = pd.DataFrame({
    'Metric': ['Total Samples', 'Level 1 Classes', 'Level 2 Classes', 
               'Avg Width (px)', 'Avg Height (px)', 'Avg Size (KB)', 
               'Imbalance Ratio'],
    'Value': [len(df), df['Level1'].nunique(), df['Level2'].nunique(),
              f"{np.mean(widths):.2f}", f"{np.mean(heights):.2f}", 
              f"{np.mean(sizes):.2f}", f"{imbalance_ratio:.2f}:1"]
})
print(f"\n{summary.to_string(index=False)}")

# Save summary
summary.to_csv(os.path.join(OUTPUT_DIR, 'summary_statistics.csv'), index=False)
print(f"\n✓ Saved: summary_statistics.csv")

print("\n" + "="*80)
print("EDA COMPLETE! All visualizations saved to:", OUTPUT_DIR)
print("="*80)
