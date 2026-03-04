# Malayalam Meme Dataset - Exploratory Data Analysis

Complete EDA for Malayalam Meme Classification Dataset with visualizations and insights.

## 📁 Files

- **malayalam_eda.py** - Python script for complete EDA
- **malayalam_eda.ipynb** - Jupyter notebook for interactive analysis
- **README.md** - This file

## 📊 Generated Outputs

After running the EDA, the following files will be created:

1. **label_distribution.png** - Bar charts showing Level 1 and Level 2 label distributions
2. **class_imbalance.png** - Pie charts showing class imbalance
3. **image_properties.png** - Histograms and scatter plots of image dimensions and sizes
4. **sample_images.png** - Grid of 18 sample memes with labels
5. **label_correlation.png** - Heatmap showing Level 1 vs Level 2 correlation
6. **summary_statistics.csv** - Summary statistics in CSV format

## 🚀 Usage

### Run Python Script
```bash
python malayalam_eda.py
```

### Run Jupyter Notebook
```bash
jupyter notebook malayalam_eda.ipynb
```

## 📈 Analysis Performed

### 1. Data Loading
- Load training dataset from Excel
- Display basic statistics and structure
- Show first 5 rows

### 2. Label Distribution Analysis
- Count samples per class for Level 1 and Level 2
- Visualize distributions with bar charts
- Add count labels on bars

### 3. Class Imbalance Analysis
- Calculate imbalance ratio for Level 1
- Create pie charts for both levels
- Show percentage distribution

### 4. Image Properties Analysis
- Extract width, height, file size, aspect ratio
- Calculate average values
- Create histograms for each property
- Scatter plot of width vs height colored by aspect ratio

### 5. Sample Images Visualization
- Display 18 sample memes in a grid
- Show Level 1 and Level 2 labels for each
- Visual inspection of dataset

### 6. Label Correlation Analysis
- Create crosstab of Level 1 vs Level 2
- Visualize with heatmap
- Identify label combinations

### 7. Data Quality Check
- Check for missing values
- Check for duplicate rows
- Verify all images exist

### 8. Summary Statistics
- Compile key metrics
- Save to CSV file
- Display in formatted table

## 🔍 Key Findings

### Dataset Overview
- **Total Samples**: 500 images
- **Level 1 Classes**: 2 (Troll/Oppose, Support)
- **Level 2 Classes**: 5 categories

### Class Imbalance
- **Severe imbalance** in Level 1:
  - Troll/Oppose: ~477 samples (95.4%)
  - Support: ~23 samples (4.6%)
- **Imbalance Ratio**: ~20.7:1

### Image Properties
- **Average Width**: ~994 px
- **Average Height**: ~1142 px
- **Average File Size**: ~196 KB
- **Average Aspect Ratio**: ~0.89
- **Variable dimensions** - requires resizing for model training

## 💡 Recommendations

### Preprocessing
1. **Resize images** to uniform size (224x224 or 299x299)
2. **Normalize** pixel values to [0, 1]
3. **Apply data augmentation**:
   - Random rotation
   - Horizontal flip
   - Brightness/contrast adjustment
   - Random crop

### Handling Class Imbalance
1. **Use class weights** in loss function
2. **Apply oversampling** for minority class
3. **Consider focal loss**
4. **Stratified train-validation split**

### Model Recommendations
1. **Transfer learning** with pre-trained models (ResNet, EfficientNet, ViT)
2. **Multi-task learning** for both Level 1 and Level 2
3. **Ensemble methods** for better performance

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
pillow
openpyxl
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn pillow openpyxl
```

## 📝 Notes

- All visualizations are saved at 300 DPI for high quality
- Images are analyzed without loading all into memory
- Script handles missing images gracefully
- Output directory is created automatically

## 🎯 Next Steps

1. ✅ Complete EDA (DONE)
2. Implement data augmentation pipeline
3. Create train-validation split (stratified)
4. Build baseline model
5. Experiment with transfer learning
6. Evaluate on validation set
7. Generate predictions for test set

---

**Dataset**: Malayalam Meme Classification  
**Language**: Malayalam  
**Task**: Binary (Level 1) and Multi-class (Level 2) Classification
