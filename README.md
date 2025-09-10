# Plant Disease Classification Project

A deep learning project for classifying plant diseases using the PlantVillage dataset. This project implements three different approaches using PyTorch and MobileNetV2 architecture to classify plant diseases from segmented, grayscale, and color images.

## 🌱 Project Overview

This project focuses on automated plant disease detection and classification using computer vision techniques. It leverages the PlantVillage dataset to train models that can identify various plant diseases across different plant species including apples, corn, grapes, tomatoes, and many others.

## 📊 Dataset

The project uses the **PlantVillage dataset** which contains:
- **38 different plant disease classes**
- **Three image variants:**
  - **Segmented**: Pre-processed segmented images (13,798 images)
  - **Grayscale**: Grayscale converted images 
  - **Color**: Original color images

### Plant Species and Diseases Covered
- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Blueberry**: Healthy
- **Cherry**: Healthy, Powdery mildew
- **Corn (Maize)**: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- **Grape**: Black rot, Esca (Black Measles), Leaf blight, Healthy
- **Orange**: Huanglongbing (Citrus greening)
- **Peach**: Bacterial spot, Healthy
- **Pepper**: Bacterial spot, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Powdery mildew
- **Strawberry**: Healthy, Leaf scorch
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Tomato mosaic virus, Tomato Yellow Leaf Curl Virus, Healthy

## 🚀 Features

- **Three Model Variants**: Trained on segmented, grayscale, and color images
- **Transfer Learning**: Uses pre-trained MobileNetV2 with ImageNet weights
- **Mixed Precision Training**: Utilizes PyTorch's Automatic Mixed Precision (AMP) for faster training
- **Data Augmentation**: Random horizontal flips and rotations for better generalization
- **Comprehensive Evaluation**: Detailed classification reports and performance metrics
- **TensorBoard Integration**: Training progress visualization
- **Model Checkpointing**: Automatic saving of best performing models

## 🛠️ Technical Details

### Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224 pixels
- **Number of Classes**: 38
- **Optimizer**: AdamW with learning rate 1e-3 and weight decay 1e-4
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 20

### Data Preprocessing
- **Training Augmentations**:
  - Resize to 224x224
  - Random horizontal flip
  - Random rotation (10 degrees)
  - Normalization using ImageNet statistics
- **Validation**: Only resize and normalization (no augmentation)

### Performance Results

| Model Type | Best Validation Accuracy | Final Training Accuracy |
|------------|-------------------------|------------------------|
| Segmented  | 98.67%                 | 99.05%                 |
| Grayscale  | 97.23%                 | 98.34%                 |
| Color      | 99.43%                 | 99.09%                 |

## 📁 Project Structure

```
plant-classification/
├── model.ipynb                    # Main training notebook
├── plantvillage dataset/
│   ├── segmented/                 # Segmented images (13,798 files)
│   ├── grayscale/                 # Grayscale images
│   └── color/                     # Color images
├── plant_segmented_model.pth      # Best segmented model checkpoint
├── plant_grayscale_model.pth      # Best grayscale model checkpoint
├── plant_color_model.pth          # Best color model checkpoint
├── logs_pt_segmented/             # TensorBoard logs for segmented model
├── logs_pt_grayscale/             # TensorBoard logs for grayscale model
├── logs_pt_color/                 # TensorBoard logs for color model
└── README.md                      # This file
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install scikit-learn
pip install matplotlib
pip install tensorboard
pip install jupyter
```

### Dataset Setup
1. Download the PlantVillage dataset
2. Extract the dataset to the `plantvillage dataset/` directory
3. Ensure the folder structure matches the expected layout

## 🚀 Usage

### Training Models
1. Open `model.ipynb` in Jupyter Notebook
2. Run all cells to train all three model variants
3. Monitor training progress with TensorBoard:
   ```bash
   tensorboard --logdir=logs_pt_segmented
   tensorboard --logdir=logs_pt_grayscale
   tensorboard --logdir=logs_pt_color
   ```

### Model Evaluation
The notebook includes comprehensive evaluation with:
- Classification reports for each model
- Precision, recall, and F1-score metrics
- Comparative accuracy and loss plots

### Loading Trained Models
```python
import torch
from torchvision import models

# Load model checkpoint
checkpoint = torch.load('plant_color_model.pth')
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 38)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 📈 Results Analysis

### Key Findings
1. **Color images perform best** with 99.43% validation accuracy
2. **Segmented images** show excellent performance (98.67%) with potential for edge deployment
3. **Grayscale images** achieve good results (97.23%) with reduced computational requirements

### Model Performance by Plant Type
- **High accuracy (>99%)**: Apple, Blueberry, Cherry, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Strawberry
- **Good accuracy (95-99%)**: Corn, Squash, Tomato

## 🔬 Technical Implementation

### Mixed Precision Training
The project uses PyTorch's Automatic Mixed Precision (AMP) to:
- Reduce memory usage
- Speed up training
- Maintain numerical stability

### Data Splitting
- **Training**: 80% of data
- **Validation**: 20% of data
- **Stratified splitting** ensures balanced representation across all classes

### Model Architecture
```python
# MobileNetV2 with custom classifier
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(in_features, num_classes)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure you have the appropriate licenses for the PlantVillage dataset.

## 🙏 Acknowledgments

- **PlantVillage Dataset**: For providing the comprehensive plant disease dataset
- **PyTorch Team**: For the excellent deep learning framework
- **MobileNetV2**: For the efficient architecture design

## 📞 Contact

For questions or suggestions about this project, please open an issue in the repository.

---

**Note**: This project is designed for educational purposes and should not be used as the sole method for plant disease diagnosis in agricultural settings. Always consult with agricultural experts for real-world applications.
