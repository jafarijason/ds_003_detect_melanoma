# Melanoma Detection using Convolutional Neural Network Case Study

## Project Overview
The goal of this project is to develop a CNN-based model capable of accurately detecting melanoma, a potentially fatal form of cancer if not identified early. Melanoma accounts for 75% of skin cancer-related deaths. By creating a system that analyzes images and alerts dermatologists to its presence, this solution could significantly reduce the manual effort required for diagnosis.

## Table of Contents
- [General Information](#general-information)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Model Evaluation](#model-evaluation)
- [Conclusions](#conclusions)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## General Information
- **Background**
  - Melanoma is one of the deadliest forms of skin cancer, and early detection is crucial for effective treatment.
  - Current diagnostic methods rely on visual examination and histopathological analysis, which can be time-consuming.
  - Machine learning, particularly deep learning with Convolutional Neural Networks (CNNs), offers a promising approach for automated melanoma detection.

- **Business Problem**
  - The project aims to reduce the manual effort involved in melanoma diagnosis by developing an automated image classification system.
  - This model can assist dermatologists in early-stage melanoma detection, potentially saving lives by improving diagnostic accuracy and efficiency.

## Dataset
The dataset consists of **2,357 images** of malignant and benign oncological diseases. The dataset is categorized into the following skin conditions:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

To handle class imbalance, data augmentation was performed using the **Augmentor** library.

## Technologies Used
- **Python**
- **TensorFlow**
- **Keras**
- **Google Colab**
- **Augmentor**

## Model Architecture
The custom CNN model includes the following layers:
- **Rescaling Layer**: Normalizes pixel values to the range [0,1].
- **Convolutional Layers**: Extracts spatial features from images.
- **Pooling Layers**: Reduces the dimensionality of feature maps.
- **Dropout Layer**: Prevents overfitting by randomly dropping units.
- **Flatten Layer**: Converts feature maps into a single vector.
- **Dense Layers**: Fully connected layers for classification.
- **Activation Functions**: ReLU for hidden layers and Softmax for output layer.

### Model Summary:
- **Input Image Size**: 180x180 pixels
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Number of Epochs**: 30

## Model Evaluation
The final model achieved the following results:
- **Validation Accuracy**: ~84%
- **Validation Loss**: 0.56

## Conclusions
- A CNN-based model can effectively classify melanoma and other skin conditions with high accuracy.
- Data augmentation helped address class imbalance, improving model performance.
- The trained model can assist dermatologists in early melanoma detection, potentially improving patient outcomes.

## Acknowledgements
- **Dataset Source**: [source](https://drive.google.com/file/d/1xLfSQUGDl8ezNNbUkpuHOYvSpTyxVhCs/view?usp=sharing)
- **Colab Code**: [source](https://colab.research.google.com/drive/1SMECFUcTNdceSLVjAcWz4nXckpoHxtbi?usp=sharing)

## Contact
Created by **[@jafarijason]** - feel free to reach out!


## How to run project
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

```
jupyter notebook \
    --notebook-dir="." \
    --ip=0.0.0.0 --port=3225
```

