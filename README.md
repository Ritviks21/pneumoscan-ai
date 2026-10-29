
# 🫁 PneumoScan AI
### Pediatric Chest X-Ray Pneumonia Classification Using Deep Learning

**A comparative study of a baseline CNN, frozen MobileNetV2, and fine-tuned MobileNetV2**

PneumoScan AI is a computer vision research project that compares three deep-learning models for binary classification of pediatric chest X-ray images into **Normal** and **Pneumonia** classes. The project investigates whether transfer learning and fine-tuning improve classification performance compared with a convolutional neural network (CNN) trained from scratch.

The project includes trained models and a Streamlit web application for interactive experimentation.

> **Disclaimer:** This is an academic research prototype, not a medical diagnostic tool. Its predictions are not clinically validated and must not be used to make medical decisions.

---

## 📌 Project Overview

Pneumonia classification from chest X-rays is a widely studied medical computer vision task. CNNs can learn visual patterns from labelled images, while transfer learning reuses features learned from larger image datasets.

This project evaluates three modelling strategies under a common experimental setup:

- **Baseline CNN:** trained from scratch on PneumoniaMNIST.
- **Frozen MobileNetV2:** uses ImageNet-pretrained features while training a new classification head.
- **Fine-tuned MobileNetV2:** adapts selected pretrained layers to the target classification task.

All three models are compared on the same held-out test split using classification metrics, ROC-AUC, confusion matrices, and additional statistical analyses.

### Research Question

Can transfer learning and fine-tuning with MobileNetV2 improve the performance and error balance of pediatric chest X-ray classification compared with a CNN trained from scratch?

---

## 🎯 Objectives

- Explore the PneumoniaMNIST benchmark dataset.
- Build a baseline CNN for binary image classification.
- Implement frozen and fine-tuned MobileNetV2 models.
- Apply model-specific image preprocessing and training-only augmentation.
- Compare performance using accuracy, precision, recall, specificity, F1-score, and ROC-AUC.
- Examine confusion matrices, bootstrap confidence intervals, paired model comparisons, and validation-selected thresholds.
- Deploy an interactive web interface to demonstrate the trained models.

---

## 🗂️ Dataset

The project uses **PneumoniaMNIST**, part of the [MedMNIST v2](https://medmnist.com/) collection.

The dataset is based on pediatric chest X-rays from Guangzhou, China, and provides two classes:

| Class | Label |
|---|---:|
| Normal | 0 |
| Pneumonia | 1 |

### Dataset Distribution

| Split | Total images | Normal | Pneumonia |
|---|---:|---:|---:|
| Training | 4,708 | 1,214 | 3,494 |
| Validation | 524 | 135 | 389 |
| Test | 624 | 234 | 390 |
| **Total** | **5,856** | **1,583** | **4,273** |

The training set is imbalanced, with approximately 74% of images belonging to the Pneumonia class. Class weighting was used during training to account for this imbalance.

### Image Preprocessing

- Configured image dimensions: **224 × 224 pixels**
- Original image representation: grayscale
- Baseline CNN: single-channel input, with pixel values rescaled internally to approximately `[0, 1]`.
- MobileNetV2: grayscale images repeated across three channels and normalized to approximately `[-1, 1]`.
- Training augmentation: small random rotations, zoom, and contrast adjustments.
- Validation and test images are not randomly augmented.

The official dataset partitions were retained for model comparison.

---

## 🧠 Models Implemented

### 1. Baseline CNN

A convolutional neural network trained from random initialization.

- Four convolutional blocks with 32, 64, 128, and 128 filters.
- 3 × 3 convolution kernels with ReLU activation.
- Max pooling after each convolutional block.
- Global average pooling.
- Dense layer with 128 units.
- Dropout rate of 0.5.
- One sigmoid output for binary classification.

**Trainable parameters:** approximately 256,897.

### 2. Frozen MobileNetV2

A transfer-learning model initialized with ImageNet-pretrained weights.

- Reuses the pretrained MobileNetV2 feature extractor.
- Keeps the convolutional base frozen.
- Uses global average pooling and dropout.
- Trains a new binary classification head.

**Total parameters:** approximately 2.26 million.  
**Trainable parameters:** 1,281.

### 3. Fine-Tuned MobileNetV2

An extension of the frozen transfer-learning model.

- Starts with ImageNet-pretrained MobileNetV2 weights.
- Unfreezes selected pretrained layers for further training.
- Uses a lower learning rate during fine-tuning.
- Adapts pretrained visual features to the chest X-ray classification task.

The aim is to improve task-specific representations while retaining useful pretrained features.

---

## 📊 Experimental Results

The following results are reported in the accompanying research paper. They correspond to the held-out test set at a sigmoid decision threshold of 0.5.

| Model | Accuracy | Precision | Recall | Specificity | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 0.803 | 0.785 | 0.944 | 0.568 | 0.857 | 0.893 |
| Frozen MobileNetV2 | 0.848 | 0.808 | **0.992** | 0.607 | 0.891 | 0.970 |
| Fine-Tuned MobileNetV2 | **0.885** | **0.850** | 0.990 | **0.709** | **0.915** | **0.983** |

### Key Findings

- Fine-tuned MobileNetV2 achieved the highest test accuracy, F1-score, specificity, and ROC-AUC in this experiment.
- Frozen MobileNetV2 achieved the highest recall, at 0.992.
- Fine-tuning reduced false positives among Normal images from 92 to 68 compared with frozen MobileNetV2.
- Fine-tuned MobileNetV2 achieved a test ROC-AUC of 0.983, indicating strong discrimination on this benchmark test split.
- The fine-tuned model still misclassified 68 of 234 Normal test images as Pneumonia at the default threshold.

These findings describe the recorded experiment only. They do not establish clinical effectiveness, reliability across hospitals, or generalization to other patient populations.

---

## ⚙️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep-learning model development and inference |
| MobileNetV2 | Transfer-learning architecture |
| NumPy | Numerical operations and image arrays |
| Pillow | Image loading and resizing |
| Streamlit | Interactive web application |
| MedMNIST / PneumoniaMNIST | Benchmark dataset |
| Google Colab | Model training environment |
| GitHub | Source code and model artifact hosting |
| Streamlit Community Cloud | Web application deployment |

---

## 🖥️ Web Application

The Streamlit interface allows users to:

- Upload a chest X-ray image in PNG, JPG, or JPEG format.
- Select one of the three trained models.
- Run inference on the uploaded image.
- View the model's sigmoid output.

### Run Locally

#### Prerequisites

- Python 3.11
- Git
- The three trained model files

#### 1. Clone the repository

```bash
git clone https://github.com/Ritviks21/pneumoscan-ai.git
cd pneumoscan-ai
```

#### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Start the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, normally:

```text
http://localhost:8501
```

Keep the model files in the same directory as `app.py`, using the exact filenames below.

---

## 📁 Repository Structure

```text
pneumoscan-ai/
│
├── app.py
│   └── Streamlit application and inference pipeline
│
├── requirements.txt
│   └── Python dependencies
│
├── cnn_model.keras
│   └── Baseline CNN trained from scratch
│
├── mobilenet_frozen.keras
│   └── Frozen MobileNetV2 model
│
├── mobilenet_finetuned.keras
│   └── Fine-tuned MobileNetV2 model
│
└── README.md
    └── Project documentation
```

The repository contains the deployment application and trained model artifacts. Training notebooks, evaluation figures, and the complete experimental pipeline should be included separately if they are to be reproducible directly from this repository.

---

## 🔬 Evaluation Methodology

The models were evaluated on the same held-out test set.

The reported evaluation includes:

- **Accuracy:** overall proportion of correct predictions.
- **Precision:** proportion of predicted Pneumonia cases that are labelled Pneumonia in the dataset.
- **Recall / Sensitivity:** proportion of Pneumonia-labelled images correctly identified.
- **Specificity:** proportion of Normal-labelled images correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** discrimination across classification thresholds.
- **Confusion matrices:** counts of true positives, true negatives, false positives, and false negatives.
- **Bootstrap confidence intervals:** uncertainty estimates for selected metrics.
- **Threshold analysis:** comparison of the default threshold with thresholds selected using validation data.
- **Calibration analysis:** exploratory examination of model score calibration.

Pneumonia is treated as the positive class. A sigmoid score of 0.5 or greater is the default decision threshold in the reported comparison.

---

## ⚠️ Limitations and Responsible Use

This project has several important limitations:

1. **Benchmark scope:** evaluation uses one processed dataset rather than multiple hospitals or clinical settings.
2. **Resolution:** the experiment uses images configured at 224 × 224 pixels; this does not represent the full complexity of clinical radiography.
3. **Class imbalance:** the dataset contains substantially more Pneumonia-labelled than Normal images.
4. **False positives:** the fine-tuned model incorrectly flags a substantial number of Normal test images.
5. **Generalization:** external validation, prospective testing, and clinical reader comparisons were not conducted.
6. **Calibration:** the reported temperature-scaling analysis is exploratory, and its fitting split requires confirmation.
7. **Reproducibility:** results are based on recorded training runs and may vary with random seeds, software versions, and training conditions.
8. **Clinical safety:** model outputs are not validated diagnoses and should not influence patient treatment decisions.

The application is intended for educational demonstration and research comparison only. Do not upload identifiable patient information or use the system for real-world medical decisions.

---

## 🚀 Future Work

Potential extensions include:

- Reproducible training scripts and documented environment configuration.
- Evaluation across multiple random seeds.
- More detailed analysis of false positives and false negatives.
- Verification of patient-level independence across dataset partitions.
- Improved calibration using validation data only.
- External validation on independent datasets.
- Model explainability experiments, such as Grad-CAM, with appropriate caution.
- Improved application reporting, including inference latency and model metadata.

Any future clinical use would require substantially more validation, oversight, and appropriate regulatory review.

---

## 📚 References and Resources

1. Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell, 2018.
2. M. Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR, 2018.
3. MedMNIST, [Official website and dataset documentation](https://medmnist.com/).
4. TensorFlow, [Transfer learning and fine-tuning guide](https://www.tensorflow.org/guide/keras/transfer_learning).
5. TensorFlow, [MobileNetV2 documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2).
6. Streamlit, [Official documentation](https://docs.streamlit.io/).

For the full literature review, experimental methodology, statistical analysis, and additional references, consult the accompanying research report and paper.

---

## 👥 Authors

**Ritvik Raj Singh**  
Computer Science and Engineering (Data Science)  
Manipal University Jaipur

**Rishabh Srivastava**  
Computer Science and Engineering (Data Science)  
Manipal University Jaipur

**Course:** DSE3140 — Computer Vision  
**Project:** Summative Research Project  
**Date:** October 2026

---

## 📄 Project Status

**Status:** Research prototype and comparative model demonstration.

The application demonstrates inference using saved model artifacts. Full reproduction of the experiments requires the original training code, dataset setup, and evaluation workflow.
