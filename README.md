# Dog Skin Disease Classification using Vision Transformers (ViT)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Timm](https://img.shields.io/badge/Timm-Library-blue?style=for-the-badge)](https://github.com/huggingface/pytorch-image-models)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dual%20T4%20GPU-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

An automated veterinary diagnostic tool powered by **Vision Transformers (ViT)** to classify six types of canine skin conditions with **91% accuracy**. This project utilizes multi-GPU acceleration and the `timm` library to provide a robust deep-learning solution for animal health.

---

## Performance Summary
* **Overall Accuracy:** 91%
* **Key Strength:** Exceptional identification of infectious/parasitic diseases.
    * **Demodicosis:** 100% Recall (Zero missed cases).
    * **Ringworm:** 99% Recall.
* **Infrastructure:** Multi-GPU (DataParallel) training on Kaggle.



---

## Model Architecture
The project utilizes the `vit_tiny_patch16_224` model. Unlike traditional CNNs, the Vision Transformer processes images as a sequence of patches using self-attention mechanisms.

- **Backbone:** `vit_tiny_patch16_224`
- **Input Size:** 224x224x3
- **Optimizer:** `AdamW` (Learning Rate: 1e-4, Weight Decay: 0.01)
- **Loss Function:** `CrossEntropyLoss`



---

## Supported Classes
The model is trained to recognize the following 6 categories:
1. **Dermatitis** (Precision: 98%)
2. **Fungal Infections**
3. **Healthy**
4. **Hypersensitivity**
5. **Demodicosis** (Parasitic)
6. **Ringworm** (Fungal)

---

##  How to Run

### 1. Prerequisites
Install the required dependencies:
```bash
pip install torch torchvision timm matplotlib seaborn scikit-learn
