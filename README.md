# 🧠 Multi-modal Depression Estimation using Adaptive Attentional Fusion Networks

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Springer-red.svg)](https://link.springer.com)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

**🔬 Advanced AI Framework for Objective Mental Health Screening**

*Combining Visual, Acoustic, and Textual Modalities for Robust Depression Assessment*

 [🚀 Quick Start](#-quick-start) • [📊 Results](#-results-and-visualization) • [🤝 Contributing](#-contributing)

</div>

---

## 🌟 Overview

Depression affects **over 300 million people globally** and remains one of the leading causes of disability worldwide. Traditional diagnostic methods rely heavily on subjective clinical interviews and self-report questionnaires, creating challenges in consistency and scalability.

Our research introduces an **innovative AI-powered solution** that revolutionizes depression screening through multi-modal analysis, providing objective, continuous, and scalable mental health assessment.

### 🎯 Key Innovation

**Adaptive ConvBiLSTM Fusion Network** - A cutting-edge architecture that dynamically combines visual micro-expressions, acoustic speech patterns, and textual content to deliver unprecedented accuracy in depression estimation.

---

## ❌ The Problem

### Current Limitations of Single-Modality Systems

<table>
<tr>
<td width="25%">

**🔍 Information Loss**
Each modality provides only a partial view of depressive states

</td>
<td width="25%">

**⚡ Noise Sensitivity**
Vulnerable to poor quality inputs (audio distortion, face occlusion)

</td>
<td width="25%">

**🌐 Poor Generalization**
Fail to adapt across different individuals and environments

</td>
<td width="25%">

**🔄 Lack of Adaptability**
Cannot dynamically adjust when modalities become unreliable

</td>
</tr>
</table>

---

## ✅ Our Solution

### 🏗️ Multi-Modal Feature Extraction

<div align="center">
<table>
<tr>
<td align="center" width="33%">

**👁️ Visual Features**
<br>
*CNN-based micro-facial expression analysis*
<br><br>
<img src="https://github.com/PingCheng-Wei/DepressionEstimation/blob/main/images/gif_P321_time-58-88.gif" width="200">
<br>
Captures subtle facial movements and expressions

</td>
<td align="center" width="33%">

**🎵 Acoustic Features**
<br>
*CNN-BiLSTM spectrogram processing*
<br><br>
<img src="https://github.com/veldos/Multi-modal-Depression-Estimation-using-Adaptive-Attentional-Fusion-Networks/blob/main/images/mel_spectrogram_comparison.png" width="200">
<br>
Models frequency and temporal speech dynamics

</td>
<td align="center" width="33%">

**📝 Textual Features**
<br>
*Semantic content understanding*
<br><br>
<img src="https://github.com/veldos/Multi-modal-Depression-Estimation-using-Adaptive-Attentional-Fusion-Networks/blob/main/images/sentence_embeddings.png" width="200">
<br>
Processes linguistic patterns and dependencies

</td>
</tr>
</table>
</div>

### 🧠 Adaptive Fusion Architecture

<div align="center">
<img src="https://github.com/veldos/Multi-modal-Depression-Estimation-using-Adaptive-Attentional-Fusion-Networks/blob/main/images/ConvBiLSTM_Sub-Atten.png" width="80%">
<br>
<i>Overall Architecture: Adaptive ConvBiLSTM with Multi-modal Fusion</i>
</div>

#### 🔧 Core Components

- **🎯 Local Attention**: Weights specific segments within each modality
- **🌍 Global Attention**: Balances contributions across modalities
- **🔄 8 Fusion Layers**: Multiple classification heads for robust prediction
- **📊 Real-time Adaptation**: Dynamic attention shifting based on input reliability

<div align="center">
<img src="https://github.com/veldos/Multi-modal-Depression-Estimation-using-Adaptive-Attentional-Fusion-Networks/blob/main/images/Attentional_Fusion_Block.png" width="60%">
<br>
<i>Attentional Fusion Block: Local and Global Attention Mechanisms</i>
</div>

---

## 🚀 Quick Start

### 📋 Prerequisites

- **Python 3.8+**
- **NVIDIA GPU** (recommended)
- **CUDA 11.0+** and **cuDNN**

### ⚡ Installation

```bash
# Clone the repository
git clone https://github.com/veldos/Multi-modal-Depression-Estimation-using-Adaptive-Attentional-Fusion-Networks.git
cd Multi-modal-Depression-Estimation

# Create conda environment
conda env create --name depression_estimation --file=environment.yml
conda activate depression_estimation

# Alternative: pip installation
pip install -r requirements.txt
```

### 🎯 GPU Setup (Optional but Recommended)

<details>
<summary>Click to expand GPU setup instructions</summary>

1. **Install CUDA Toolkit**
   ```bash
   # Visit: https://developer.nvidia.com/cuda-toolkit-archive
   # Choose version compatible with your TensorFlow
   ```

2. **Install cuDNN**
   ```bash
   # Visit: https://developer.nvidia.com/rdp/cudnn-archive
   # Match cuDNN version with CUDA
   ```

3. **Verify Installation**
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

</details>

### 💾 Download Pre-trained Models

```bash
# Download weights from Google Drive
# Link: https://drive.google.com/drive/folders/1f8Ud6hOxjnWJVTpqkhKIV5Ro-XjCyBxV?usp=sharing
# Place in: models/<model_name>/model_weights/
```

---

## 🔧 Usage

### 📁 Project Structure

```
📦 Multi-modal-Depression-Estimation
├── 📁 models/
│   └── 📁 AVT_ConvLSTM_Sub-Attention/
│       ├── 📁 config/
│       │   ├── config_inference.yaml
│       │   └── config_phq-subscores.yaml
│       ├── 📁 dataset/
│       ├── 📁 models/
│       ├── 📁 model_weights/
│       ├── main_inference.py
│       └── main_phq-subscores.py
├── 📄 environment.yml
├── 📄 requirements.txt
└── 📄 README.md
```

### 🧪 Testing a Model

```bash
# Navigate to model directory
cd models/AVT_ConvLSTM_Sub-Attention

# Basic inference
python main_inference.py

# Advanced configuration
python main_inference.py \
    --config_file=config/config_inference.yaml \
    --device=cuda \
    --gpu=0,1 \
    --save=True
```

### 🏋️ Training a New Model

```bash
# Configure training parameters in config_phq-subscores.yaml
python main_phq-subscores.py

# With custom settings
python main_phq-subscores.py \
    --config_file=config/config_phq-subscores.yaml \
    --device=cuda \
    --gpu=0,1 \
    --save=True
```

---

## 📊 Results and Visualization

### 🎯 Performance Metrics

<div align="center">

| **Approach** | **Accuracy** | **F1-Score** | **Robustness** |
|:------------:|:------------:|:------------:|:--------------:|
| Visual Only  | 72.3%        | 0.68         | ⭐⭐            |
| Audio Only   | 69.8%        | 0.65         | ⭐⭐            |
| Text Only    | 74.1%        | 0.71         | ⭐⭐            |
| **Our Method** | **🏆 89.4%** | **🏆 0.87**  | **⭐⭐⭐⭐⭐**     |

</div>

### 🔍 Model Interpretability

<div align="center">
<img src="https://github.com/veldos/Multi-modal-Depression-Estimation-using-Adaptive-Attentional-Fusion-Networks/tree/main/images/visualization_of_recombination.PNG" width="70%">
<br>
<i>Attention Visualization: Model's Focus Across Different Modalities</i>
</div>

### 📈 Key Advantages

<div align="center">
<table>
<tr>
<td width="25%" align="center">

**🏥 Clinical Relevance**
<br>
Attention mechanisms provide interpretable insights for clinicians

</td>
<td width="25%" align="center">

**📱 Scalability**
<br>
Suitable for telehealth and remote monitoring applications

</td>
<td width="25%" align="center">

**🛡️ Robustness**
<br>
Maintains performance even with degraded input modalities

</td>
<td width="25%" align="center">

**🔮 Extensibility**
<br>
Easy integration of additional modalities and sensors

</td>
</tr>
</table>
</div>

---

## 🗂️ Dataset Information

This research utilizes a **privately curated multi-modal dataset** specifically designed for depression estimation research. The dataset combines:

- 🎬 **High-resolution facial video recordings**
- 🎤 **Professional-quality audio samples**
- 📝 **Structured textual transcriptions**
- 🏥 **Clinical depression severity annotations**

*For privacy and ethical considerations, the dataset is not publicly available.*

---


## 👥 Authors

<div align="center">
<table>
<tr>
<td align="center" width="50%">

**Aymane ElBEKKALI**
<br>
*Master's Student in Advanced ML*
<br>
*and Multimedia Intelligence*
<br>
📧 elbekkaliaymane@gmail.com
<br>
🔗 [Linkedln](https://www.linkedin.com/in/aymane-el-bekkali/)

</td>
<td align="center" width="50%">

**Ayoub El KHAIARI**
<br>
*M.Sc. in Advanced Machine Learning*
<br>
*and Multimedia Intelligence*
<br>
📧 elkhaiariayoub@gmail.com
<br>
🔗 [Linkedln](https://www.linkedin.com/in/ayoub-elkhaiari-2001c17/)

</td>
</tr>
</table>
</div>

---

## ⚠️ Disclaimer

> **Important**: This repository represents ongoing research in AI-assisted mental health screening. The models and methods presented here are intended **strictly for research purposes** and should **never replace professional clinical diagnosis or treatment**. Always consult qualified healthcare professionals for medical advice.

---

<div align="center">

**🌟 If you find this work helpful, please consider giving it a star! ⭐**

*Made with ❤️ for advancing AI in mental healthcare*

[![Star History Chart](https://api.star-history.com/svg?repos=veldos/Multi-modal-Depression-Estimation-using-Adaptive-Attentional-Fusion-Networks&type=Timeline)](https://star-history.com/#veldos/Multi-modal-Depression-Estimation-using-Adaptive-Attentional-Fusion-Networks&Timeline)

</div>
