# Noise-Robust Speech Emotion Recognition (SER) 🎙️🔊

**An advanced machine learning project to enhance emotion recognition systems by mitigating the impact of environmental noise using state-of-the-art denoising and multimodal techniques.**

---

## 📖 Overview

Emotion recognition systems are significantly impacted by noise, which degrades their accuracy. This project explores robust solutions to improve speech emotion recognition under noisy conditions by leveraging advanced denoising techniques like **MetricGAN+** and **Wav2Vec2**, and multimodal fusion using **HuBERT** and **BERT**.

---

## 🚀 Objectives

- Develop a robust emotion recognition system capable of performing under noisy environments.
- Implement denoising techniques to recover the accuracy of emotion classification models.
- Explore multimodal approaches to integrate text and audio for enhanced recognition performance.

---

## 📂 Repository Structure

- **`MLSP_project_RAVDESS.ipynb`**: Jupyter Notebook for the main project implementation using the RAVDESS dataset.
- **`cremad_metricGAN_wav2vec.py`**: Implementation of MetricGAN+ and Wav2Vec2 denoising techniques.
- **`deepLearning_denoiser.py`**: Denoising pipeline for audio inputs.
- **`emotion_recognition_metricGAN.py`**: Emotion recognition model integrating denoising techniques.
- **`interface.py`**: User interface for real-time emotion recognition.
- **`main.py`**: Main script for running the emotion recognition system.
- **`pca_denoising_testing.py`**: PCA-based denoising implementation and evaluation.
- **`spectrograms.py`**: Script for generating and analyzing spectrograms.
- **`timeDomainDenoising.py`**: Time-domain denoising approach.
- **`wav2vec_denoised.py`**: Integration of Wav2Vec2 with denoised inputs.
- **`wav2vec_withoutDenoising.py`**: Baseline Wav2Vec2 model without denoising.

---

## 🗂️ Datasets

### **RAVDESS Dataset**

- **Description**: Contains 1440 audio recordings by 24 actors across 8 emotions.
- **Emotions**: Happy, Sad, Fearful, Disgust, Angry, Neutral, Surprised, Calm.

### **ESC-50 Dataset**

- **Description**: Environmental sounds dataset used for noise simulation.
- **Classes**: 50 environmental sounds such as rain, fire, and animals.

### **MELD Dataset**

- **Description**: Multimodal dataset with audio, text, and visual components.
- **Emotions**: Anger, Disgust, Sadness, Joy, Neutral, Surprise, Fear.

---

## 🛠️ Methodology

### Denoising Techniques

1. **MetricGAN+**: GAN-based speech enhancement trained on VoiceBank Dataset.
2. **PCA-Based Denoising**: Signal reconstruction using PCA and STFT.
3. **Spectral Gating**: Threshold-based filtering in the frequency domain.

### Multimodal Fusion

1. **HuBERT + BERT (Concatenation)**: Combines text and audio features directly.
2. **HuBERT + BERT (Attention)**: Employs attention mechanism to better fuse text and audio representations.
3. **WaveLM + BERT**: Advanced multimodal approach using WaveLM for audio.

---

## 📊 Key Findings

1. **Denoising Impact**:

   - Noise significantly impacts accuracy (18% accuracy for noisy data).
   - MetricGAN+ improved accuracy to 51% at 5dB SNR.

2. **Multimodal Fusion**:

   - Multimodal models outperformed single-input models.
   - Attention mechanisms provided superior feature fusion over concatenation.

3. **Challenges**:
   - Non-stationary noise handling.
   - Balancing model complexity and real-time performance.

---

## 📈 Results

- **Baseline Accuracy**: 74.7% on clean RAVDESS dataset.
- **Noisy Data Accuracy**: Dropped to 18.8%.
- **Denoised Data Accuracy**: Recovered to 51.4% using MetricGAN+.

---

## 🔮 Future Work

- Develop an end-to-end pipeline for real-time emotion recognition.
- Train models with diverse and realistic noise scenarios.
- Explore advanced multimodal fusion techniques for higher accuracy.

---

## 🛠️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Harshith292002/noise-robust-ser.git
   cd noise-robust-ser
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

---

## 📚 References

1. [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477)
2. [MetricGAN+](https://arxiv.org/abs/1905.04874)
3. [HuBERT](https://arxiv.org/abs/2106.07447)
4. [BERT](https://arxiv.org/abs/1810.04805)

---

**License**: MIT
