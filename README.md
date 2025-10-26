# Advanced-Audio-Sentiment-Analysis-Model-with-GUI_
# 🎧 Advanced Audio Emotion Recognition Model

This project is a **deep learning-based Audio Sentiment Analysis system** built with **PyTorch**, capable of detecting **8 emotions** from speech audio.  
It can train on custom datasets, analyze uploaded audio files, and even perform **real-time emotion detection** through a microphone (locally via PyAudio).

---

## 🔥 Features

- 🎙️ Detects **8 emotions**: `Happy`, `Sad`, `Angry`, `Neutral`, `Calm`, `Fearful`, `Disgust`, `Surprised`  
- ⚡ Fast **CNN architecture** using Mel-Spectrograms  
- 📂 Train on your **own dataset** or uploaded audio files  
- 🔊 **Real-time voice emotion detection** via microphone  
- 🧠 Built with **PyTorch**, **Librosa**, and **Torchaudio**  
- 🚀 Fully runnable in **Google Colab** or **VS Code (local machine)**  

---

## 🧩 Model Architecture

The model uses a **Convolutional Neural Network (CNN)** that learns emotional features from mel-spectrograms (frequency vs time representations of audio).  
It captures tone, pitch, and rhythm patterns that distinguish human emotions.

---

## 🛠️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/your-username/audio-emotion-recognition.git
cd audio-emotion-recognition
