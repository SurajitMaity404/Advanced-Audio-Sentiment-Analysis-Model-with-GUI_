import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import gradio as gr
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
EMOTIONS = ["happy", "sad", "angry", "neutral", "calm", "fearful", "disgust", "surprised"]
SAMPLE_RATE = 22050
N_MELS = 128
MAX_LEN = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MODEL
# ============================================================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=len(EMOTIONS)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (N_MELS // 8) * (MAX_LEN // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================
def predict_audio(model, filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    melspec = librosa.power_to_db(melspec, ref=np.max)
    melspec = (melspec - melspec.mean()) / (melspec.std() + 1e-6)
    if melspec.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - melspec.shape[1]
        melspec = np.pad(melspec, ((0,0),(0,pad_width)))
    else:
        melspec = melspec[:, :MAX_LEN]

    tensor = torch.tensor(melspec).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        preds = model(tensor)
        probs = torch.softmax(preds, dim=1).cpu().numpy()[0]
        pred_idx = preds.argmax(1).item()
        emotion = EMOTIONS[pred_idx]
    return emotion, probs, melspec

def record_audio(duration=3, filename="live_test.wav"):
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    return filename

# ============================================================
# LOAD MODEL
# ============================================================
model = EmotionCNN().to(DEVICE)
model_path = os.path.join(os.getcwd(), "audio_emotion_model.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# ============================================================
# GRADIO FUNCTIONS
# ============================================================
def gradio_predict_upload(file_path):
    emotion, probs, spec = predict_audio(model, file_path)
    plt.figure(figsize=(6,3))
    plt.imshow(spec, origin='lower', aspect='auto', cmap='magma')
    plt.title(f"Predicted: {emotion}")
    plt.xlabel("Frames")
    plt.ylabel("Mel bins")
    plt.tight_layout()
    plt.savefig("temp_spec.png")
    plt.close()
    prob_dict = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
    return emotion, prob_dict, "temp_spec.png"

def gradio_predict_record(duration=3):
    filename = record_audio(duration)
    return gradio_predict_upload(filename)

# ============================================================
# GRADIO BLOCKS INTERFACE
# ============================================================
with gr.Blocks() as demo:
    gr.Markdown("# 🎵 Audio Emotion Recognition")
    gr.Markdown("Upload a `.wav` file or record live audio to predict emotion.")

    with gr.Row():
        with gr.Column():
            upload_audio = gr.Audio(label="Upload Audio (.wav)", type="filepath")
            record_button = gr.Button("Record Audio (3 sec)")
            record_duration = gr.Slider(1,5, step=1, value=3, label="Record Duration (seconds)")
        with gr.Column():
            pred_text = gr.Textbox(label="Predicted Emotion")
            prob_label = gr.Label(num_top_classes=8, label="Emotion Probabilities")
            spec_img = gr.Image(label="Mel Spectrogram")

    upload_audio.change(fn=gradio_predict_upload, inputs=upload_audio, outputs=[pred_text, prob_label, spec_img])
    record_button.click(fn=lambda _: gradio_predict_record(record_duration.value), inputs=None, outputs=[pred_text, prob_label, spec_img])

demo.launch(share=True)
