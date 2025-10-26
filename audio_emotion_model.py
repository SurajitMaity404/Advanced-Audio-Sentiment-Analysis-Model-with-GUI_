import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
from scipy.io.wavfile import write

# ============================================================
# CONFIG
# ============================================================
EMOTIONS = ["happy", "sad", "angry", "neutral", "calm", "fearful", "disgust", "surprised"]
SAMPLE_RATE = 22050
N_MELS = 128
MAX_LEN = 400  # truncate or pad to this many frames
BATCH_SIZE = 8
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# DATASET CLASS
# ============================================================
class AudioEmotionDataset(Dataset):
    def __init__(self, base_dir):
        self.files = []
        self.labels = []
        self.encoder = LabelEncoder()
        self.encoder.fit(EMOTIONS)

        for emotion in EMOTIONS:
            folder = os.path.join(base_dir, emotion)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.endswith(".wav"):
                        self.files.append(os.path.join(folder, f))
                        self.labels.append(emotion)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        y, sr = librosa.load(path, sr=SAMPLE_RATE)

        # Compute mel-spectrogram
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        melspec = librosa.power_to_db(melspec, ref=np.max)

        # Normalize
        melspec = (melspec - melspec.mean()) / (melspec.std() + 1e-6)

        # Pad or crop
        if melspec.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - melspec.shape[1]
            melspec = np.pad(melspec, ((0, 0), (0, pad_width)))
        else:
            melspec = melspec[:, :MAX_LEN]

        melspec = torch.tensor(melspec).unsqueeze(0).float()
        label_idx = torch.tensor(self.encoder.transform([label])[0]).long()
        return melspec, label_idx

# ============================================================
# MODEL: CNN for Emotion Detection
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
# TRAINING LOOP
# ============================================================
def train_model(model, loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for mel, label in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            mel, label = mel.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out = model(mel)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")
    print("✅ Training complete.")

# ============================================================
# PREDICTION FROM FILE
# ============================================================
def predict_audio(model, filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    melspec = librosa.power_to_db(melspec, ref=np.max)
    melspec = (melspec - melspec.mean()) / (melspec.std() + 1e-6)

    if melspec.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - melspec.shape[1]
        melspec = np.pad(melspec, ((0, 0), (0, pad_width)))
    else:
        melspec = melspec[:, :MAX_LEN]

    melspec = torch.tensor(melspec).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        preds = model(melspec)
        pred_idx = preds.argmax(1).item()
        emotion = EMOTIONS[pred_idx]
    return emotion

# ============================================================
# LIVE RECORDING (optional)
# ============================================================
def record_and_predict(model, duration=3):
    print("🎤 Recording...")
    fs = SAMPLE_RATE
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write("live_test.wav", fs, audio)
    print("✅ Saved as live_test.wav")

    emotion = predict_audio(model, "live_test.wav")
    print(f"🎧 Predicted Emotion: {emotion}")

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    base_dir = "audio_dataset"  # Folder with subfolders for each emotion
    dataset = AudioEmotionDataset(base_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EmotionCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    train_model(model, loader, criterion, optimizer, EPOCHS)

    # Save model
    torch.save(model.state_dict(), "audio_emotion_model.pth")
    print("✅ Model saved as audio_emotion_model.pth")

    # Test with live mic (optional)
    record_and_predict(model)
