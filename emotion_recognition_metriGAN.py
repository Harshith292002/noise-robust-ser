import os
import torch
import torchaudio
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from speechbrain.inference import SpectralMaskEnhancement  # Updated import as per deprecation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import utility functions from bc_utils.py
# import bc_utils as U

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Paths to datasets (Update these paths accordingly)
RAVDESS_PATH = '/Users/harshith/noCloud/mlsp_project/RAVDESS'  # Path to RAVDESS dataset
ESC50_PATH = '/Users/harshith/noCloud/mlsp_project/ESC - 50/audio/audio'  # Corrected path to ESC-50 .wav files

# Emotion labels in RAVDESS
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to get emotion label from RAVDESS filename
def get_emotion_label(filename):
    # Example filename: "03-01-01-01-01-01-01.wav"
    parts = filename.split('-')
    if len(parts) < 3:
        return 'unknown'
    emotion_code = parts[2]
    return EMOTIONS.get(emotion_code, 'unknown')

# Custom Dataset Class
class NoisyEmotionDataset(Dataset):
    def __init__(self, ravdess_files, noise_files, snr_db=5, transform=None, label_encoder=None):
        self.ravdess_files = ravdess_files
        self.noise_files = noise_files
        self.snr_db = snr_db
        self.transform = transform

        if label_encoder is None:
            raise ValueError("A label_encoder must be provided")
        else:
            self.label_encoder = label_encoder

    def __len__(self):
        return len(self.ravdess_files)

    def __getitem__(self, idx):
        # Load RAVDESS audio
        speech_path = self.ravdess_files[idx]
        speech_waveform, speech_sr = torchaudio.load(speech_path)
        speech_waveform = speech_waveform.mean(dim=0, keepdim=True)  # Convert to mono

        # Resample if necessary
        if speech_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=speech_sr, new_freq=16000)
            speech_waveform = resampler(speech_waveform)

        # Randomly select a noise file
        noise_path = random.choice(self.noise_files)
        noise_waveform, noise_sr = torchaudio.load(noise_path)
        noise_waveform = noise_waveform.mean(dim=0, keepdim=True)  # Convert to mono

        # Resample noise to match speech
        if noise_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=noise_sr, new_freq=16000)
            noise_waveform = resampler(noise_waveform)

        # Trim or pad noise to match speech length
        if noise_waveform.size(1) < speech_waveform.size(1):
            # Repeat noise to match length
            repeats = int(np.ceil(speech_waveform.size(1) / noise_waveform.size(1)))
            noise_waveform = noise_waveform.repeat(1, repeats)
        noise_waveform = noise_waveform[:, :speech_waveform.size(1)]

        # Adjust noise level based on desired SNR
        speech_power = speech_waveform.norm(p=2)
        noise_power = noise_waveform.norm(p=2)
        snr = 10 ** (self.snr_db / 20)
        if noise_power == 0:
            scale = 0
        else:
            scale = speech_power / (snr * noise_power)
        noisy_waveform = speech_waveform + scale * noise_waveform

        # Get label
        label = get_emotion_label(os.path.basename(speech_path))
        label = self.label_encoder.transform([label])[0]

        sample = {
            'noisy_waveform': noisy_waveform.squeeze(0),
            'label': label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

# Load file paths
def load_file_paths():
    # Load RAVDESS file paths
    ravdess_files = []
    for root, dirs, files in os.walk(RAVDESS_PATH):
        for file in files:
            if file.endswith('.wav'):
                ravdess_files.append(os.path.join(root, file))

    # Load ESC-50 noise file paths
    noise_files = []
    for root, dirs, files in os.walk(ESC50_PATH):
        for file in files:
            if file.endswith('.wav'):
                noise_files.append(os.path.join(root, file))

    print(f"Loaded {len(ravdess_files)} RAVDESS files.")
    print(f"Loaded {len(noise_files)} ESC-50 noise files from {ESC50_PATH}.")

    return ravdess_files, noise_files

# Advanced Emotion Classifier Model
class AdvancedEmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AdvancedEmotionClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, time_steps, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, time_steps)
        x = self.cnn(x)        # (batch_size, 128, time_steps / 2)
        x = x.transpose(1, 2)  # (batch_size, time_steps / 2, 128)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)    # (batch_size, time_steps / 2, 256)
        x = x[:, -1, :]        # Take the last time step (batch_size, 256)
        x = self.fc(x)         # (batch_size, num_classes)
        return x

# Custom collate_fn to pad waveforms
def collate_fn(batch):
    # batch is a list of samples
    noisy_waveforms = [sample['noisy_waveform'] for sample in batch]
    labels = [sample['label'] for sample in batch]

    # Find the maximum length in the batch
    max_length = max([waveform.size(0) for waveform in noisy_waveforms])

    # Initialize padded tensors
    padded_noisy = torch.zeros(len(noisy_waveforms), max_length)
    lengths = torch.zeros(len(noisy_waveforms), dtype=torch.long)

    # Pad waveforms and compute lengths
    for i in range(len(noisy_waveforms)):
        length = noisy_waveforms[i].size(0)
        padded_noisy[i, :length] = noisy_waveforms[i]
        lengths[i] = length

    labels = torch.tensor(labels, dtype=torch.long)

    return {
        'noisy_waveform': padded_noisy,
        'label': labels,
        'lengths': lengths
    }

# Main function
def main():
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load file paths
    ravdess_files, noise_files = load_file_paths()
    print(f"Number of RAVDESS files: {len(ravdess_files)}")
    print(f"Number of ESC-50 noise files: {len(noise_files)}")

    if len(noise_files) == 0:
        raise ValueError(f"No noise files found in {ESC50_PATH}. Please check the dataset path and ensure that it contains .wav files.")

    # Filter out 'unknown' labels
    ravdess_files = [f for f in ravdess_files if get_emotion_label(os.path.basename(f)) != 'unknown']

    # Limit data for demonstration purposes (optional)
    max_samples = 1000  # Adjust as needed based on your computational resources
    ravdess_files = ravdess_files[:max_samples]

    # Split into train and test sets
    random.shuffle(ravdess_files)
    split_idx = int(0.8 * len(ravdess_files))
    train_files = ravdess_files[:split_idx]
    test_files = ravdess_files[split_idx:]

    # Initialize and fit LabelEncoder using training labels
    train_labels = [get_emotion_label(os.path.basename(f)) for f in train_files]
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    num_classes = len(label_encoder.classes_)
    print(f"Emotion classes: {label_encoder.classes_}")

    # Create datasets
    train_dataset = NoisyEmotionDataset(train_files, noise_files, snr_db=5, label_encoder=label_encoder)
    test_dataset = NoisyEmotionDataset(test_files, noise_files, snr_db=5, label_encoder=label_encoder)

    # Create data loaders with custom collate_fn
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Load pre-trained denoiser (MetricGAN+)
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
        run_opts={"device": device}
    )
    enhancer.eval()

    # Feature extractor (MFCC)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={
            "n_mels": 40,
            "n_fft": 400,
            "hop_length": 160,
            "mel_scale": "htk"
        },
        log_mels=True
    ).to(device)

    # Initialize advanced emotion classifier
    input_dim = 40  # Number of MFCC features
    model = AdvancedEmotionClassifier(input_dim=input_dim, num_classes=num_classes)
    model = model.to(device)
    model.train()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            noisy_waveform = batch['noisy_waveform'].to(device)  # Shape: (batch_size, max_length)
            labels = batch['label'].to(device)  # Shape: (batch_size,)
            lengths = batch['lengths'].to(device)  # Shape: (batch_size,)

            # Denoise the audio
            with torch.no_grad():
                lengths_norm = lengths.float() / lengths.max().float()  # Normalize lengths between 0 and 1
                denoised_waveform = enhancer.enhance_batch(noisy_waveform, lengths_norm)

            # Extract features without averaging over time
            features = mfcc_transform(denoised_waveform)  # (batch_size, n_mfcc, time_steps)
            features = features.transpose(1, 2)  # (batch_size, time_steps, n_mfcc)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            noisy_waveform = batch['noisy_waveform'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['lengths'].to(device)

            # Denoise the audio
            lengths_norm = lengths.float() / lengths.max().float()
            denoised_waveform = enhancer.enhance_batch(noisy_waveform, lengths_norm)

            # Extract features
            features = mfcc_transform(denoised_waveform)
            features = features.transpose(1, 2)  # (batch_size, time_steps, n_mfcc)

            # Forward pass
            outputs = model(features)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy and print classification report
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

if __name__ == '__main__':
    main()