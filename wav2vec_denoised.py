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
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import copy
from sklearn.metrics import f1_score, precision_score, recall_score

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Paths to datasets (Update these paths accordingly)
RAVDESS_PATH = '/Users/harshith/noCloud/mlsp_project/RAVDESS'  # Path to RAVDESS dataset
ESC50_PATH = '/Users/harshith/noCloud/mlsp_project/ESC - 50/audio/audio'  # Corrected path to ESC-50 .wav files

# Emotion labels in RAVDESS mapped to 4 classes
EMOTIONS = {
    '01': 'neutral',
    '02': 'neutral',   # 'calm' mapped to 'neutral'
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'sad',        # 'fearful' mapped to 'sad'
    '07': 'angry',      # 'disgust' mapped to 'angry'
    '08': 'neutral'     # 'surprised' mapped to 'neutral'
}

# Function to get emotion label from RAVDESS filename
def get_emotion_label(filename):
    """
    Extracts the emotion label from a RAVDESS filename.

    Args:
        filename (str): Filename of the audio file.

    Returns:
        str: Emotion label.
    """
    # Example filename: "03-01-01-01-01-01-01.wav"
    parts = filename.split('-')
    if len(parts) < 3:
        return 'unknown'
    emotion_code = parts[2]
    return EMOTIONS.get(emotion_code, 'unknown')

# Custom Dataset Class
class NoisyEmotionDataset(Dataset):
    """
    Dataset class for Emotion Recognition with noisy audio.

    Args:
        ravdess_files (list): List of paths to RAVDESS audio files.
        noise_files (list): List of paths to noise audio files.
        snr_db (int, optional): Desired Signal-to-Noise Ratio in decibels. Defaults to 5.
        transform (callable, optional): Optional transform to be applied on a sample.
        label_encoder (LabelEncoder, optional): Label encoder fitted on training labels.
    """
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
        if label == 'unknown':
            # Handle unknown labels if any
            label = 'neutral'  # Assigning to 'neutral' or handle appropriately
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
    """
    Loads file paths for RAVDESS and ESC-50 datasets.

    Returns:
        tuple: (list of RAVDESS file paths, list of ESC-50 noise file paths)
    """
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

# Custom collate_fn to pad waveforms
def collate_fn(batch):
    """
    Custom collate function to handle variable-length waveforms.

    Args:
        batch (list): List of samples.

    Returns:
        dict: Batched noisy waveforms, labels, and lengths.
    """
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
    # Check for mps
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load file paths
    ravdess_files, noise_files = load_file_paths()
    print(f"Number of RAVDESS files: {len(ravdess_files)}")
    print(f"Number of ESC-50 noise files: {len(noise_files)}")

    if len(noise_files) == 0:
        raise ValueError(f"No noise files found in {ESC50_PATH}. Please check the dataset path and ensure that it contains .wav files.\n")

    # Filter out 'unknown' labels
    ravdess_files = [f for f in ravdess_files if get_emotion_label(os.path.basename(f)) != 'unknown']

    # Shuffle and split into train, validation, and test sets
    random.shuffle(ravdess_files)
    split_train = int(0.8 * len(ravdess_files))
    split_val = int(0.9 * len(ravdess_files))
    train_files = ravdess_files[:split_train]
    val_files = ravdess_files[split_train:split_val]
    test_files = ravdess_files[split_val:]

    # Initialize and fit LabelEncoder using training labels
    train_labels = [get_emotion_label(os.path.basename(f)) for f in train_files]
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    num_classes = len(label_encoder.classes_)
    print(f"Emotion classes: {label_encoder.classes_}")

    # Check label distribution
    label_counts = Counter(train_labels)
    print("Training data label distribution:")
    for label in label_encoder.classes_:
        print(f"{label}: {label_counts[label]}")

    # Create datasets
    train_dataset = NoisyEmotionDataset(train_files, noise_files, snr_db=5, label_encoder=label_encoder)
    val_dataset = NoisyEmotionDataset(val_files, noise_files, snr_db=5, label_encoder=label_encoder)
    test_dataset = NoisyEmotionDataset(test_files, noise_files, snr_db=5, label_encoder=label_encoder)

    # Create data loaders with custom collate_fn
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Load pre-trained denoiser (MetricGAN+)
    print("Loading pre-trained MetricGAN+ denoiser...")
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
        run_opts={"device": device}
    )
    enhancer.eval()
    print("Denoiser loaded successfully.")

    # Load pre-trained Wav2Vec2 feature extractor and model for emotion recognition
    print("Loading Wav2Vec2 Feature Extractor and Model for Emotion Recognition...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er").to(device)
    print("Model loaded successfully.")

    # Verify model's num_labels
    print(f"Model num_labels: {model.config.num_labels}")
    if model.config.num_labels != num_classes:
        print(f"Adjusting model's num_labels from {model.config.num_labels} to {num_classes}")
        # Update the model's configuration
        config = model.config
        config.num_labels = num_classes
        # Reinitialize the classification head
        model.classifier = nn.Linear(config.hidden_size, num_classes).to(device)
        print("Model's classification head reinitialized.")

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-6)
    num_epochs = 10
    total_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Update loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop with validation and early stopping
    best_accuracy = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 3
    trigger_times = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            noisy_waveform = batch['noisy_waveform']
            labels = batch['label']
            lengths = batch['lengths']

            # Denoise the audio
            with torch.no_grad():
                lengths_norm = lengths.float() / lengths.max().float()
                denoised_waveform = enhancer.enhance_batch(noisy_waveform, lengths_norm)

            # Process with feature extractor
            inputs = feature_extractor(
                denoised_waveform.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=160000  # Adjust based on your data's maximum length
            )
            input_values = inputs.input_values.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_values, labels=labels)
            loss = outputs.loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validation phase
        model.eval()
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                noisy_waveform = batch['noisy_waveform']
                labels = batch['label']
                lengths = batch['lengths']

                # Denoise the audio
                lengths_norm = lengths.float() / lengths.max().float()
                denoised_waveform = enhancer.enhance_batch(noisy_waveform, lengths_norm)

                # Process with feature extractor
                inputs = feature_extractor(
                    denoised_waveform.numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=160000  # Same as training
                )
                input_values = inputs.input_values.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(input_values)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        # Calculate validation accuracy
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Check for improvement
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            trigger_times = 0
            print("Validation accuracy improved. Saving best model.")
        else:
            trigger_times += 1
            print(f"No improvement for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("Early stopping!")
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Evaluation on Test Set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            noisy_waveform = batch['noisy_waveform']
            labels = batch['label']
            lengths = batch['lengths']

            # Denoise the audio
            lengths_norm = lengths.float() / lengths.max().float()
            denoised_waveform = enhancer.enhance_batch(noisy_waveform, lengths_norm)

            # Process with feature extractor
            inputs = feature_extractor(
                denoised_waveform.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=160000  # Same as training
            )
            input_values = inputs.input_values.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_values)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy and print classification report
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

    # After predictions
    f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
    precision = precision_score(all_val_labels, all_val_preds, average='weighted')
    recall = recall_score(all_val_labels, all_val_preds, average='weighted')
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    #Saving the model weights
    # Save the trained model
    torch.save(model.state_dict(), 'emotion_recognition_model.pth')

    # Save the label encoder
    import joblib
    joblib.dump(label_encoder, 'label_encoder.pkl')

if __name__ == '__main__':
    main()