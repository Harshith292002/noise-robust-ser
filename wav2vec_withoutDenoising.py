import os
import torch
import torchaudio
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    get_linear_schedule_with_warmup
)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('mps' if torch.mps.is_available() else 'cpu')

# Paths to datasets (Update these paths accordingly)
RAVDESS_PATH = '/Users/harshith/noCloud/mlsp_project/RAVDESS'  # Path to RAVDESS dataset

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
class EmotionDataset(Dataset):
    """
    Dataset class for Emotion Recognition.

    Args:
        ravdess_files (list): List of paths to RAVDESS audio files.
        transform (callable, optional): Optional transform to be applied on a sample.
        label_encoder (LabelEncoder, optional): Label encoder fitted on training labels.
    """
    def __init__(self, ravdess_files, transform=None, label_encoder=None):
        self.ravdess_files = ravdess_files
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

        # Get label
        label = get_emotion_label(os.path.basename(speech_path))
        label = self.label_encoder.transform([label])[0]

        sample = {
            'waveform': speech_waveform.squeeze(0),
            'label': label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

# Custom collate_fn to handle list of waveforms
def collate_fn(batch):
    """
    Custom collate function to handle list of waveforms.

    Args:
        batch (list): List of samples.

    Returns:
        dict: Batched waveforms and labels.
    """
    waveforms = [sample['waveform'] for sample in batch]
    labels = [sample['label'] for sample in batch]

    labels = torch.tensor(labels, dtype=torch.long)

    return {
        'waveform': waveforms,  # list of tensors
        'label': labels
    }

# Load file paths
def load_file_paths():
    """
    Loads file paths for RAVDESS dataset.

    Returns:
        list: List of RAVDESS file paths
    """
    # Load RAVDESS file paths
    ravdess_files = []
    for root, dirs, files in os.walk(RAVDESS_PATH):
        for file in files:
            if file.endswith('.wav'):
                ravdess_files.append(os.path.join(root, file))

    print(f"Loaded {len(ravdess_files)} RAVDESS files.")

    return ravdess_files, None  # Return None for compatibility

# Main function
def main():
    # Check for mps
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load file paths
    ravdess_files, _ = load_file_paths()
    print(f"Number of RAVDESS files: {len(ravdess_files)}")

    # Filter out 'unknown' labels
    ravdess_files = [f for f in ravdess_files if get_emotion_label(os.path.basename(f)) != 'unknown']

    # Shuffle and split into train and test sets
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
    train_dataset = EmotionDataset(train_files, label_encoder=label_encoder)
    test_dataset = EmotionDataset(test_files, label_encoder=label_encoder)

    # Create data loaders with custom collate_fn
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Load pre-trained Wav2Vec2 model and processor
    print("Loading pre-trained Wav2Vec2 model and processor...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base-960h",
        num_labels=num_classes,
        problem_type="single_label_classification"
    ).to(device)
    print("Wav2Vec2 model and processor loaded successfully.")

    # Freeze feature extractor (updated method)
    model.freeze_feature_encoder()

    # Loss and optimizer (use torch.optim.AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Scheduler
    num_epochs = 5  # Adjust number of epochs as needed
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            waveforms = batch['waveform']  # list of tensors
            labels = batch['label'].to(device)

            # Convert list of tensors to list of numpy arrays
            waveforms = [w.numpy() for w in waveforms]

            # Process inputs with attention_mask
            inputs = processor(
                waveforms,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # Forward pass
            outputs = model(
                input_values,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            waveforms = batch['waveform']  # list of tensors
            labels = batch['label'].to(device)

            # Convert list of tensors to list of numpy arrays
            waveforms = [w.numpy() for w in waveforms]

            # Process inputs
            inputs = processor(
                waveforms,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # Forward pass
            outputs = model(
                input_values,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy and print classification report
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

if __name__ == '__main__':
    main()