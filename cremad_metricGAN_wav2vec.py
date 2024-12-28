import os
import glob
import torch
import torchaudio
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Set the data path
data_path = '/Users/harshith/noCloud/mlsp_project/CREAM-D/'

# Collect all .wav files
audio_files = glob.glob(os.path.join(data_path, '*.wav'))

# Define emotion mapping from file codes to labels
emotion_mapping = {
    'ANG': 'Angry',
    'DIS': 'Disgust',
    'FEA': 'Fear',
    'HAP': 'Happy',
    'NEU': 'Neutral',
    'SAD': 'Sad'
}

# Function to extract emotion code from filename
def extract_emotion_label(filename):
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 3:
        emotion_code = parts[2]
        return emotion_code
    else:
        return None

# Collect data with emotion labels
data = []
for audio_file in audio_files:
    emotion_code = extract_emotion_label(audio_file)
    emotion_label = emotion_mapping.get(emotion_code)
    if emotion_label:
        data.append((audio_file, emotion_label))

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Option 1: Load the MetricGAN+ enhancer from torchaudio
try:
    import torchaudio
    # Extract major and minor version numbers and convert to a comparable float
    version_parts = torchaudio.__version__.split('.')
    major_minor_version = float(f"{version_parts[0]}.{version_parts[1]}")

    if major_minor_version >= 0.10:
        # Load the MetricGAN+ enhancer
        bundle = torchaudio.pipelines.METRICGAN_U
        enhancer = bundle.get_model().to(device)
        enhancer.eval()
    else:
        raise ImportError("Torchaudio version is below 0.10")
except (AttributeError, ImportError):
    print("MetricGAN+ model not found in torchaudio. Using SpeechBrain's implementation.")
    import speechbrain as sb
    from speechbrain.inference import SpectralMaskEnhancement
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank"
    ).to(device)

# Load the wav2vec 2.0 model
wav2vec_bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec2_model = wav2vec_bundle.get_model().to(device)
wav2vec2_model.eval()

# Process each audio file
features = []
labels = []

for idx, (audio_file, emotion_label) in enumerate(data):
    print(f"Processing {idx+1}/{len(data)}: {audio_file}")
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_file)

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Move to device
    waveform = waveform.to(device)

    # Denoise using MetricGAN+
    with torch.no_grad():
        try:
            # For torchaudio's MetricGAN+
            enhanced_waveform = enhancer(waveform)
        except:
            # For SpeechBrain's MetricGAN+
            lengths = torch.tensor([1.0], device=device)
            enhanced_waveform = enhancer.enhance_batch(waveform, lengths=lengths)

    # Extract features using wav2vec 2.0
    with torch.no_grad():
        features_vector, _ = wav2vec2_model.extract_features(enhanced_waveform)
        # Use the last layer's features
        last_hidden_state = features_vector[-1]
        # Average over the time dimension to get a fixed-size feature vector
        features_mean = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    features.append(features_mean)
    labels.append(emotion_label)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Prepare data for training
X = np.array(features)
y = labels_encoded

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the trained classifier
import joblib

# Create a directory to save the models if it doesn't exist
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Save the RandomForestClassifier
model_path = 'saved_models/random_forest_emotion_model.joblib'
joblib.dump(clf, model_path)
print(f"Trained model saved at {model_path}")

# Save the LabelEncoder
label_encoder_path = 'saved_models/label_encoder.joblib'
joblib.dump(label_encoder, label_encoder_path)
print(f"Label encoder saved at {label_encoder_path}")