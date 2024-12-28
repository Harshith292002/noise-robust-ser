import os
import torch
import torchaudio
import numpy as np
import joblib
import sys

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved RandomForestClassifier
model_path = 'saved_models/random_forest_emotion_model.joblib'
clf = joblib.load(model_path)
print(f"Loaded trained model from {model_path}")

# Load the saved LabelEncoder
label_encoder_path = 'saved_models/label_encoder.joblib'
label_encoder = joblib.load(label_encoder_path)
print(f"Loaded label encoder from {label_encoder_path}")

# Load the MetricGAN+ enhancer
try:
    import torchaudio
    version_parts = torchaudio.__version__.split('.')
    major_minor_version = float(f"{version_parts[0]}.{version_parts[1]}")

    if major_minor_version >= 0.10:
        bundle = torchaudio.pipelines.METRICGAN_U
        enhancer = bundle.get_model().to(device)
        enhancer.eval()
    else:
        raise ImportError("Torchaudio version is below 0.10")
except (AttributeError, ImportError):
    print("Using SpeechBrain's MetricGAN+ implementation.")
    from speechbrain.pretrained import SpectralMaskEnhancement
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
        run_opts={"device": device}
    )
    enhancer.eval()

# Load the Wav2Vec 2.0 model
wav2vec_bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec2_model = wav2vec_bundle.get_model().to(device)
wav2vec2_model.eval()

def predict_emotion(audio_file):
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

    # Reshape features for prediction
    features_mean = features_mean.reshape(1, -1)

    # Predict emotion
    predicted_label_index = clf.predict(features_mean)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

    return predicted_label

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py path_to_audio_file.wav")
        sys.exit(1)

    audio_file_path = sys.argv[1]

    if not os.path.isfile(audio_file_path):
        print(f"Audio file '{audio_file_path}' does not exist.")
        sys.exit(1)

    print(f"Processing audio file: {audio_file_path}")
    predicted_emotion = predict_emotion(audio_file_path)
    print(f"Predicted Emotion: {predicted_emotion}")