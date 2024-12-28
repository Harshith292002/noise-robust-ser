import torch
import torchaudio
from speechbrain.inference import SpectralMaskEnhancement
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import joblib
import sys
import os
import numpy as np
import torch.nn as nn

def process_audio_file(audio_file_path, model, enhancer, feature_extractor, label_encoder, device):
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_file_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Denoise the waveform
        lengths = torch.tensor([waveform.size(1)])
        lengths_norm = lengths.float() / lengths.max().float()
        with torch.no_grad():
            denoised_waveform = enhancer.enhance_batch(waveform, lengths_norm)

        # Process with feature extractor
        inputs = feature_extractor(
            denoised_waveform.squeeze(0).numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=160000  # Adjust if necessary
        )
        input_values = inputs.input_values.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_values)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

        # Map the prediction to the emotion label
        predicted_label = label_encoder.inverse_transform(preds.cpu().numpy())[0]
        print(f"{audio_file_path}: {predicted_label}")
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")

def main(directory_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the label encoder
    label_encoder = joblib.load('/Users/harshith/noCloud/mlsp_project/label_encoder.pkl')
    num_classes = len(label_encoder.classes_)

    # Load the trained model
    print("Loading the trained model...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")

    # Update the model's configuration if necessary
    if model.config.num_labels != num_classes:
        config = model.config
        config.num_labels = num_classes
        model.classifier = nn.Linear(config.hidden_size, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load('emotion_recognition_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Load the denoiser
    print("Loading the pre-trained denoiser...")
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
        run_opts={"device": device}
    )
    enhancer.eval()
    print("Denoiser loaded successfully.")

    # Load the feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")

    # Process each .wav file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            audio_file_path = os.path.join(directory_path, filename)
            process_audio_file(audio_file_path, model, enhancer, feature_extractor, label_encoder, device)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python inference.py path_to_audio_directory")
        sys.exit(1)
    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print(f"{directory_path} is not a valid directory.")
        sys.exit(1)
    main(directory_path)