import torch
import torchaudio
from speechbrain.inference import SpectralMaskEnhancement

# Load the pretrained MetricGAN+ model
enhancer = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank"
)

# Load and preprocess audio
waveform, sample_rate = torchaudio.load('/Users/harshith/noCloud/mlsp_project/noisy_data/noisy_2_0dB.wav')
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Convert to mono if necessary
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Compute lengths tensor
lengths = torch.tensor([1.0])

# Enhance the audio
with torch.no_grad():
    enhanced_waveform = enhancer.enhance_batch(waveform, lengths)

# Save the enhanced audio
torchaudio.save('enhanced_audio_gan.wav', enhanced_waveform.cpu(), sample_rate=16000)