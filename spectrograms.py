import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

def plot_spectrogram(audio, sr, title):
    """
    Plots the spectrogram of an audio signal.
    """
    plt.figure(figsize=(10, 6))
    spectrogram = librosa.stft(audio, n_fft=2048, hop_length=512)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    librosa.display.specshow(spectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

# Load clean and noisy audio files (replace with your file paths)
clean_audio_path = "/Users/harshith/noCloud/mlsp_project/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
noisy_audio_path = "/Users/harshith/noCloud/mlsp_project/noisy_ravdess/03-01-01-01-01-01-01_SNR0dB.wav"

# Load audio files
clean_audio, sr_clean = librosa.load(clean_audio_path, sr=None)
noisy_audio, sr_noisy = librosa.load(noisy_audio_path, sr=None)

# Plot spectrograms
plot_spectrogram(clean_audio, sr_clean, title="Spectrogram of Clean Audio")
plot_spectrogram(noisy_audio, sr_noisy, title="Spectrogram of Noisy Audio")