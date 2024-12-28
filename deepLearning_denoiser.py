import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, Model
import soundfile as sf

def build_crced_model():
    input_shape = (129, 8)
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Permute((2, 1))(inputs)  # Swap time and frequency axes
    x = layers.LSTM(256, return_sequences=False)(x)
    x = layers.Dense(129)(x)
    outputs = layers.Reshape((129, 1))(x)
    model = tf.keras.Model(inputs, outputs)
    return model

class AudioPreprocessor:
    def __init__(self, sample_rate=8000, n_fft=256, hop_length=64, seq_length=8):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.seq_length = seq_length

    def load_and_preprocess(self, audio_path):
        """Load and preprocess audio file"""
        # Load audio and resample
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Optionally remove silence (disabled for now)
        audio = self._remove_silence(audio)

        # Ensure minimum length for STFT
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))

        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                            window='hamming')

        # Get magnitude spectrogram and remove symmetric half
        mag_spec = np.abs(stft)[:129, :]

        # Create sequences of consecutive frames
        sequences = self._create_sequences(mag_spec)

        return sequences, stft.shape

    def _remove_silence(self, audio, threshold=0.01):
        """Optionally remove silent frames based on energy threshold"""
        # For simplicity, we'll skip silence removal
        return audio

    def _create_sequences(self, mag_spec):
        """Create sequences of consecutive STFT vectors"""
        sequences = []
        for i in range(self.seq_length, mag_spec.shape[1]):
            seq = mag_spec[:, i - self.seq_length:i]
            sequences.append(seq)
        return np.array(sequences)

    def mix_signals(self, clean_seq, noise_seq, snr_db=0):
        """Mix clean and noise sequences with specified SNR"""
        # Get minimum length
        min_len = min(len(clean_seq), len(noise_seq))

        # Truncate sequences to same length
        clean_seq = clean_seq[:min_len]
        noise_seq = noise_seq[:min_len]

        # Calculate scaling factor for desired SNR
        clean_power = np.mean(clean_seq ** 2)
        noise_power = np.mean(noise_seq ** 2)
        scaling_factor = np.sqrt(clean_power / (noise_power * (10 ** (snr_db / 10))))

        # Mix signals
        noisy_seq = clean_seq + scaling_factor * noise_seq

        return noisy_seq, clean_seq

class AudioDenoiser:
    def __init__(self, model_weights_path=None):
        """Initialize the audio denoiser with trained weights"""
        self.preprocessor = AudioPreprocessor()
        self.model = build_crced_model()
        if model_weights_path:
            self.model.load_weights(model_weights_path)

    def denoise_file(self, input_path, output_path):
        """
        Denoise an audio file and save the result

        Parameters:
        -----------
        input_path : str
            Path to the noisy audio file
        output_path : str
            Path to save the denoised audio file
        """
        # Load and preprocess audio
        sequences, stft_shape = self.preprocessor.load_and_preprocess(input_path)

        if len(sequences) == 0:
            print("Error: No valid sequences extracted from audio file")
            return

        # Get predictions
        print("Denoising audio...")
        predictions = self.model.predict(sequences)

        # Get original audio and phase
        audio, sr = librosa.load(input_path, sr=self.preprocessor.sample_rate)
        stft_original = librosa.stft(audio, n_fft=self.preprocessor.n_fft,
                                     hop_length=self.preprocessor.hop_length)
        phase = np.angle(stft_original)

        # Initialize full magnitude spectrogram
        num_frames = stft_original.shape[1]
        mag_spec = np.zeros((129, num_frames))

        # Fill in the predictions
        seq_length = self.preprocessor.seq_length
        for i, pred in enumerate(predictions):
            frame_idx = i + seq_length
            if frame_idx < num_frames:
                mag_spec[:, frame_idx] = pred[:, 0]

        # Handle initial frames by copying from the noisy magnitude spectrum
        initial_mag_spec = np.abs(stft_original)[:129, :seq_length]
        mag_spec[:, :seq_length] = initial_mag_spec

        # Ensure shapes match
        min_width = min(mag_spec.shape[1], phase.shape[1])
        mag_spec = mag_spec[:, :min_width]
        phase = phase[:129, :min_width]

        print(f"Magnitude shape: {mag_spec.shape}, Phase shape: {phase.shape}")

        # Reconstruct complex STFT
        complex_spec = mag_spec * np.exp(1j * phase)

        # Inverse STFT
        denoised = librosa.istft(complex_spec,
                                 hop_length=self.preprocessor.hop_length,
                                 length=len(audio))

        # Normalize output
        denoised = denoised / np.max(np.abs(denoised))

        # Save output
        print(f"Saving denoised audio to {output_path}")
        sf.write(output_path, denoised, self.preprocessor.sample_rate)

        # Calculate and print SNR improvement
        original_snr = self._calculate_snr(audio, audio - denoised)
        print(f"Original SNR: {original_snr:.2f} dB")
        return denoised

    def _calculate_snr(self, signal, noise):
        """Calculate Signal-to-Noise Ratio in dB"""
        signal_power = np.sum(signal ** 2)
        noise_power = np.sum(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)

def train_model(clean_files, noise_files, model_save_path, epochs=100, batch_size=32):
    """Train the denoising model"""
    preprocessor = AudioPreprocessor()
    model = build_crced_model()
    model.compile(optimizer='adam', loss='mse')

    def generate_batch(clean_files, noise_files, batch_size):
        while True:
            batch_inputs = []
            batch_targets = []

            while len(batch_inputs) < batch_size:
                # Randomly select files
                clean_file = np.random.choice(clean_files)
                noise_file = np.random.choice(noise_files)

                try:
                    # Load and preprocess
                    clean_seq, _ = preprocessor.load_and_preprocess(clean_file)
                    noise_seq, _ = preprocessor.load_and_preprocess(noise_file)

                    if len(clean_seq) == 0 or len(noise_seq) == 0:
                        continue

                    # Mix signals
                    noisy_seq, clean_seq = preprocessor.mix_signals(clean_seq, noise_seq)

                    # Add to batch
                    for i in range(len(noisy_seq)):
                        batch_inputs.append(noisy_seq[i])
                        batch_targets.append(clean_seq[i, :, -1:])

                        if len(batch_inputs) >= batch_size:
                            break

                except Exception as e:
                    print(f"Error processing files: {e}")
                    continue

            batch_inputs = np.array(batch_inputs)
            batch_targets = np.array(batch_targets)

            yield batch_inputs, batch_targets

    # Train the model
    steps_per_epoch = 1000  # Set a reasonable number for steps per epoch
    model.fit(generate_batch(clean_files, noise_files, batch_size),
              steps_per_epoch=steps_per_epoch,
              epochs=epochs)

    # Save model weights
    if not model_save_path.endswith('.weights.h5'):
        model_save_path = model_save_path.replace('.h5', '.weights.h5')
    model.save_weights(model_save_path)

# Example usage
if __name__ == "__main__":
    import glob

    # Training phase
    clean_files = glob.glob("/Users/harshith/noCloud/mlsp_project/RAVDESS/**/*.wav", recursive=True)
    noise_files = glob.glob("/Users/harshith/noCloud/mlsp_project/noisy_ravdess/**/*.wav", recursive=True)

    if not clean_files or not noise_files:
        print("No audio files found. Please check the paths.")
    else:
        print(f"Found {len(clean_files)} clean files and {len(noise_files)} noise files")

        # Train model
        print("Training model...")
        weights_path = "denoiser.weights.h5"
        train_model(clean_files, noise_files, weights_path)

        # Denoise test file
        print("\nDenoising test file...")
        denoiser = AudioDenoiser(weights_path)

        # You can replace these paths with your actual audio files
        input_file = "/Users/harshith/noCloud/mlsp_project/noisy_ravdess/03-01-01-01-01-01-01_SNR5dB.wav"
        output_file = "/Users/harshith/noCloud/mlsp_project/denoised_wav.wav"

        try:
            denoiser.denoise_file(input_file, output_file)
            print(f"Denoised audio saved to {output_file}")
        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found.")
        except Exception as e:
            print(f"Error during denoising: {e}")