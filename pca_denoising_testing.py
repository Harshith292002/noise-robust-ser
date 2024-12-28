import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa
from sklearn.decomposition import PCA

class AudioDenoiser:
    def __init__(self, n_fft=2048, hop_length=512, n_components=0.95):
        """
        Initialize the denoiser with STFT and PCA parameters
        
        Parameters:
        -----------
        n_fft : int
            Number of FFT components
        hop_length : int
            Number of samples between successive frames
        n_components : float
            PCA variance ratio to maintain (between 0 and 1)
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_components = n_components
        
    def _apply_stft(self, audio_signal):
        """Apply Short-Time Fourier Transform"""
        return librosa.stft(audio_signal, 
                          n_fft=self.n_fft, 
                          hop_length=self.hop_length,
                          center=True,
                          pad_mode='reflect')
    
    def _apply_istft(self, stft_signal, original_length):
        """
        Apply Inverse Short-Time Fourier Transform
        
        Parameters:
        -----------
        stft_signal : numpy.ndarray
            STFT signal to inverse transform
        original_length : int
            Length of the original signal to match
        """
        return librosa.istft(stft_signal, 
                           hop_length=self.hop_length,
                           length=original_length)
    
    def _create_bandpass_filter(self, stft_signal, low_freq=500, high_freq=4000, sample_rate=22050):
        """Create and apply bandpass filter in frequency domain"""
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=self.n_fft)
        
        # Create frequency mask
        mask = np.ones_like(frequencies, dtype=bool)
        mask[frequencies < low_freq] = False
        mask[frequencies > high_freq] = False
        
        # Apply mask to STFT
        filtered_stft = stft_signal.copy()
        filtered_stft[~mask] = 0
        
        return filtered_stft
    
    def _apply_pca_denoising(self, stft_signal):
        """Apply PCA-based denoising to the STFT signal"""
        # Prepare data for PCA
        X = np.abs(stft_signal.T)  # Use magnitude spectrum
        
        # Fit and transform PCA
        pca = PCA(n_components=self.n_components)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Reconstruct complex STFT
        phase = np.angle(stft_signal)
        reconstructed_stft = X_reconstructed.T * np.exp(1j * phase)
        
        return reconstructed_stft
    
    def denoise(self, audio_signal, sample_rate=22050):
        """
        Apply the complete denoising process to an audio signal
        
        Parameters:
        -----------
        audio_signal : numpy.ndarray
            Input audio signal
        sample_rate : int
            Sampling rate of the audio signal
            
        Returns:
        --------
        numpy.ndarray
            Denoised audio signal
        """
        # Store original length
        original_length = len(audio_signal)
        
        # Normalize audio
        audio_signal = audio_signal.astype(float) / np.max(np.abs(audio_signal))
        
        # Apply STFT
        stft_signal = self._apply_stft(audio_signal)
        
        # Apply bandpass filter
        filtered_stft = self._create_bandpass_filter(stft_signal, 
                                                   sample_rate=sample_rate)
        
        # Apply PCA denoising
        denoised_stft = self._apply_pca_denoising(filtered_stft)
        
        # Inverse STFT with original length
        denoised_signal = self._apply_istft(denoised_stft, original_length)
        
        # Ensure the output has the same length as input
        if len(denoised_signal) > original_length:
            denoised_signal = denoised_signal[:original_length]
        elif len(denoised_signal) < original_length:
            denoised_signal = np.pad(denoised_signal, 
                                   (0, original_length - len(denoised_signal)))
        
        return denoised_signal
    
    def compute_snr(self, original_signal, denoised_signal):
        """
        Compute Signal-to-Noise Ratio (SNR)
        
        Parameters:
        -----------
        original_signal : numpy.ndarray
            Original audio signal
        denoised_signal : numpy.ndarray
            Processed audio signal
            
        Returns:
        --------
        float
            SNR value in dB
        """
        # Ensure signals have the same length
        min_length = min(len(original_signal), len(denoised_signal))
        original_signal = original_signal[:min_length]
        denoised_signal = denoised_signal[:min_length]
        
        # Normalize signals
        original_signal = original_signal.astype(float)
        denoised_signal = denoised_signal.astype(float)
        
        noise = original_signal - denoised_signal
        signal_power = np.sum(denoised_signal ** 2)
        noise_power = np.sum(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)

def test_denoiser(input_file, output_file):
    """
    Test the denoising implementation on an audio file
    
    Parameters:
    -----------
    input_file : str
        Path to input audio file
    output_file : str
        Path to save denoised audio file
    """
    # Load audio file
    sample_rate, audio_signal = wavfile.read(input_file)
    
    # Convert to mono if stereo
    if len(audio_signal.shape) > 1:
        audio_signal = np.mean(audio_signal, axis=1)
    
    # Initialize denoiser
    denoiser = AudioDenoiser()
    
    # Apply denoising
    denoised_signal = denoiser.denoise(audio_signal, sample_rate=sample_rate)
    
    # Ensure signals are the same length before computing SNR
    min_length = min(len(audio_signal), len(denoised_signal))
    audio_signal = audio_signal[:min_length]
    denoised_signal = denoised_signal[:min_length]
    
    # Compute SNR
    snr_original = denoiser.compute_snr(audio_signal, denoised_signal)
    print(f"Original Signal SNR: {snr_original:.2f} dB")
    
    # Save denoised audio
    wavfile.write(output_file, sample_rate, denoised_signal.astype(np.float32))

# Example usage
if __name__ == "__main__":
    input_file = "/Users/harshith/noCloud/mlsp_project/noisy_data/noisy_37394_10dB.wav"
    output_file = "/Users/harshith/noCloud/mlsp_project/denoised_audio.wav"
    
    test_denoiser(input_file, output_file)