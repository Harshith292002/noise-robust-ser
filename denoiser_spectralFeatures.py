import numpy as np
import soundfile as sf
from scipy import signal
import scipy.ndimage
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SpeechDenoiser: 
    def __init__(self, noise_reduce_kwargs=None):
        """
        Initialize speech denoiser with optional parameters
        """
        self.default_params = {
            'n_grad_freq': 2,     # How many frequency channels to smooth over
            'n_grad_time': 4,     # How many time channels to smooth over
            'n_fft': 2048,        # Length of FFT window
            'win_length': 2048,   # Window length
            'hop_length': 512,    # Hop length
            'n_std_thresh': 1.5,  # Number of standard deviations for threshold
            'prop_decrease': 0.8,  # Scale factor for noise reduction
            'pad_mode': 'constant'
        }
        
        if noise_reduce_kwargs is not None:
            self.default_params.update(noise_reduce_kwargs)

    def _get_spec(self, y, params):
        """Get spectrogram of signal"""
        return librosa.stft(y, n_fft=params['n_fft'], 
                            hop_length=params['hop_length'],
                            win_length=params['win_length'])

    def reduce_noise(self, audio_clip, noise_clip=None, visual=False, sr=22050):
        """
        Reduce noise in audio clip using spectral gating
        
        Parameters:
        -----------
        audio_clip : array_like
            The audio signal to denoise
        noise_clip : array_like, optional
            A clip containing isolated noise (if available)
        visual : bool
            Whether to display spectrograms
        sr : int
            Sample rate of the audio clip
            
        Returns:
        --------
        array_like
            The denoised audio signal
        """
        params = self.default_params
        
        # Get spectrogram of signal
        sig_stft = self._get_spec(audio_clip, params)
        sig_stft_db = librosa.amplitude_to_db(np.abs(sig_stft))
        
        # Calculate statistics over time
        mean_freq_noise = np.mean(sig_stft_db, axis=1)
        std_freq_noise = np.std(sig_stft_db, axis=1)
        noise_thresh = mean_freq_noise + std_freq_noise * params['n_std_thresh']
        
        # Calculate mask
        db_thresh = np.repeat(noise_thresh[:, np.newaxis], sig_stft_db.shape[1], axis=1)
        sig_mask = sig_stft_db >= db_thresh
        
        # Smooth mask
        sig_mask = scipy.ndimage.gaussian_filter(sig_mask.astype(float), 
                                                 [params['n_grad_freq'], params['n_grad_time']])
        
        # Scale mask
        sig_mask = sig_mask * params['prop_decrease']
        sig_mask = np.clip(sig_mask, 0, 1)  # Ensure mask values are between 0 and 1
        
        # Apply mask
        sig_stft_denoised = sig_stft * sig_mask
        
        # Transform back to time domain
        audio_denoised = librosa.istft(sig_stft_denoised,
                                       hop_length=params['hop_length'],
                                       win_length=params['win_length'])
        
        if visual:
            self._plot_spectrograms(audio_clip, audio_denoised, params, sr)
        
        return audio_denoised

    def _plot_spectrograms(self, audio_clip, audio_denoised, params, sr):
        """Plot original and denoised spectrograms"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original spectrogram
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(audio_clip, 
                                                             n_fft=params['n_fft'], 
                                                             hop_length=params['hop_length'], 
                                                             win_length=params['win_length'])), 
                                         ref=np.max)
        img = librosa.display.specshow(D_orig, y_axis='linear', x_axis='time',
                                       sr=sr, hop_length=params['hop_length'],
                                       ax=axes[0])
        axes[0].set_title('Original Spectrogram')
        fig.colorbar(img, ax=axes[0], format="%+2.f dB")
        
        # Denoised spectrogram
        D_denoised = librosa.amplitude_to_db(np.abs(librosa.stft(audio_denoised, 
                                                                 n_fft=params['n_fft'], 
                                                                 hop_length=params['hop_length'], 
                                                                 win_length=params['win_length'])), 
                                             ref=np.max)
        img = librosa.display.specshow(D_denoised, y_axis='linear', x_axis='time',
                                       sr=sr, hop_length=params['hop_length'],
                                       ax=axes[1])
        axes[1].set_title('Denoised Spectrogram')
        fig.colorbar(img, ax=axes[1], format="%+2.f dB")
        
        plt.tight_layout()
        plt.show()

def process_audio_file(input_path, output_path, visual=False):
    """
    Process an audio file to reduce noise
    
    Parameters:
    -----------
    input_path : str
        Path to input audio file
    output_path : str
        Path to save denoised audio file
    visual : bool
        Whether to display spectrograms
    """
    # Load audio file
    audio, sr = librosa.load(input_path, sr=None)
    
    # Initialize and apply denoiser
    denoiser = SpeechDenoiser()
    audio_denoised = denoiser.reduce_noise(audio, visual=visual, sr=sr)
    
    # Normalize audio
    audio_denoised = librosa.util.normalize(audio_denoised)
    audio_denoised = audio_denoised.astype(np.float32)
    
    # Save denoised audio
    sf.write(output_path, audio_denoised, sr)
    
    return audio_denoised, sr

if __name__ == "__main__":
    import sys

    try:
        if len(sys.argv) != 3:
            print("Usage: python script.py input_audio.wav output_audio.wav")
            sys.exit(1)
            
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        
        # Process audio with visualization
        audio_denoised, sr = process_audio_file(input_path, output_path, visual=True)
        print(f"Processed audio saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1) 