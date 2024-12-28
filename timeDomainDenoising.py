import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio

class TimeDomainModule(nn.Module):
    def __init__(self, in_channels=1):
        super(TimeDomainModule, self).__init__()
        
        # Downsampling blocks
        self.encoder = nn.ModuleList([
            self._make_layer(in_channels, 16),
            self._make_layer(16, 32),
            self._make_layer(32, 64),
            self._make_layer(64, 128),
            self._make_layer(128, 256),
            self._make_layer(256, 512),
        ])
        
        # Upsampling blocks
        self.decoder = nn.ModuleList([
            self._make_layer(512, 256, upsample=True),
            self._make_layer(512, 128, upsample=True),
            self._make_layer(256, 64, upsample=True),
            self._make_layer(128, 32, upsample=True),
            self._make_layer(64, 16, upsample=True),
            self._make_layer(32, 1, upsample=True),
        ])

    def _make_layer(self, in_channels, out_channels, kernel_size=5, stride=2, upsample=False):
        if upsample:
            return nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=2, output_padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )

    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoding
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)
                
        # Decoding with skip connections
        encoder_outputs = encoder_outputs[::-1][1:]  # Reverse and remove last output
        for i, decoder_layer in enumerate(self.decoder[:-1]):
            x = decoder_layer(x)
            x = torch.cat([x, encoder_outputs[i]], dim=1)
            
        x = self.decoder[-1](x)
        return x

class ComplexTFModule(nn.Module):
    def __init__(self, n_fft=400, hop_length=100):
        super(ComplexTFModule, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Encoder with downsampling
        self.conv_encoder = nn.ModuleList([
            self._make_layer(2, 16),
            self._make_layer(16, 32),
            self._make_layer(32, 64),
            self._make_layer(64, 128),
            self._make_layer(128, 256),
        ])
        
        # LSTM layers
        self.lstm = nn.LSTM(256 * (n_fft // (2 ** 5)), 256, num_layers=2, batch_first=True)
        
        # Decoder with upsampling
        self.conv_decoder = nn.ModuleList([
            self._make_layer(256, 128, upsample=True),
            self._make_layer(256, 64, upsample=True),
            self._make_layer(128, 32, upsample=True),
            self._make_layer(64, 16, upsample=True),
            self._make_layer(32, 2, upsample=True),
        ])

    def _make_layer(self, in_channels, out_channels, kernel_size=(3,3), stride=(2,2), upsample=False):
        if upsample:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

    def stft(self, x):
        return torch.stft(x, self.n_fft, self.hop_length, window=torch.hann_window(self.n_fft).to(x.device),
                          return_complex=False)

    def istft(self, x):
        x_complex = torch.complex(x[..., 0], x[..., 1])
        return torch.istft(x_complex, self.n_fft, self.hop_length,
                           window=torch.hann_window(self.n_fft).to(x.device))

    def forward(self, x):
        # STFT
        spec = self.stft(x.squeeze(1))  # [B, F, T, 2]
        spec = spec.permute(0, 3, 1, 2)  # [B, 2, F, T]
        
        # Encoding
        encoder_outputs = []
        for encoder_layer in self.conv_encoder:
            spec = encoder_layer(spec)
            encoder_outputs.append(spec)
        
        # Reshape for LSTM
        batch, channels, freq, time = spec.size()
        spec = spec.view(batch, channels * freq, time).permute(0, 2, 1)  # [B, T, C*F]
        
        # LSTM processing
        spec, _ = self.lstm(spec)
        
        # Reshape back
        spec = spec.permute(0, 2, 1).view(batch, channels, freq, time)
        
        # Decoding with skip connections
        encoder_outputs = encoder_outputs[::-1][1:]  # Reverse and remove last output
        for i, decoder_layer in enumerate(self.conv_decoder[:-1]):
            spec = decoder_layer(spec)
            spec = torch.cat([spec, encoder_outputs[i]], dim=1)
            
        spec = self.conv_decoder[-1](spec)
        
        # ISTFT
        spec = spec.permute(0, 2, 3, 1)  # [B, F, T, 2]
        output = self.istft(spec)
        
        return output.unsqueeze(1)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class ParallelModel(nn.Module):
    def __init__(self):
        super(ParallelModel, self).__init__()
        self.time_module = TimeDomainModule()
        self.tf_module1 = ComplexTFModule()
        self.tf_module2 = ComplexTFModule()
        self.attention = ChannelAttention(3)

    def forward(self, x):
        # Original noisy input
        noisy = x

        # Parallel processing
        time_output = self.time_module(x)
        tf_output = self.tf_module1(x)

        # Ensure outputs are the same size
        min_length = min(noisy.size(-1), time_output.size(-1), tf_output.size(-1))
        noisy = noisy[..., :min_length]
        time_output = time_output[..., :min_length]
        tf_output = tf_output[..., :min_length]

        # Concatenate features
        concat_features = torch.cat([noisy, time_output, tf_output], dim=1)

        # Apply attention
        attended_features = self.attention(concat_features)

        # Final TF module
        output = self.tf_module2(attended_features)

        return output

def enhance_audio(audio_path, model_path=None):
    """
    Enhance an audio file using the trained model
    
    Args:
        audio_path: Path to the noisy audio file
        model_path: Path to the trained model weights (optional)
    
    Returns:
        Enhanced audio as numpy array
    """
    import torchaudio
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Ensure mono audio
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Initialize model
    model = ParallelModel()
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Process audio
    with torch.no_grad():
        # Convert to appropriate format
        waveform = waveform.unsqueeze(0)  # Add batch dimension
        
        # Process through model
        enhanced = model(waveform)
        
        # Convert back to numpy
        enhanced = enhanced.squeeze().cpu().numpy()
    
    return enhanced, sample_rate

# Example usage
if __name__ == "__main__":
    # Example usage of the enhancement function
    audio_path = "/Users/harshith/noCloud/mlsp_project/noisy_ravdess/03-01-01-01-01-01-01_SNR5dB.wav"
    model_path = ""  # Path to trained weights if available
    
    enhanced_audio, sr = enhance_audio(audio_path, model_path)
    
    # Save enhanced audio
    import soundfile as sf
    sf.write("enhanced_speech.wav", enhanced_audio, sr)
