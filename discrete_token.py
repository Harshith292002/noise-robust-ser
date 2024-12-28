import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import numpy as np
from speechbrain.inference import SpectralMaskEnhancement

class TokenAttention(nn.Module):
    def __init__(self, input_dim=1024, num_codebooks=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )
        self.num_codebooks = num_codebooks

    def forward(self, x):
        # x shape: (batch, num_codebooks, sequence_length, input_dim)
        scores = self.mlp(x)  # (batch, num_codebooks, sequence_length, 1)
        attention = torch.softmax(scores, dim=1)  # Along codebook dimension
        weighted = (x * attention).sum(dim=1)  # (batch, sequence_length, input_dim)
        return weighted

class ConformerBlock(nn.Module):
    def __init__(self, dim=1024, num_heads=8):
        super().__init__()
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(dim, num_heads)
        
        # Convolution module
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1)
        )
        
        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: (batch_size, seq_len, embedding_dim)
        
        # Self attention
        x_attn = x.transpose(0, 1)  # Shape: (seq_len, batch_size, embedding_dim)
        attended, _ = self.attention(x_attn, x_attn, x_attn)
        attended = attended.transpose(0, 1)  # Back to (batch_size, seq_len, embedding_dim)
        x = x + self.dropout(attended)
        x = self.norm1(x)
        
        # Convolution
        conv_input = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        conv_output = self.conv(conv_input).transpose(1, 2)  # Back to (batch_size, seq_len, embedding_dim)
        x = x + self.dropout(conv_output)
        x = self.norm2(x)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)
        
        return x


class AudioEnhancementSystem:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Initialize WavLM for tokenization
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)

        # Define projection layers with original selected_layers
        self.selected_layers = [1, 3, 7, 12, 18, 23]
        hidden_size = self.wavlm.config.hidden_size  # 1024 for large models
        self.projections = nn.ModuleList([
            nn.Linear(hidden_size, 1024).to(device) for _ in self.selected_layers
        ])

        # Initialize the pretrained speech enhancement model with the compatible model
        self.enhancer = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",  # Changed model
            savedir="pretrained_models/metricgan-plus-voicebank",
            run_opts={"device": device}
        )

        # Set models to evaluation mode
        self.wavlm.eval()
        self.enhancer.eval()

    def extract_tokens(self, audio_waveform):
        """Extract discrete tokens from audio using WavLM"""
        # Convert waveform tensor to numpy array and remove extra dimensions
        audio_waveform = audio_waveform.squeeze().numpy()  # Shape: (samples,)
        
        # Ensure the waveform is a 1D numpy array
        if audio_waveform.ndim > 1:
            audio_waveform = audio_waveform.reshape(-1)  # Flatten to 1D
        
        print(f"Audio waveform shape: {audio_waveform.shape}")  # Debugging
        
        # Pass the waveform to the feature extractor
        input_values = self.feature_extractor(
            audio_waveform,  # Numpy array
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.to(self.device)  # Should be shape: (batch_size, sequence_length)
        
        print(f"Input values shape before processing: {input_values.shape}")  # Debugging
        
        # If input_values has an extra dimension, squeeze it
        if input_values.ndim == 3:
            input_values = input_values.squeeze(1)
            print(f"Input values shape after squeezing: {input_values.shape}")  # Debugging
        
        # Get WavLM representations
        with torch.no_grad():
            outputs = self.wavlm(input_values, output_hidden_states=True)
        
        # Extract representations from different layers as described in the paper
        hidden_states = outputs.hidden_states  # List of tensors (batch_size, seq_len, hidden_size)
        
        tokens = []
        for idx, layer_idx in enumerate(self.selected_layers):
            layer_output = hidden_states[layer_idx]  # (batch_size, seq_len, hidden_size)
            # Apply projection
            projected = self.projections[idx](layer_output)  # (batch_size, seq_len, 1024)
            tokens.append(projected)
        
        tokens = torch.stack(tokens, dim=1)  # (batch_size, num_codebooks, seq_len, 1024)
        return tokens
    
    def enhance_audio(self, noisy_audio_path):
        """Enhance noisy audio file using a pretrained model"""
        # Load and resample audio
        waveform, sample_rate = torchaudio.load(noisy_audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # If the waveform has more than one channel, convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Compute lengths tensor
        lengths = torch.tensor([1.0], dtype=torch.float32).to(self.device)

        # Enhance audio using the pretrained model
        with torch.no_grad():
            enhanced_waveform = self.enhancer.enhance_batch(waveform.to(self.device), lengths=lengths)

        return enhanced_waveform

# Example usage
if __name__ == "__main__":
    # Initialize system
    enhancer = AudioEnhancementSystem()
    
    # Enhance audio file
    enhanced_waveform = enhancer.enhance_audio("/Users/harshith/noCloud/mlsp_project/noisy_data_renamed/noisy_37445_5dB.wav")
    
    # Save the enhanced waveform to a file
    torchaudio.save("enhanced_audio.wav", enhanced_waveform.cpu(), sample_rate=16000)
    print("Enhanced audio saved as 'enhanced_audio.wav'")