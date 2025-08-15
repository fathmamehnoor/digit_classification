import numpy as np
import librosa
import soundfile as sf
import torch
import pickle
import os
from typing import Optional, Tuple, List
import threading
import queue
import time

class AudioProcessor2D:
    """Real-time audio processing for 2D CNN digit classification"""
    
    def __init__(self, sample_rate=8000, n_fft=512, hop_length=256, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Audio buffer for real-time processing
        self.audio_buffer = np.array([])
        self.buffer_size = sample_rate * 2  # 2 seconds buffer
        self.min_audio_length = sample_rate * 0.5  # Minimum 0.5 seconds
        
        # Processing parameters
        self.silence_threshold = 0.01
        self.energy_threshold = 0.005
        
    def extract_spectrogram_from_audio(self, audio: np.ndarray, target_length: int = 128) -> Optional[np.ndarray]:
        """Extract mel-spectrogram from audio array"""
        try:
            # Ensure correct sample rate
            if len(audio) == 0:
                return None
                
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Pad or truncate to fixed length
            mel_spec_db = self.pad_or_truncate_spectrogram(mel_spec_db, target_length)
            
            return mel_spec_db
            
        except Exception as e:
            print(f"Error extracting spectrogram: {e}")
            return None
    
    def pad_or_truncate_spectrogram(self, spec: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate spectrogram to fixed length"""
        if spec.shape[1] > target_length:
            # Truncate
            spec = spec[:, :target_length]
        elif spec.shape[1] < target_length:
            # Pad with zeros
            padding = np.zeros((spec.shape[0], target_length - spec.shape[1]))
            spec = np.hstack([spec, padding])
        
        return spec
    
    def detect_speech_segment(self, audio: np.ndarray) -> Tuple[Optional[np.ndarray], int, int]:
        """Detect speech segment in audio buffer"""
        if len(audio) < self.min_audio_length:
            return None, 0, 0
        
        # Calculate energy
        energy = np.mean(audio ** 2)
        
        # Find speech segments using energy threshold
        speech_frames = np.where(np.abs(audio) > self.energy_threshold)[0]
        
        if len(speech_frames) == 0:
            return None, 0, 0
        
        # Find start and end of speech
        start_idx = speech_frames[0]
        end_idx = speech_frames[-1]
        
        # Ensure minimum length
        if end_idx - start_idx < self.min_audio_length:
            return None, 0, 0
        
        # Extract speech segment
        speech_segment = audio[start_idx:end_idx + 1]
        
        return speech_segment, start_idx, end_idx
    
    def add_audio_to_buffer(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer"""
        self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
        
        # Keep only the last buffer_size samples
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]
    
    def get_speech_from_buffer(self) -> Optional[np.ndarray]:
        """Get speech segment from buffer and clear it"""
        speech_segment, start_idx, end_idx = self.detect_speech_segment(self.audio_buffer)
        
        if speech_segment is not None:
            # Clear the buffer
            self.audio_buffer = np.array([])
            return speech_segment
        
        return None
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        if len(audio) == 0:
            return audio
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Simple noise reduction using spectral subtraction"""
        try:
            # Compute spectrogram
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first few frames (assuming silence at start)
            noise_estimate = np.mean(magnitude[:, :5], axis=1, keepdims=True)
            
            # Spectral subtraction
            cleaned_magnitude = np.maximum(magnitude - 0.5 * noise_estimate, 0.1 * magnitude)
            
            # Reconstruct signal
            cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
            cleaned_audio = librosa.istft(cleaned_stft, hop_length=self.hop_length)
            
            return cleaned_audio
            
        except Exception as e:
            print(f"Error in noise reduction: {e}")
            return audio

class AudioClassifier2D:
    """2D CNN Audio classifier with pre-trained model"""
    
    def __init__(self, model_path: str, preprocessor_path: str, scaler_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.audio_processor = AudioProcessor2D()
        
        # Load model and preprocessor
        self.model = self.load_model(model_path)
        self.preprocessor = self.load_preprocessor(preprocessor_path)
        self.scaler = self.load_scaler(scaler_path)
        
        # Class labels
        self.class_labels = list(range(10))  # 0-9
        
    def load_model(self, model_path: str):
        """Load trained 2D CNN model"""
        try:
            from models.model import AudioDigitClassifier2D
            
            # Create model instance
            model = AudioDigitClassifier2D(num_classes=10)
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            print(f"2D CNN Model loaded from {model_path}")
            return model
            
        except Exception as e:
            print(f"Error loading 2D CNN model: {e}")
            return None
    
    def load_preprocessor(self, preprocessor_path: str):
        """Load preprocessor"""
        try:
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            print(f"2D Preprocessor loaded from {preprocessor_path}")
            return preprocessor
        except Exception as e:
            print(f"Error loading 2D preprocessor: {e}")
            return None
    
    def load_scaler(self, scaler_path: str):
        """Load scaler"""
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"2D Scaler loaded from {scaler_path}")
            return scaler
        except Exception as e:
            print(f"Error loading 2D scaler: {e}")
            return None
    
    def predict(self, audio: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Predict digit from audio using 2D CNN"""
        if self.model is None:
            return -1, 0.0, np.zeros(10)
        
        try:
            # Preprocess audio to get spectrogram
            spectrogram = self.audio_processor.extract_spectrogram_from_audio(audio)
            if spectrogram is None:
                return -1, 0.0, np.zeros(10)
            
            # Normalize features using the saved scaler
            spectrogram_flat = spectrogram.flatten().reshape(1, -1)
            spectrogram_norm = self.scaler.transform(spectrogram_flat)
            spectrogram_norm = spectrogram_norm.reshape(spectrogram.shape)
            
            # Convert to tensor and add batch and channel dimensions
            # Shape: (1, 1, height, width) for 2D CNN
            spectrogram_tensor = torch.FloatTensor(spectrogram_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(spectrogram_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            return predicted_class, confidence, probabilities[0].cpu().numpy()
            
        except Exception as e:
            print(f"Error in 2D CNN prediction: {e}")
            return -1, 0.0, np.zeros(10)
    
    def predict_from_file(self, audio_path: str) -> Tuple[int, float, np.ndarray]:
        """Predict digit from audio file using 2D CNN"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.audio_processor.sample_rate)
            
            # Normalize
            audio = self.audio_processor.normalize_audio(audio)
            
            # Predict
            return self.predict(audio)
            
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return -1, 0.0, np.zeros(10)
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index"""
        if 0 <= class_idx <= 9:
            return str(class_idx)
        return "Unknown"
