import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm

class AudioPreprocessor2D:
    """Handles audio preprocessing and spectrogram extraction for 2D CNN"""
    
    def __init__(self, sample_rate=8000, n_fft=512, hop_length=256, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.scaler = StandardScaler()
        
    def extract_spectrogram(self, audio_path):
        """Extract mel-spectrogram from audio file"""
        try:
            # Load audio and resample if needed
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
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
            
            return mel_spec_db
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def extract_spectrogram_from_audio(self, audio_array, target_length=128):
        """Extract mel-spectrogram from audio array"""
        try:
            # Ensure correct sample rate
            if len(audio_array) == 0:
                return None
                
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_array,
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
    
    def pad_or_truncate_spectrogram(self, spec, target_length=128):
        """Pad or truncate spectrogram to fixed length"""
        if spec.shape[1] > target_length:
            # Truncate
            spec = spec[:, :target_length]
        elif spec.shape[1] < target_length:
            # Pad with zeros
            padding = np.zeros((spec.shape[0], target_length - spec.shape[1]))
            spec = np.hstack([spec, padding])
        
        return spec
    
    def normalize_features(self, features):
        """Normalize features using StandardScaler"""
        return self.scaler.fit_transform(features)
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor state"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_preprocessor(cls, filepath):
        """Load a saved preprocessor"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class FSDDDataset2D(Dataset):
    """PyTorch Dataset for FSDD with spectrograms"""
    
    def __init__(self, features, labels):
        # Reshape features for 2D CNN: (batch, 1, height, width)
        self.features = torch.FloatTensor(features).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_fsdd_dataset():
    """Load and prepare FSDD dataset"""
    print("Loading FSDD dataset...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("mteb/free-spoken-digit-dataset")
    
    # Extract audio data and labels
    audio_data = []
    labels = []
    
    for split in ['train', 'test']:
        for item in dataset[split]:
            # Get audio array directly
            audio_array = item['audio']['array']
            sample_rate = item['audio']['sampling_rate']
            audio_data.append((audio_array, sample_rate))
            labels.append(int(item['label']))
    
    print(f"Loaded {len(audio_data)} audio files")
    return audio_data, labels

def prepare_dataset_2d(audio_data, labels, preprocessor, target_length=128):
    """Prepare dataset with extracted spectrograms"""
    print("Extracting spectrograms...")
    
    features = []
    valid_labels = []
    
    for (audio_array, sample_rate), label in tqdm(zip(audio_data, labels), total=len(audio_data)):
        # Resample if needed
        if sample_rate != preprocessor.sample_rate:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=sample_rate, 
                target_sr=preprocessor.sample_rate
            )
        
        # Extract spectrogram
        spec = preprocessor.extract_spectrogram_from_audio(audio_array, target_length)
        if spec is not None:
            features.append(spec)
            valid_labels.append(label)
    
    features = np.array(features)
    labels = np.array(valid_labels)
    
    print(f"Extracted spectrograms shape: {features.shape}")
    return features, labels

def create_data_loaders_2d(features, labels, batch_size=32, test_size=0.2, random_state=42):
    """Create train and validation data loaders for 2D CNN"""
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    X_train_norm = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)
    
    # Create datasets
    train_dataset = FSDDDataset2D(X_train_norm, y_train)
    val_dataset = FSDDDataset2D(X_val_norm, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler
