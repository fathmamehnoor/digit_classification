import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AudioDigitClassifier2D(nn.Module):
    """2D CNN for audio digit classification using spectrograms"""
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(AudioDigitClassifier2D, self).__init__()
        
        # 2D CNN layers for spectrogram processing
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Input shape: (batch, 1, time_steps, frequency_bins)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def predict_proba(self, x):
        """Return probability predictions"""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def predict(self, x):
        """Return class predictions"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

class LightweightAudioClassifier2D(nn.Module):
    """Lighter 2D CNN version for faster inference"""
    
    def __init__(self, num_classes=10):
        super(LightweightAudioClassifier2D, self).__init__()
        
        # Smaller network
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Simple fully connected layers
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_2d_model(model_type='standard', num_classes=10):
    """Create 2D CNN model based on type"""
    if model_type == 'lightweight':
        model = LightweightAudioClassifier2D(num_classes)
    else:
        model = AudioDigitClassifier2D(num_classes)
    
    print(f"Created {model_type} 2D CNN model with {count_parameters(model):,} parameters")
    return model
