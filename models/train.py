import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import (
    load_fsdd_dataset, 
    prepare_dataset_2d, 
    create_data_loaders_2d, 
    AudioPreprocessor2D
)
from models.model import create_2d_model, count_parameters

class Trainer2D:
    """Training class for 2D CNN audio digit classification"""
    
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training parameters
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs=50, early_stopping_patience=10):
        """Train the model"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model has {count_parameters(self.model):,} parameters")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/saved/best_model_2d.pth')
                print("Saved best model!")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss (2D CNN)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy (2D CNN)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/saved/training_history_2d.png')
        plt.show()

def main():
    """Main training function for 2D CNN"""
    # Create directories
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading dataset...")
    audio_data, labels = load_fsdd_dataset()
    
    # Create preprocessor for 2D CNN
    preprocessor = AudioPreprocessor2D(
        sample_rate=8000,
        n_fft=512,
        hop_length=256,
        n_mels=128
    )
    
    # Prepare features (spectrograms)
    features, labels = prepare_dataset_2d(audio_data, labels, preprocessor, target_length=128)
    
    # Save preprocessor
    preprocessor.save_preprocessor('models/saved/preprocessor_2d.pkl')
    
    # Create data loaders
    train_loader, val_loader, scaler = create_data_loaders_2d(
        features, labels, batch_size=16, test_size=0.2  # Smaller batch size for 2D CNN
    )
    
    # Save scaler
    with open('models/saved/scaler_2d.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create 2D CNN model
    model = create_2d_model('standard', num_classes=10)
    
    # Create trainer
    trainer = Trainer2D(model, train_loader, val_loader, device)
    
    # Train model
    train_losses, val_losses, train_accs, val_accs = trainer.train(epochs=30)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save final model
    torch.save(model.state_dict(), 'models/saved/final_model_2d.pth')
    
    print("Training completed!")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")

if __name__ == "__main__":
    main()
