# LLM Coding Challenge - Audio Digit Classification

This project demonstrates a lightweight prototype that listens to spoken digits (0-9) and predicts the correct number using 2D CNN with spectrograms.

## 🚀 Features

### Core Functionality
- **2D CNN Model**: Uses spectrograms for robust audio feature extraction
- **Real-time Processing**: Minimal delay between input and output
- **Dual Input Methods**: 
  - File upload with manual trimming
  - Live microphone recording
- **Interactive UI**: Clean Gradio interface with confidence scoring

### Technical Highlights
- **Lightweight Architecture**: Optimized for speed and functionality
- **Noise Handling**: Built-in noise reduction for real-world conditions
- **Modular Design**: Clean separation of concerns across components
- **Extensible Codebase**: Easy to modify and extend

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd digit_classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

Uses the **Free Spoken Digit Dataset (FSDD)** from Hugging Face:
- WAV recordings at 8kHz
- Spoken digits (0-9) by multiple English speakers
- Dataset: `mteb/free-spoken-digit-dataset`

## 🏗️ Project Structure

```
digit_classification/
├── models/
│   ├── model.py             # 2D CNN architecture
│   ├── train.py             # Training script
│   └── saved/               # Trained models
├── utils/
│   └── preprocessing.py     # Data preprocessing
├── audio/
│   └── processor.py         # Audio processing utilities
├── ui/
│   └── gradio_app.py        # Interactive web interface
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Train the Model
```bash
# Train 2D CNN model
python models/train.py
```

### 2. Launch the Interface
```bash
# Start Gradio app
python ui/gradio_app.py
```

### 3. Use the Application
- **File Upload**: Upload audio files and use manual trimming
- **Microphone**: Record live audio for real-time prediction
- **Results**: View predicted digit with confidence scores

## 🧠 Model Architecture

### 2D CNN with Spectrograms
- **Input**: Mel-spectrograms (128x128)
- **Architecture**: 4 convolutional layers with batch normalization
- **Features**: 256 filters in final layer
- **Output**: 10-class classification (digits 0-9)

### Key Design Decisions
- **Spectrograms over MFCCs**: Better spatial feature representation
- **2D CNN**: Leverages spatial patterns in spectrograms
- **Lightweight**: Optimized for speed without sacrificing accuracy
- **Real-time**: Minimal preprocessing for low latency

## Performance

### Model Metrics
- **Training Time**: ~5-10 minutes on CPU
- **Inference Time**: <100ms per prediction
- **Test Accuracy**: 96.17% on FSDD dataset
- **Model Size**: 10.0 MB (2D CNN with 2.62M parameters)
- **Memory Usage**: <50MB RAM during inference

### Detailed Performance Breakdown
- **Overall Accuracy**: 96.17% (577/600 correct predictions)
- **Per-digit Performance**: 
  - **Most Accurate**: Digit "4" (100.0% accuracy)
  - **Strong Performers**: Digits "0", "6", "9" (98.3% each)
  - **Most Challenging**: Digit "8" (90.0% accuracy)
  - **All digits**: 90-100% individual accuracy
- **Precision & Recall**: 
  - **Macro Average**: 96% precision, 96% recall, 96% F1-score
  - **Weighted Average**: 96% precision, 96% recall, 96% F1-score
- **Confusion Matrix**: Very low confusion between digits
- **Real-time Performance**: <100ms end-to-end processing

### Detailed Classification Report
```
              precision    recall  f1-score   support

     Digit 0       1.00      0.98      0.99        60
     Digit 1       0.98      0.97      0.97        60
     Digit 2       0.97      0.95      0.96        60
     Digit 3       0.93      0.93      0.93        60
     Digit 4       1.00      1.00      1.00        60
     Digit 5       1.00      0.97      0.98        60
     Digit 6       0.86      0.98      0.91        60
     Digit 7       0.95      0.95      0.95        60
     Digit 8       0.98      0.90      0.94        60
     Digit 9       0.97      0.98      0.98        60

    accuracy                           0.96       600
   macro avg       0.96      0.96      0.96       600
weighted avg       0.96      0.96      0.96       600
```

### Confusion Matrix
```
[[59  0  0  0  0  0  0  0  1  0]  # Digit 0
 [ 0 58  0  0  0  0  0  1  0  1]  # Digit 1
 [ 0  0 57  3  0  0  0  0  0  0]  # Digit 2
 [ 0  0  1 56  0  0  2  1  0  0]  # Digit 3
 [ 0  0  0  0 60  0  0  0  0  0]  # Digit 4
 [ 0  1  0  0  0 58  0  0  0  1]  # Digit 5
 [ 0  0  0  1  0  0 59  0  0  0]  # Digit 6
 [ 0  0  0  0  0  0  3 57  0  0]  # Digit 7
 [ 0  0  1  0  0  0  5  0 54  0]  # Digit 8
 [ 0  0  0  0  0  0  0  1  0 59]] # Digit 9
```

### Real-world Considerations
- **Noise Robustness**: Built-in noise reduction
- **Latency**: Optimized for real-time processing
- **Usability**: Intuitive interface for non-technical users


## Future Extensions

### Potential Enhancements
- **Multi-language Support**: Extend to other languages
- **Noise Simulation**: Add synthetic noise for robustness testing
- **Model Ensembling**: Combine multiple model architectures
- **Mobile Deployment**: Optimize for mobile devices

### Scalability
- **Batch Processing**: Handle multiple audio files
- **API Endpoints**: RESTful API for integration
- **Cloud Deployment**: Deploy to cloud platforms
