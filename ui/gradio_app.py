import gradio as gr
import numpy as np
import librosa
import os
import sys
import time
import threading
from typing import Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.processor import AudioClassifier2D, AudioProcessor2D

class GradioAudioClassifier:
    """Gradio interface for audio digit classification with 2D CNN"""
    
    def __init__(self, model_path: str, preprocessor_path: str, scaler_path: str):
        self.classifier = AudioClassifier2D(model_path, preprocessor_path, scaler_path)
        self.audio_processor = AudioProcessor2D()
        
        # Real-time processing state
        self.is_recording = False
        self.audio_buffer = np.array([])
        
    def predict_from_audio(self, audio: Tuple[int, np.ndarray]) -> Tuple[str, str, str, str]:
        """Predict digit from uploaded audio"""
        if audio is None:
            return "No audio provided", "0%", "Upload an audio file to get started", ""
        
        sample_rate, audio_data = audio
        
        # Convert audio data to float if needed
        if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
            # Convert to float and normalize to [-1, 1] range
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Resample if needed
        if sample_rate != self.audio_processor.sample_rate:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=self.audio_processor.sample_rate
            )
        
        # Normalize audio
        audio_data = self.audio_processor.normalize_audio(audio_data)
        
        # Predict
        predicted_digit, confidence, probabilities = self.classifier.predict(audio_data)
        
        if predicted_digit == -1:
            return "Error", "0%", "Could not process audio", "failed"
        
        # Format results
        digit_name = self.classifier.get_class_name(predicted_digit)
        confidence_str = f"{confidence:.1%}"
        
        # Create probability distribution text
        prob_text = "Probability distribution:\n"
        for i, prob in enumerate(probabilities):
            prob_text += f"Digit {i}: {prob:.3f}\n"
        
        # Determine status for styling
        status = "success" if confidence > 0.7 else "warning" if confidence > 0.4 else "low"
        
        return f"{digit_name}", confidence_str, prob_text, status
    

    
    def predict_from_microphone(self, audio: Tuple[int, np.ndarray]) -> Tuple[str, str, str, str]:
        """Predict digit from microphone input"""
        if audio is None:
            return "No audio recorded", "0%", "Click 'Start Recording' to begin", ""
        
        sample_rate, audio_data = audio
        
        # Convert audio data to float if needed
        if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
            # Convert to float and normalize to [-1, 1] range
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Resample if needed
        if sample_rate != self.audio_processor.sample_rate:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=self.audio_processor.sample_rate
            )
        
        # Normalize audio
        audio_data = self.audio_processor.normalize_audio(audio_data)
        
        # Apply noise reduction
        audio_data = self.audio_processor.apply_noise_reduction(audio_data)
        
        # Predict
        predicted_digit, confidence, probabilities = self.classifier.predict(audio_data)
        
        if predicted_digit == -1:
            return "Processing Error", "0%", "Could not process audio. Try speaking more clearly.", "failed"
        
        # Format results
        digit_name = self.classifier.get_class_name(predicted_digit)
        confidence_str = f"{confidence:.1%}"
        
        # Create probability distribution text
        prob_text = "Probability distribution:\n"
        for i, prob in enumerate(probabilities):
            prob_text += f"Digit {i}: {prob:.3f}\n"
        
        # Determine status for styling
        status = "success" if confidence > 0.7 else "warning" if confidence > 0.4 else "low"
        
        return f"{digit_name}", confidence_str, prob_text, status
    
    def update_result_styling(self, result: str, confidence: str, status: str):
        """Update result styling based on confidence level"""
        if status == "success":
            return f"{result}", f"{confidence}"
        elif status == "warning":
            return f"{result}", f"{confidence}"
        elif status == "low":
            return f"{result}", f"{confidence}"
        else:
            return result, confidence
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Minimal CSS
        css = """
        .header-text {
            text-align: center;
            margin-bottom: 2rem;
        }

        .instructions-box {
            background-color: var(--block-background-fill);
            color: white; /* Light text */
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .section-title {
            font-weight: 600;
            color: white; /* Dark navy text */
            margin-bottom: 0.5rem;
        }
        """
      

        
        with gr.Blocks(css=css, title="Audio Digit Classifier") as interface:
            
            # Simple header
            gr.HTML("""
                <div class="header-text">
                    <h1>🎤 Audio Digit Classifier</h1>
                    <p>Recognize spoken digits (0-9) using 2D CNN</p>
                </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    # File upload section
                    gr.HTML('<h3 class="section-title">📁 Upload Audio File</h3>')
                    
                    file_audio = gr.Audio(
                        label="Select audio file",
                        type="numpy",
                        sources=["upload"]
                    )
                    
                    file_predict_btn = gr.Button(
                        "🔍 Predict from File", 
                        variant="primary"
                    )
                    
                    # File results
                    file_result = gr.Textbox(
                        label="Predicted Digit",
                        value="Upload an audio file to get started",
                        interactive=False
                    )
                    file_confidence = gr.Textbox(
                        label="Confidence",
                        value="0%",
                        interactive=False
                    )
                    file_probabilities = gr.Textbox(
                        label="Probability Distribution",
                        value="",
                        lines=8,
                        interactive=False
                    )
                
                with gr.Column():
                    # Microphone section
                    gr.HTML('<h3 class="section-title">🎙️ Record from Microphone</h3>')
                    
                    mic_audio = gr.Audio(
                        label="Record audio",
                        type="numpy",
                        sources=["microphone"]
                    )
                    
                    mic_predict_btn = gr.Button(
                        "🔍 Predict from Recording", 
                        variant="primary"
                    )
                    
                    # Microphone results
                    mic_result = gr.Textbox(
                        label="Predicted Digit",
                        value="Click 'Record' to start recording",
                        interactive=False
                    )
                    mic_confidence = gr.Textbox(
                        label="Confidence",
                        value="0%",
                        interactive=False
                    )
                    mic_probabilities = gr.Textbox(
                        label="Detailed Results",
                        value="",
                        lines=8,
                        interactive=False
                    )
            
            # Simple instructions
            gr.HTML("""
                <div class="instructions-box">
                    <h3>How to Use</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <h4>🎙️ Microphone Recording</h4>
                            <ul>
                                <li>Click "Record" button</li>
                                <li>Speak a digit (0-9) clearly</li>
                                <li>Click "Predict from Recording"</li>
                            </ul>
                        </div>
                        <div>
                            <h4>📁 File Upload</h4>
                            <ul>
                                <li>Upload WAV, MP3, or M4A file</li>
                                <li>Use trim option to select required part</li>
                                <li>Click "Predict from File"</li>
                                <li>View results and confidence</li>
                            </ul>
                        </div>
                    </div>
                    <p><strong>Tips:</strong> Speak clearly in a quiet environment. Use the trim feature in file upload to select only the digit part of longer recordings.</p>
                </div>
            """)
            
            # Hidden status components for styling updates
            file_status = gr.Textbox(visible=False)
            mic_status = gr.Textbox(visible=False)
            
            # Event handlers
            file_predict_btn.click(
                fn=self.predict_from_audio,
                inputs=[file_audio],
                outputs=[file_result, file_confidence, file_probabilities, file_status]
            )
            
            mic_predict_btn.click(
                fn=self.predict_from_microphone,
                inputs=[mic_audio],
                outputs=[mic_result, mic_confidence, mic_probabilities, mic_status]
            )
        
        return interface

def main():
    """Main function to run the Gradio app"""
    
    # Check if model files exist
    model_path = "models/saved/best_model_2d.pth"
    preprocessor_path = "models/saved/preprocessor_2d.pkl"
    scaler_path = "models/saved/scaler_2d.pkl"
    
    if not all(os.path.exists(path) for path in [model_path, preprocessor_path, scaler_path]):
        print("2D CNN model files not found!")
        print("Please run the 2D CNN training script first:")
        print("python models/train_2d.py")
        return
    
    # Create classifier
    classifier = GradioAudioClassifier(model_path, preprocessor_path, scaler_path)
    
    # Create and launch interface
    interface = classifier.create_interface()
    
    print("Launching Enhanced Audio Digit Classifier...")
    print("🔗 Share link will be generated for easy access")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )

if __name__ == "__main__":
    main()