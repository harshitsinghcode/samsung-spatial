# Speech & Music Dereverberation with GAN + U-Net 🎵 
### By Batch 5 [Harshit, Hemanth, Akshita, Vishnuraj] 

This project implements a deep learning pipeline for **speech and music dereverberation** using a **U-Net / GAN architecture**. It is designed to enhance audio quality by removing reverberation while maintaining perceptual fidelity.

Key features include:

- **DSP preprocessing**: High-Pass Filter (HPF) and Weighted Prediction Error (WPE) dereverberation
- **Spectrogram conversion** for deep learning models
- **Generator & Discriminator (GAN) training** with adversarial + L1 loss
- **Evaluation metrics**: PESQ (speech quality) and SDR (music distortion ratio)
- **Model complexity analysis**: GMACs calculation for efficiency


## 📦 Setup

Install the required Python packages for audio processing, model training, and evaluation:

- PyTorch  
- Librosa  
- Pyroomacoustics  
- Pesq & Pystoi  
- Ptflops  
- Torch-AudioAugmentations  

> Ensure your environment supports GPU acceleration for faster training.


## 📂 Project Structure

- **Installations**: Set up dependencies and libraries.  
- **Imports**: Load core libraries like PyTorch, NumPy, Pandas, Librosa, and Pyroomacoustics.  
- **Config Class**: Defines dataset paths, sample rate, FFT parameters, WPE settings, and model hyperparameters.  
- **DSP Front-End**: Implements high-pass filtering, WPE dereverberation, and full DSP preprocessing.  
- **Models**:  
  - TinyUNet for lightweight evaluation  
  - GeneratorUNet + PatchGANDiscriminator for adversarial training  
- **Dataset Class**: Loads paired reverberant and clean audio, computes STFT, and returns spectrograms & raw waveforms.  
- **GAN Training Loop**:  
  - Trains generator and discriminator  
  - Includes validation at each epoch  
  - Saves final weights for inference  
- **Audio Playback & Visualization**: Converts predicted spectrograms back to audio, plays audio samples, and visualizes spectrograms.  
- **Model Complexity (GMACs)**: Calculates GMACs and parameter count for 1-second audio input.  
- **Evaluation Metrics**: Computes PESQ for speech quality and SDR for music quality.


## 🚀 Usage

### 1. Prepare Dataset
- Provide a CSV file listing paths to reverberant and clean WAV files.  
- Ensure folder structure is correct and paths are updated in the configuration.

### 2. Train GAN
- Run the GAN training pipeline to optimize both generator and discriminator networks.  
- The process includes epoch-wise validation and saving final weights.

### 3. Audio Reconstruction & Visualization
- Convert predicted spectrograms to waveform audio.  
- Play original reverberant, predicted dereverberated, and clean ground truth audio.  
- Plot spectrogram comparisons.

### 4. Model Complexity
- Evaluate GMACs and parameters to ensure the model fits within the computational budget.

### 5. Evaluation Metrics
- **PESQ**: Objective measure of speech quality  
- **SDR**: Signal-to-Distortion Ratio for music quality  

> Metrics are automatically computed for validation datasets.


## 📊 Metrics

- **GMACs**: Model efficiency (budget < 50 GMAC/s)  
- **PESQ**: Perceptual evaluation of speech quality  
- **SDR**: Signal-to-Distortion Ratio for music fidelity

<img width="928" height="301" alt="image" src="https://github.com/user-attachments/assets/3c1dc81c-b3b1-47ad-ae60-18507c3f1def" />


## 🎧 Outputs

- **Original Reverberant Audio**: Input audio with room reverberation  
- **Predicted Dereverberated Audio**: Model output  
- **Clean Ground Truth Audio**: Reference audio  
- **Spectrogram Plots**: Input, predicted, and target comparison


## ✅ Status Checks

- Dataset path verification  
- GMACs check (<50)  
- Training and validation loss monitoring  
- PESQ and SDR evaluation for performance benchmarking
