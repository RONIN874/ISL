# Indian Sign Language Gesture Recognition

This repository contains a deep learning project for real-time Indian Sign Language (ISL) recognition. It uses **MediaPipe** to extract hand landmarks and a **TensorFlow/Keras** Neural Network to classify the gestures into English alphabets (A-Z).

## Features
- **Real-Time Prediction:** Uses your webcam to predict gestures on the fly.
- **Two-Hand Support:** Extracts features dynamically whether one or two hands are present.
- **High Performance:** Pre-trained weights and robust preprocessing (Scikit-Learn Standard Scaler) ensure high-confidence classifications.
- **End-to-End Pipeline:** Contains a Jupyter Notebook (`module.ipynb`) for training the model on the raw CSV dataset, and a Python script (`predict.py`) for inference.

## Project Structure
- `predict.py` - The main Python script that runs the real-time webcam feed and outputs gesture predictions.
- `module.ipynb` - The Jupyter notebook containing the full data processing and model training pipeline.
- `isl_landmark_model_fixed.h5` - The trained TensorFlow model weights.
- `scaler.pkl` - A Scikit-learn StandardScaler fitted on the training data, used to normalize real-time landmarks.
- `label_encoder.pkl` - Maps the numerical classes back to the original text labels.
- `Indian Sign Language Gesture Landmarks.csv` - The original dataset containing coordinates for hand landmarks (Managed via Git LFS).
- `requirements.txt` - Required Python packages.

## Prerequisites & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RONIN874/ISL.git
   cd ISL
   ```

2. **Pull the Large Dataset (Optional):**
   Because the dataset is >100MB, it is stored using Git LFS. Ensure you have Git LFS installed on your system.
   ```bash
   git lfs pull
   ```

3. **Install the dependencies:**
   It is recommended to use a virtual environment. Run the following command:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Real-time Prediction
Run the prediction script to open your webcam and start predicting gestures. Ensure you are well-lit and your hands are clearly visible in the frame.
```bash
python predict.py
```
*Press `q` to quit the video stream.*

### 2. Training the Model
If you want to tweak the architecture or retrain the model, open the Jupyter Notebook:
```bash
jupyter notebook module.ipynb
```

## Technologies Used
- Python 3
- OpenCV
- MediaPipe
- TensorFlow / Keras
- Scikit-Learn
- Pandas & NumPy
