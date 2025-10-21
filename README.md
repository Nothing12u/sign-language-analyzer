# sign-language-analyzer

Hand Gesture Recognition System 🖐️
This is a Python-based Hand Gesture Recognition System that uses computer vision (OpenCV) and a Convolutional Neural Network (Keras/TensorFlow) to classify hand gestures captured by a webcam. It includes modules for skin calibration, data collection, model training, and real-time gesture recognition in two modes: Text Mode and Calculator Mode.

🚀 Features
Skin Color Calibration: Calibrates the system to your skin tone using HSV histograms for robust hand segmentation.

Data Collection: Captures 50x50 pixel grayscale images of gestures from the webcam, automatically crops to the hand contour, and saves them for training.

Data Augmentation: Flips existing images to double the training dataset size.

CNN Model Training: Trains a Sequential Convolutional Neural Network model using the collected image data.

Real-time Recognition: Uses the trained model to predict gestures in real-time.

Application Modes:

Text Mode: Spells out letters one by one into a word or sentence.

Calculator Mode: Allows performing basic arithmetic operations using numbered gestures (0-9 for numbers/operators) and a specialized 'equals' gesture.

⚙️ Prerequisites
Before running the system, ensure you have Python and the following libraries installed:

Bash

pip install opencv-python numpy pickle-mixin tensorflow keras scikit-learn pyttsx3
🛠️ Usage
The system is managed through a main command-line menu.

1. Setup and Calibration
This is the most crucial first step for accurate recognition.

Run the main script:

Bash

python your_script_name.py
Choose Option 0: Calibrate skin color.

Place your hand inside the blue rectangle displayed in the webcam window.

Press 'c' to capture the skin color histogram (saved to a file named hist).

Press 'q' to quit the calibration window.

2. Data Collection and Preparation
For each gesture you want the system to recognize, follow these steps:

Choose Option 1: Collect new gesture data.

Enter a unique Gesture ID (e.g., 0, 1, 2) and Gesture Name (e.g., A, B, C).

A window will open. Hold your hand in the required gesture within the green box.

Press 'c' to start capturing 1200 images. The count will increase on the screen.

Repeat steps 1-4 for every unique gesture (e.g., all 26 letters, numbers 0-9, and special control gestures like C for clear, Best of Luck for equals/enter).

3. Model Training
Option 2: Augment data (flip images) (Optional but Recommended)

Option 3: Prepare training data (Splits images into training/validation sets and saves them as pickle files).

Option 4: Train model. This trains the CNN on the prepared data. The best model will be saved as cnn_model_keras2.h5.

4. Real-time Recognition
Choose Option 5: Run real-time recognition.

The application starts in Text Mode (type 'c' to switch to Calculator Mode, or 'q' to exit).

In Text Mode, hold a gesture until the prediction is stable (the word is built up letter by letter).

In Calculator Mode (type 't' to switch back), use number gestures (e.g., 1-9, 0) for operands/operators and the 'Best of Luck ' gesture (if collected) to perform the calculation.
