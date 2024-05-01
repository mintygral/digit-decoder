# Handwritten-Digit-Recognizer

I made this project for my ENGR 133 class. This program creates a visual interface that allows users to draw handwritten digits on a canvas and have those digits recognized using a trained machine learning model.

## Files:
train_model.py: Contains the code for training the TensorFlow model using the MNIST dataset.

digit_recognizer.py: Contains the code for the graphical user interface and digit recognition functionality.

best_model.h5: The saved trained model used for prediction.

## Dependencies:
TensorFlow, Keras, Tkinter, PIL, NumPy

## Usage:
Run digit_recognizer.py to start the application.
Draw a digit on the canvas using your mouse.
Click the "Predict Digit" button to see the predicted digit and confidence score.
Click the "Clear Canvas" button to start a new drawing.
