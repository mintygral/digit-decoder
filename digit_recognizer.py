import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import Tk, Canvas, Button, Label
from PIL import Image, ImageDraw
import numpy as np

# Load the trained model
try:
    model = load_model('best_model.h5')
    model_loaded = True
except Exception as e:
    print(f"Error loading the model: {e}")
    model_loaded = False

# Global variables for canvas and image
canvas = None
draw = None
confidence_threshold = 0.85

def predict_digit():
    global canvas

    if not model_loaded:
        result_label.config(text="Error: Model not loaded!")
        return

    try:
        # Get the image from the canvas
        filename = "image.png"
        canvas.postscript(file=filename, colormode='color')
        img = Image.open(filename)

        # Convert the image to grayscale
        img = img.convert('L')

        # Resize the image to 28x28 pixels
        img = img.resize((28, 28))

        # Invert the colors (black background, white drawing)
        img = Image.eval(img, lambda x: 255 - x)

        # Save the image for debugging (optional)
        img.save("drawn_image.png")

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Normalize pixel values
        img_array = img_array / 255.0  

        # Reshape the array to match the model's expected input shape
        img_array = img_array.reshape(1, 28, 28, 1)

        if np.max(img_array) == 0:
            # Handle the case where no drawing is detected
            result_label.config(text="No drawing detected!")
        else:
            # Make multiple predictions
            num_predictions = 10
            predictions = model.predict(img_array)
            
            # Check each prediction
            valid_predictions = []
            for _ in range(num_predictions):
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions)
                confidence = predictions[0][predicted_class]

                if confidence >= confidence_threshold:
                    valid_predictions.append((predicted_class, confidence))

            if not valid_predictions:
                # No valid predictions found
                result_label.config(text="Not recognized as a digit!")
            else:
                # Display the highest confidence prediction
                best_prediction = max(valid_predictions, key=lambda x: x[1])
                predicted_class, confidence = best_prediction
                result_label.config(text=f"Predicted Digit: {predicted_class}, Confidence: {confidence:.3f}")

    except Exception as e:
        # Handle any errors that may occur during the prediction process
        result_label.config(text=f"Error predicting digit: {e}")

def draw_on_canvas(event):
    global canvas, draw

    if draw:
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        canvas.create_oval(x1, y1, x2, y2, fill="black", width=8)

def start_drawing(event):
    global draw
    draw = True

def stop_drawing(event):
    global draw
    draw = False

def clear_canvas():
    global canvas, draw

    canvas.delete("all")
    result_label.config(text="Predicted Digit: ")

    # Reset the drawing
    draw = None

def main():
    # Create the main window
    global root
    root = Tk()
    root.title("Digit Recognizer")

    # Create a drawing canvas
    global canvas
    canvas = Canvas(root, width=280, height=280, bg="white")
    canvas.pack()

    # Bind mouse events to the canvas
    canvas.bind("<B1-Motion>", draw_on_canvas)
    canvas.bind("<ButtonRelease-1>", stop_drawing)
    canvas.bind("<Button-1>", start_drawing)

    # Create buttons
    predict_button = Button(root, text="Predict Digit", command=predict_digit)
    predict_button.pack(pady=10)

    clear_button = Button(root, text="Clear Canvas", command=clear_canvas)
    clear_button.pack(pady=10)

    # Display the predicted digit
    global result_label
    result_label = Label(root, text="Predicted Digit: ")
    result_label.pack(pady=10)

    # Run the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()
