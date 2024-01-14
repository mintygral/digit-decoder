import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flattening step
model.add(layers.Flatten())

# Fully connected layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=3, monitor='val_loss'),
    ModelCheckpoint(filepath='best_model.h5', save_best_only=True)
]

# Define the number of epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    # Code to perform training for one epoch
    print(f"Epoch #{epoch}")
    history = model.fit(train_images[..., tf.newaxis], train_labels, epochs=1, validation_split=0.2, callbacks=callbacks)

test_loss, test_acc = model.evaluate(test_images[..., tf.newaxis], test_labels)

model.save('best_model.h5')
