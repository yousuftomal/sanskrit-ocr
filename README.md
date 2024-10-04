```markdown
# Sanskrit OCR using CV2 and CNN

This project implements Optical Character Recognition (OCR) for Sanskrit characters using Convolutional Neural Networks (CNN) and OpenCV (CV2). The model is trained on a dataset of Sanskrit character images, which are processed using image resizing and grayscale conversion. The trained model is capable of predicting the class of a Sanskrit character from an image.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Prediction Example](#prediction-example)
- [Contributing](#contributing)


## Features
- Sanskrit character recognition using deep learning.
- Utilizes TensorFlow and Keras for building and training a CNN.
- Dataset preprocessing with OpenCV: resizing, grayscale conversion, and normalization.
- Model evaluation and performance metrics.
- Save and load the trained model for future use.

## Installation

### Prerequisites
To run the project, you'll need the following libraries:

```bash
pip install -r requirements.txt
```

### Clone the repository
```bash
git clone https://github.com/yousuftomal/sanskrit-ocr.git
cd sanskrit-ocr
```

### Dataset
The project uses a preprocessed dataset of Sanskrit characters. This can either be created by you or obtained from the given sample dataset file, `dev_letter_D.p`, which should be in the same directory as your code.

## Usage

1. **Preprocess Dataset**: Convert the images into grayscale, resize them, and normalize the pixel values.
2. **Train the Model**: Train the CNN model with the training dataset and evaluate it with the test dataset.
3. **Predict Sanskrit Characters**: Use the trained model to predict Sanskrit characters from new images.

Here's how you can run the provided script:

```python
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the model
model = load_model('sanskrit_letters.model')

# Load and preprocess a sample image for prediction
sample_image = cv2.imread('path_to_image.png')
sample_image = cv2.resize(sample_image, (32, 32))
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
sample_image = np.array(sample_image, dtype=np.float32) / 255.0
sample_image = np.expand_dims(sample_image, axis=0)

# Predict the class
prediction = model.predict(sample_image)
predicted_class = np.argmax(prediction)

# Output the predicted class
print(f"Predicted class: {predicted_class}")
```

## Model Training and Evaluation

The model is defined as a simple CNN architecture with convolutional layers, max pooling, and dense layers. The dataset is split into training and testing sets using an 80/20 ratio.

```python
model = keras.Sequential([
    layers.Input(shape=(32, 32, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(602, activation='softmax')  # Assuming 602 classes for Sanskrit characters
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
```

### Evaluation
After training, the model can be evaluated on the test set:

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
```

### Saving the Model
The trained model can be saved and later loaded for predictions.

```python
model.save("sanskrit_letters.model")
```

## Prediction Example

Once the model is trained, you can use it to predict new Sanskrit character images. Here's how to use the model for character recognition:

```python
sample_image = cv2.imread("path_to_image.png")
sample_image = cv2.resize(sample_image, (32, 32))
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
sample_image = np.expand_dims(sample_image, axis=0)

# Predict the character
prediction = model.predict(sample_image)
predicted_class = np.argmax(prediction)
print(f"Predicted class: {predicted_class}")
```

## Contributing

We welcome contributions to improve the project! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.
