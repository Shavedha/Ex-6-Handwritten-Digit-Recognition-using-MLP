# Skill Assisessment-Handwritten Digit Recognition using MLP
## Aim:
To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## Theory:
To recognize handwritten digits using a Multilayer Perceptron (MLP) in Python, we can use a library like TensorFlow or PyTorch. Below is an example using TensorFlow and the  MNIST dataset, which is a collection of 28x28 pixel grayscale images of handwritten digits from 0 to 9. The MLP model is defined using the Keras Sequential API with two dense layers. The input data is flattened to a 1D array before passing through the dense layers. The output layer has 10 units with softmax activation for multiclass classification.


## Algorithm :
1. **Import Libraries:**
   - Import the necessary libraries, including TensorFlow, Keras layers, Matplotlib, and NumPy.

2. **Load MNIST Dataset:**
   - Use `tf.keras.datasets.mnist.load_data()` to load the MNIST dataset, which consists of handwritten digit images.

3. **Normalize Pixel Values:**
   - Normalize the pixel values of the images to be between 0 and 1 by dividing the pixel values by 255.0.

4. **Build the MLP Model:**
   - Create an MLP model using `models.Sequential()`.
   - Flatten the 28x28 images to a 1D array using `layers.Flatten()`.
   - Add a dense hidden layer with 128 units and ReLU activation using `layers.Dense(128, activation='relu')`.
   - Apply dropout regularization with a dropout rate of 0.2 using `layers.Dropout(0.2)`.
   - Add the output layer with 10 units (for 10 classes) and softmax activation using `layers.Dense(10, activation='softmax')`.

5. **Compile the Model:**
   - Compile the model using `model.compile()` with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the metric.

6. **Train the Model:**
   - Train the model on the training data using `model.fit()` with the specified number of epochs (e.g., 5).

7. **Evaluate the Model:**
   - Evaluate the trained model on the test set using `model.evaluate()` and print the test accuracy.

8. **Visualize Handwritten Numbers:**
   - Choose a number of images (e.g., 5) to visualize randomly from the test set.
   - For each selected image, reshape it to the format expected by the model (expand dimensions).
   - Use the trained model to make predictions on the reshaped image.
   - Display the original image, along with its true label and the predicted label using Matplotlib.

9. **Run the Code:**
   - Execute the entire code to load data, build, compile, train, evaluate, and visualize results.

10. **Review Results:**
    - Observe the printed test accuracy and visually inspect the displayed images.



## Program:
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build the MLP model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Visualize some handwritten numbers
num_images_to_visualize = 5
indices_to_visualize = np.random.choice(len(test_images), num_images_to_visualize, replace=False)

for i in indices_to_visualize:
    image = test_images[i]
    label = test_labels[i]

    # Reshape the image from (28, 28) to (1, 28, 28) for prediction
    image = np.expand_dims(image, axis=0)

    # Make a prediction
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    # Display the image and its true/predicted labels
    plt.figure()
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.title(f'True Label: {label}, Predicted Label: {predicted_label}')
    plt.show()

```


## Output :
<img width="628" alt="image" src="https://github.com/Shavedha/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/93427376/2e618a15-4fda-4880-b257-e82bd34eaed3">

<img width="541" alt="image" src="https://github.com/Shavedha/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/93427376/bd0cc29b-707d-487d-b899-15497eb980ce">

<img width="535" alt="image" src="https://github.com/Shavedha/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/93427376/e20c1822-4dd9-4fd6-aa75-7f878736e91e">

<img width="550" alt="image" src="https://github.com/Shavedha/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/93427376/28a92c88-f060-4b45-89ad-cf9876119ccc">

<img width="565" alt="image" src="https://github.com/Shavedha/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/93427376/489a18a3-e869-4515-a87a-b7dbca453a93">


<img width="402" alt="image" src="https://github.com/Shavedha/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/93427376/775a90a5-c2c6-46ba-9e2f-a1c1824a45d6">


## Result:
Thus handwritten digits are recognised using MLP successfully.
