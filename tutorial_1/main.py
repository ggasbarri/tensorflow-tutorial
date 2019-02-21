import random

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# Step 1 - Import the data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Step 1.a - Visualize images
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
plt.figure(figsize=(10, 10))

# Step 1.b - Pre-process data
# -- Images are matrices with values from 0 to 255 that describe the color
# -- We can limit the range to 0 to 1 in order to enhance performance
train_images = train_images / 255.0
test_images = test_images / 255.0

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()

# Step 2 - Define Neural Network
model = tf.keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax, ))  # Output layer

# Step 3 - Compile the model
model.compile(optimizer='rmsprop',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

# Step 4.1.a - Train the model
model.fit(train_images, train_labels, epochs=5)

# -------------------  OR  ---------------------

# Step 4.2.b - Restore the weights (optional)
# model.load_weights('./checkpoints/saved_weights')

# Step 4.3 - Save the weights (optional)
# model.save_weights('./checkpoints/saved_weights')


# Step 5 - Evaluate
loss, acc = model.evaluate(test_images, test_labels)
print('Loss: ' + str(loss))
print('Accuracy: ' + str(acc))


# Show random prediction + corresponding image

random_index = random.randint(0, len(test_images) - 1)

test_prediction = model.predict(test_images[random_index:random_index + 1])

plt.figure()
plt.imshow(test_images[random_index])
plt.grid(False)
plt.xlabel(np.argmax(test_prediction))
plt.show()
plt.figure(figsize=(10, 10))
