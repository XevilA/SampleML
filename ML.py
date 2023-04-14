import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense

# Create some toy data
x_train = np.random.randn(100, 10)
y_train = np.random.randn(100, 1)

# Define the neural network model
model = tf.keras.Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(x_train, y_train, epochs=10)

# Export the model as a graph image
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
