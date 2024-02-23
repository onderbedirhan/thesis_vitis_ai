import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# Random number fixed
tf.random.set_seed(0)

# Prepare dataset for learning and testing
# This function can handle the dataset needed to train MNIST
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize image data for training
x_train = x_train/255.0
x_test = x_test/255.0

# AI model layer creation
layers_list = []

# Convert to one-dimensional array
layers_list.append(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
# all binding layers (128), activation specifies ReLu
layers_list.append(tf.keras.layers.Dense(128, activation='relu'))
# Dropout(0.2)
layers_list.append(tf.keras.layers.Dropout(0.2))
# all coupling layers (10), activation specifies softmax, where is the final output
layers_list.append(tf.keras.layers.Dense(10, activation='softmax'))

# AI Model Setup
mnist_model = tf.keras.models.Sequential(layers_list)

# AI model training setup
mnist_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# AI model training run
mnist_model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test))

# Quantize and store AI models
quantizer = vitis_quantize.VitisQuantizer(mnist_model)
# Number of data sets to be passed is 100-1000 without labels, so let's say 500.
quantized_model = quantizer.quantize_model(calib_dataset=x_train[0:500])
quantized_model.save("quantized_model.h5")
