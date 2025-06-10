import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, Reshape
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D



# === Step 1: Load Data ===
(x_train, y_train), _ = fashion_mnist.load_data()
x_train = x_train / 255.0

# === Step 2: Stronger, Regularized Model ===
model = Sequential([
    Reshape((28,28,1), input_shape=(28,28)),
    Conv2D(32, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === Step 3: Train ===
model.fit(x_train, y_train, epochs=60, batch_size=64, validation_split=0.1)

# === Step 4: Export Model (.json + .npz) ===
os.makedirs('model', exist_ok=True)
model.save('model/fashion_mnist.h5')

# Save architecture
layers_config = []
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Flatten):
        layers_config.append({'name': layer.name, 'type': 'Flatten', 'config': {}, 'weights': []})
    elif isinstance(layer, tf.keras.layers.Dense):
        activation = layer.get_config()['activation']
        layers_config.append({
            'name': layer.name,
            'type': 'Dense',
            'config': {'activation': activation},
            'weights': [layer.weights[0].name, layer.weights[1].name]
        })

with open('model/fashion_mnist.json', 'w') as f:
    json.dump(layers_config, f, indent=2)

# Save weights
weights_dict = {}
for weight in model.weights:
    weights_dict[weight.name] = weight.numpy()

np.savez('model/fashion_mnist.npz', **weights_dict)

print("\n最終強化模型完成！")
