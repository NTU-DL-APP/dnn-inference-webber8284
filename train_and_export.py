import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist

# === Step 1: Train Model ===
(x_train, y_train), _ = fashion_mnist.load_data()
x_train = x_train / 255.0  # Normalize to 0~1

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# Create output folder if needed
os.makedirs('model', exist_ok=True)

# Save h5 model (optional)
model.save('model/fashion_mnist.h5')

# === Step 2: Export JSON structure ===
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

# === Step 3: Export Weights ===
weights_dict = {}
for weight in model.weights:
    weights_dict[weight.name] = weight.numpy()

np.savez('model/fashion_mnist.npz', **weights_dict)

print("\n 模型訓練與轉換完成")
