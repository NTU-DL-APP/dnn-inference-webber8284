import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import fashion_mnist

# === Step 1: Load & Normalize Data ===
(x_train, y_train), _ = fashion_mnist.load_data()
x_train = x_train / 255.0

# === Step 2: Build Model (強化版) ===
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === Step 3: Train Model ===
model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.1)

# === Step 4: Export model as JSON + NPZ ===
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

print("\n✅ 強化版模型訓練與轉換完成，準備測試！")
