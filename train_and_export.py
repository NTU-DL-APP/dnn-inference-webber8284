import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import fashion_mnist

# === Step 1: Load Data ===
(x_train, y_train), _ = fashion_mnist.load_data()
x_train = x_train / 255.0

# === Step 2: Build Stronger Model ===
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(768, activation='relu'),
    Dropout(0.3),
    Dense(384, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),   # 新增
    Dropout(0.1),                     # 新增
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === Step 3: Train (長一點) ===
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.1)

# === Step 4: Export as .json + .npz ===
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

print("\n✅ 高階模型訓練與轉換完成，準備測試！")
