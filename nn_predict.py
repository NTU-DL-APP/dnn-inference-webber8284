import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

# === Flatten Layer ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense Layer ===
def dense(x, W, b):
    return x @ W + b

# === Forward Pass ===
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            act = cfg.get("activation", None)
            if act == "relu":
                x = relu(x)
            elif act == "softmax":
                x = softmax(x)
    return x

# === Entry Point ===
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
