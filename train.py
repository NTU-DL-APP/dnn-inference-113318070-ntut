import numpy as np
import tensorflow as tf
import os, json
from utils import mnist_reader

# === Step 1: 讀取 t10k 全部資料作為訓練 ===
x_all, y_all = mnist_reader.load_mnist("data/fashion", kind="t10k")
x_all = x_all.astype(np.float32) / 255.0
x_all = x_all.reshape(-1, 28, 28)
x_train, y_train = x_all, y_all  # 全部資料都拿來訓練

# === Step 2: 建立強化模型 ===
model = tf.keras.Sequential(name="mlp")
model.add(tf.keras.Input(shape=(28, 28)))
model.add(tf.keras.layers.Flatten(name="flatten"))
model.add(tf.keras.layers.Dense(512, activation="relu", name="dense_relu1"))
model.add(tf.keras.layers.Dropout(0.4, name="dropout1"))
model.add(tf.keras.layers.Dense(256, activation="relu", name="dense_relu2"))
model.add(tf.keras.layers.Dropout(0.3, name="dropout2"))
model.add(tf.keras.layers.Dense(10, activation="softmax", name="dense_softmax"))

# === Step 3: Callbacks ===
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# === Step 4: 訓練模型 ===
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=150,
          batch_size=128,
          validation_split=0.2,
          callbacks=[lr_scheduler, early_stop])

# === Step 5: 儲存 .h5 模型（可選）===
os.makedirs("model", exist_ok=True)
model.save("model/fashion_mnist.h5")

# === Step 6: 儲存權重為 .npz ===
weights = {}
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        w, b = layer.get_weights()
        weights[f"{layer.name}_W"] = w
        weights[f"{layer.name}_b"] = b
np.savez("model/fashion_mnist.npz", **weights)

# === Step 7: 儲存架構為 .json ===
model_arch = []
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Flatten):
        model_arch.append({
            "name": layer.name,
            "type": "Flatten",
            "config": {},
            "weights": []
        })
    elif isinstance(layer, tf.keras.layers.Dense):
        model_arch.append({
            "name": layer.name,
            "type": "Dense",
            "config": {"activation": layer.activation.__name__},
            "weights": [f"{layer.name}_W", f"{layer.name}_b"]
        })
with open("model/fashion_mnist.json", "w") as f:
    json.dump(model_arch, f, indent=2)
