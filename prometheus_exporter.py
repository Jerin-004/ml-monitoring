from prometheus_client import start_http_server, Gauge
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import time

print("✅ Prometheus Metrics Exporter Running at http://127.0.0.1:8000/metrics")

loss_metric = Gauge("training_loss", "Training Loss")
accuracy_metric = Gauge("training_accuracy", "Training Accuracy")

start_http_server(8000)

(x_train, y_train), _ = mnist.load_data()
x_train = x_train / 255.0

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

for epoch in range(5):
    history = model.fit(x_train, y_train, epochs=1, verbose=0)
    loss = history.history['loss'][0]
    accuracy = history.history['accuracy'][0]
    loss_metric.set(loss)
    accuracy_metric.set(accuracy)
    print(f"Epoch {epoch+1} | Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# ✅ Keep server running forever
while True:
    time.sleep(1)
