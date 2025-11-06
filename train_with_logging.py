import logging
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

class LogTrainingCallback:
    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs):
        loss = logs.get('loss')
        acc = logs.get('accuracy')
        logging.info(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")

log_callback = LogTrainingCallback(model)

for epoch in range(5):
    history = model.fit(x_train, y_train, epochs=1, verbose=0)
    logs = history.history
    log_callback.on_epoch_end(epoch, {'loss': logs['loss'][0], 'accuracy': logs['accuracy'][0]})
