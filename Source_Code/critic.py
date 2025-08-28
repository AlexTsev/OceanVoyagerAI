import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
from myconfig import myconfig

# GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")
else:
    print("GPU not found, using CPU.")

class Critic:
    def __init__(self, observation_dimensions):
        self.alpha = myconfig['critic_alpha']
        self.epochs = myconfig['critic_epochs']

        inputs = Input(shape=(observation_dimensions,), name='critic_input')
        x = Dense(100)(inputs)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(100)(x)
        x = LeakyReLU(alpha=0.1)(x)
        outputs = Dense(1, activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.alpha),
                           loss='mse')

    def train(self, x, y, batch_size=64):
        self.model.fit(x, y, epochs=self.epochs, batch_size=batch_size, verbose=0)

    def predict(self, x):
        return self.model.predict(x, verbose=0)