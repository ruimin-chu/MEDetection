import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import *
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class Sampling(Layer):
    def __init__(self, name='sampling_z'):
        super(Sampling, self).__init__(name=name)

    def call(self, inputs):
        z_mean, z_log_sigma = inputs
        epsilon = K.random_normal(shape=(tf.shape(z_mean)[0], 1), mean=0.0, stddev=1.0)
        return z_mean + z_log_sigma * epsilon

loss_metric = Mean(name='loss')
recon_metric = Mean(name='recon_loss')
kl_metric = Mean(name='kl_loss')

class Encoder(Model):
    def __init__(self, timestep, input_dim, hid_dim, activation, z_dim, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.encoder_inputs = Input(shape=(timestep, input_dim), name='Input')
        self.encoder = Dense(hid_dim, activation=activation)
        self.z_mean = Dense(z_dim, name='z_mean')
        self.z_log_sigma = Dense(z_dim, name='z_log_var')
        self.z_sample = Sampling()
        self.flat = Flatten()
        self.dropout = Dropout(0.2)

    def call(self, inputs):
        self.encoder_inputs = inputs
        flat = self.flat(self.encoder_inputs)
        hidden = self.encoder(flat)
        z_mean = self.z_mean(hidden)
        z_log_sigma = self.z_log_sigma(hidden)
        z = self.z_sample((z_mean, z_log_sigma))
        return z_mean, z_log_sigma, z

class Decoder(Layer):
    def __init__(self, timestep, input_dim, hid_dim, activation, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.decoder = Dense(hid_dim, activation=activation)
        self.dec = Dense(timestep, activation='linear')
        self.reshape = Reshape((timestep, input_dim))
        self.dropout = Dropout(0.2)

    def call(self, inputs):
        hidden =self.decoder(inputs)
        pred = self.reshape(self.dec(hidden))
        return pred

class VAE(Model):
    def __init__(self, timestep, input_dim, lstm_dim, activation, z_dim, kl_weight, name='lstm_vae', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(timestep, input_dim, lstm_dim, activation, z_dim, **kwargs)
        self.decoder = Decoder(timestep, input_dim, lstm_dim, activation, **kwargs)
        self.timestep = timestep
        self.kl_weight = kl_weight

    def call(self, inputs):
        z_mean, z_log_sigma, z = self.encoder(inputs)
        pred = self.decoder(z)
        return z_mean, z_log_sigma, z, pred

    def reconstruct_loss(self, inputs, pred):
        return K.mean(K.square(inputs - pred))

    def kl_divergence(self, z_mean, z_log_sigma):
        return -0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            z_mean, z_log_sigma, z, pred = self(inputs, training=True)
            reconstruction = self.reconstruct_loss(inputs, pred)
            kl = self.kl_divergence(z_mean, z_log_sigma)
            loss = reconstruction + self.kl_weight * kl
            loss += sum(self.losses)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # print(loss)
        loss_metric.update_state(loss)
        recon_metric.update_state(reconstruction)
        kl_metric.update_state(kl)
        return {'loss': loss_metric.result(), 'rec_loss': recon_metric.result(), 'kl_loss': kl_metric.result()}

    def test_step(self, inputs):
        z_mean, z_log_sigma, z, pred = self(inputs, training=True)
        reconstruction = self.reconstruct_loss(inputs, pred)
        kl = self.kl_weight * self.kl_divergence(z_mean, z_log_sigma)
        total_loss = K.mean(reconstruction + kl)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction,
            "kl_loss": kl,
        }