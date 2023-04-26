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

class LSTM_Encoder(Model):
    def __init__(self, timestep, input_dim, lstm_dim, activation, z_dim, rd, d, name='lstm_encoder', **kwargs):
        super(LSTM_Encoder, self).__init__(name=name, **kwargs)
        self.encoder_inputs = Input(shape=(timestep, input_dim), name='Input')
        self.encoder_lstm = LSTM(lstm_dim, activation=activation, recurrent_dropout=rd, dropout=d, name='lstm')
        self.encoder_batchnorm = BatchNormalization(momentum=0.6)
        self.z_mean = Dense(z_dim, name='z_mean', activation='tanh')
        self.z_log_sigma = Dense(z_dim, name='z_log_var', activation='tanh')
        self.z_sample = Sampling()

    def call(self, inputs):
        self.encoder_inputs = inputs
        hidden = self.encoder_lstm(self.encoder_inputs)
        hidden = self.encoder_batchnorm(hidden)
        z_mean = self.z_mean(hidden)
        z_log_sigma = self.z_log_sigma(hidden)
        z = self.z_sample((z_mean, z_log_sigma))
        return z_mean, z_log_sigma, z

class LSTM_Decoder(Layer):
    def __init__(self, timestep, input_dim, lstm_dim, activation, rd, d, name='lstm_decoder', **kwargs):
        super(LSTM_Decoder, self).__init__(name=name, **kwargs)
        self.z_inputs = RepeatVector(timestep, name='repeat_vector')
        self.decoder_lstm = LSTM(lstm_dim, activation=activation, recurrent_dropout=rd, dropout=d, return_sequences=True,
                                               name='lstm')
        self.dec = TimeDistributed(Dense(input_dim, activation='linear', name='time_distributed'))

    def call(self, inputs):
        z = self.z_inputs(inputs)
        hidden = self.decoder_lstm(z)
        pred = self.dec(hidden)
        return pred

loss_metric = Mean(name='loss')
recon_metric = Mean(name='recon_loss')
kl_metric = Mean(name='kl_loss')

class LSTM_VAE(Model):
    def __init__(self, timestep, input_dim, lstm_dim, activation, z_dim, rd, d, kl_weight, name='lstm_vae', **kwargs):
        super(LSTM_VAE, self).__init__(name=name, **kwargs)
        self.encoder = LSTM_Encoder(timestep, input_dim, lstm_dim, activation, z_dim, rd, d, **kwargs)
        self.decoder = LSTM_Decoder(timestep, input_dim, lstm_dim, activation, rd, d, **kwargs)
        self.timestep = timestep
        self.kl_weight= kl_weight

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
            loss = reconstruction + self.kl_weight*kl
            loss += sum(self.losses)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        loss_metric.update_state(loss)
        recon_metric.update_state(reconstruction)
        kl_metric.update_state(kl)
        return {'loss': loss_metric.result(), 'rec_loss': recon_metric.result(), 'kl_loss': kl_metric.result()}

    def test_step(self, inputs):
        z_mean, z_log_sigma, z, pred = self(inputs, training=True)
        reconstruction = self.reconstruct_loss(inputs, pred)
        kl = self.kl_weight *self.kl_divergence(z_mean, z_log_sigma)
        total_loss = K.mean(reconstruction + kl)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction,
            "kl_loss": kl,
        }