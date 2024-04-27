import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from keras import backend as K
import numpy as np
import math, os

from dotenv import load_dotenv
load_dotenv()

ok_pic_formats = {"jpg", "jpeg", "webp", "png"}

POISON_PIC_LOCATION = os.getenv("NIGHTSHADE_PIC_LOCATION")
REGULAR_PIC_LOCATION = os.getenv("REGULAR_PIC_LOCATION")


def nearest_lower_exponent_of_2(n):
    if n < 1:
        return 0
    power = 1
    while power <= n:
        power <<= 1  # This is equivalent to power = power * 2
    return power >> 1  # Shift right to get the nearest lower power of 2


class AutoEncoderHiddenLayerDimensionIterator:
    def __init__(self, current_dim, input_dim, scaling_rate, input_to_waist_ratio):
        self.__current_dim = current_dim

        self.__input_dim = input_dim # what is the size of the input?
        self.__scaling_rate = scaling_rate
        self.__input_to_waist_ratio = input_to_waist_ratio

    def __iter__(self):
        return self
    
    def __next__(self):
        new_dim = math.ceil(self.__current_dim*self.__scaling_rate)
        if  self.__input_dim/new_dim <= self.__input_to_waist_ratio:
            self.__current_dim = new_dim # update_current dim
            return self.__current_dim
        else:
            raise StopIteration

class AutoEncoder(models.Model):
    def __init__(self, input_dim, latent_dim_scaling_rate, input_to_waist_ratio, add_reconstruction_loss: bool = True, variational: bool = True):
        super(AutoEncoder, self).__init__()
        self.__latent_dim = nearest_lower_exponent_of_2(input_dim[0]) # size of layer right after the input layer

        # higher to lower number, reverse this on the decoder side
        self.__hidden_layer_dimensions = [self.__latent_dim]
        self.__hidden_layer_dimensions.extend(
            [
                hidden_dim for hidden_dim in AutoEncoderHiddenLayerDimensionIterator(
                    current_dim=self.__latent_dim, input_dim=input_dim, 
                    scaling_rate=latent_dim_scaling_rate, input_to_waist_ratio=input_to_waist_ratio
                )
            ]
        )

        self.__add_reconstruction_loss = add_reconstruction_loss
        self.__variational = variational

        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(hidden_layer_dim, activation='relu') for hidden_layer_dim in self.__hidden_layer_dimensions
        ])

        # Decoder
        decoder = [
            layers.Dense(hidden_layer_dim, activation='relu') for hidden_layer_dim in reversed(self.__hidden_layer_dimensions)
        ]
        decoder.append(layers.Dense(input_dim, activation='sigmoid'))
        self.decoder = tf.keras.Sequential(decoder)

    def call(self, x):
        # TODO is this lazy execution until compile is called?
        # Encode
        encoded = self.encoder(x)
        
        # Add Random Noise to make it Variational & Set Decoder
        if self.__variational:
            mean, log_var = tf.split(encoded, num_or_size_splits=2, axis=1)
            epsilon = tf.random.normal(shape=(tf.shape(mean)[0], self.__next_dim))
            z = mean + tf.exp(log_var * 0.5) * epsilon
            reconstructed = self.decoder(z)
        else:
            reconstructed = self.decoder(encoded)
        
        # Loss Function
        kl_loss = -0.5 * tf.reduce_mean(log_var - tf.square(mean) - tf.exp(log_var) + 1)
        if self.__add_reconstruction_loss:
            reconstruction_loss = tf.mean(tf.square(reconstructed - x))
            self.add_loss(kl_loss + reconstruction_loss)
        else:
            self.add_loss(kl_loss)
        
        return reconstructed


# Example usage:
input_dim = 784
latent_dim = 2

exp_map = {
    "ae_kl_only": AutoEncoder(input_dim, latent_dim_scaling_rate=0.7, input_to_waist_ratio=4, add_reconstruction_loss=False, variational=False),
    "vae_kl_only": AutoEncoder(input_dim, latent_dim_scaling_rate=0.7, input_to_waist_ratio=4, add_reconstruction_loss=False, variational=True),
    "ae_kl_and_rl": AutoEncoder(input_dim, latent_dim_scaling_rate=0.7, input_to_waist_ratio=4, add_reconstruction_loss=True, variational=False),
    "vae_kl_and_rl": AutoEncoder(input_dim, latent_dim_scaling_rate=0.7, input_to_waist_ratio=4, add_reconstruction_loss=True, variational=True)
}

for exp_type, exp in exp_map.items():
    exp.compile(optimizer='adam')
    exp.fit()



