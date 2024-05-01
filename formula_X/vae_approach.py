import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Lambda, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, InputLayer, Reshape
    )
from tensorflow.keras import Model, Input
from tensorflow.keras.losses import MeanSquaredError

import numpy as np
import math, os, cv2, argparse


from dotenv import load_dotenv
load_dotenv()

import image_resizing as img

ok_pic_formats = {"jpg", "jpeg", "webp", "png"}

POISON_PIC_LOCATION = os.getenv("NIGHTSHADE_PIC_LOCATION")
REGULAR_PIC_LOCATION = os.getenv("REGULAR_PIC_LOCATION")
HOMEMADE_POISON_PIC_LOCATION = os.getenv("HOMEMADE_POISON_PIC_LOCATION")

RESIZED_REGULAR_PIC_FILE_NAME = os.getenv("RESIZED_REGULAR_PIC_FILE_NAME")
PADDED_REGULAR_PIC_FILE_NAME = os.getenv("PADDED_REGULAR_PIC_FILE_NAME")

RESIZED_POISONED_PIC_FILE_NAME = os.getenv("RESIZED_POISONED_PIC_FILE_NAME")
PADDED_POISONED_PIC_FILE_NAME = os.getenv("PADDED_POISONED_PIC_FILE_NAME")


def nearest_lower_exponent_of_2(n):
    if n < 1:
        return 0
    power = 1
    while power <= n:
        power <<= 1  # -> power = power * 2
    return power >> 1  # Shift right to get the nearest lower power of 2 | binary byte shifting man


def reconstruction_loss(y_true, y_pred):
    # Choose an appropriate loss based on your data:
    return tf.mean(tf.square(y_true - y_pred))  # Example: Mean Squared Error

def kl_loss(z_mean, z_log_var):
    return -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)


def log_normal_pdf(sample, mean, log_var, r_axis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-log_var) + log_var + log2pi),
      axis=r_axis)


def CVAE_loss(model, x):
    z_mean, z_log_var = model.encode(x)
    z = model.reparameterize(z_mean, z_log_var)
    x_logit = model.decode(z)
            
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=input
        )
    logpx_z = -tf.reduce_sum(cross_entropy, axis=[1,2,3])
    logpz = log_normal_pdf(z, 0, 0)
    logqz_x = log_normal_pdf(z, z_mean, z_log_var)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = CVAE_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

class AutoEncoderHiddenLayerDimensionIterator: # Helps you iterate thru automatic scaling of neural net just once
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


class Autoencoder(Model):
    def __init__(self, input_dim, intermediate_dim_scaling_rate, input_to_waist_ratio,
                 add_reconstruction_loss: bool = True, variational: bool = True):
        super(Autoencoder, self).__init__()

        self.__input_dim = input_dim

        # higher to lower number, reverse this on the decoder side
        hidden_layer_1_dim = nearest_lower_exponent_of_2(self.__input_dim[0])
        print(hidden_layer_1_dim)
        self.__hidden_layer_dimensions = [nearest_lower_exponent_of_2(self.__input_dim[0])]
        self.__hidden_layer_dimensions.extend(
            [
                hidden_dim for hidden_dim in AutoEncoderHiddenLayerDimensionIterator(
                    current_dim=hidden_layer_1_dim, input_dim=self.__input_dim[0], 
                    scaling_rate=intermediate_dim_scaling_rate, input_to_waist_ratio=input_to_waist_ratio
                )
            ]
        )

        print(self.__hidden_layer_dimensions)
        self.__latent_dim = self.__hidden_layer_dimensions[-1]

        self.__add_reconstruction_loss = add_reconstruction_loss
        self.__variational = variational

        # Encoder Network
        self.__encoder = [
            Dense(hidden_layer_dim, activation='relu') for hidden_layer_dim in self.__hidden_layer_dimensions
            ] 
        
        if self.__variational:
            self.__dense_mean = Dense(self.__latent_dim)
            self.__dense_log_var = Dense(self.__latent_dim)
            self.__sampling = Lambda(self.sampling_fn)

        # Decoder Network
        self.__decoder = [
            Dense(hidden_layer_dim, activation='relu') for hidden_layer_dim in reversed(self.__hidden_layer_dimensions)
            ]
        self.__output = Dense(self.__input_dim, activation='sigmoid')

    def sampling_fn(self, args): # only used when variational is True
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.int_shape(z_mean)[1]
        epsilon = tf.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def encode(self, x):
        for layer in self.__encoder:
            x = layer(x)
        if self.__variational:
            z_mean = self.__dense_mean
            z_log_var = self.__dense_log_var
            z = self.__sampling([z_mean, z_log_var])
            return z
        else:
            return x

    def decode(self, z):
        for layer in self.__decoder:
            z = layer(z)
        return self.__output(z)

    def call(self, inputs):
        z = Input(shape=self.__input_dim)(inputs)
        if self.__variational:
            z_mean = self.__dense_mean(x)
            z_log_var = self.__dense_log_var(x)
            return z_mean, z_log_var
        else:
            x = self.encode(inputs)
            z = self.decode(z)

        self.add_loss(reconstruction_loss)
        self.add_loss(kl_loss)

        return z

class ConvAutoencoder(Model):
    def __init__(self, input_dim, intermediate_dim_scaling_rate, input_to_waist_ratio,
                 add_max_pooling: bool = True, flat_waist: bool = False,
                 add_reconstruction_loss: bool = True, variational: bool = True):
        super(ConvAutoencoder, self).__init__()

        self.__input_dim = input_dim

        # higher to lower number, reverse this on the decoder side
        hidden_layer_1_dim = nearest_lower_exponent_of_2(self.__input_dim[0])
        print(hidden_layer_1_dim)
        self.__hidden_layer_dimensions = [nearest_lower_exponent_of_2(self.__input_dim[0])]
        self.__hidden_layer_dimensions.extend(
            [
                hidden_dim for hidden_dim in AutoEncoderHiddenLayerDimensionIterator(
                    current_dim=hidden_layer_1_dim, input_dim=self.__input_dim[0], 
                    scaling_rate=intermediate_dim_scaling_rate, input_to_waist_ratio=input_to_waist_ratio
                )
            ]
        )

        # print(self.__hidden_layer_dimensions)
        self.__latent_dim = self.__hidden_layer_dimensions[-1]

        self.__add_reconstruction_loss = add_reconstruction_loss
        self.__variational = variational
        self.__add_max_pooling = add_max_pooling
        self.__flat_waist = flat_waist


        # Encoder Network
        self.__encoder = [] 
        if self.__add_max_pooling: # Add max pooling
            for hidden_layer_dim in reversed(self.__hidden_layer_dimensions):
                self.__encoder.append(
                    Conv2D(hidden_layer_dim, (3,3), strides=2, activation='relu', padding="same")
                )
                self.__encoder.append(
                    MaxPooling2D((2, 2), padding="same")
                )
        else:
            self.__encoder = [
                Conv2D(hidden_layer_dim, (3,3), strides=2, activation='relu') for hidden_layer_dim in reversed(self.__hidden_layer_dimensions)
                ] 
        if self.__flat_waist:
            self.__encoder.extend([
                Flatten(), Dense(self.__latent_dim + self.__latent_dim)
            ])
        

        # Decoder Network
        self.__decoder = []
        if self.__flat_waist:
            self.__decoder = [
                InputLayer(input_shape=(self.__latent_dim,)), 
                Dense(units=7*7*32, activation="relu"),
                Reshape(target_shape=(7, 7, 32))
            ]
        self.__decoder.extend([
            Conv2DTranspose(hidden_layer_dim, (3,3), 2, activation='relu') for hidden_layer_dim in self.__hidden_layer_dimensions
            ])
        self.__output = Conv2D(1, (3, 3), activation='sigmoid', padding="same")

    def sampling_fn(self, args): # only used when variational is True
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.int_shape(z_mean)[1]
        epsilon = tf.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=z_mean.shape)
        return eps * tf.exp(z_log_var * .5) + z_mean

    def encode(self, x):
        for layer in self.__encoder:
            x = layer(x)
        if self.__variational: 
            z_mean, z_log_var = tf.split(x, num_or_size_splits=2, axis=1)
            return z_mean, z_log_var
        else:
            return x

    def decode(self, z):
        for layer in self.__decoder:
            z = layer(z)
        return self.__output(z)


    def call(self, inputs):
        # inputs = InputLayer(input_shape=self.__input_dim)(input)
        x = self.encode(inputs)
        return self.decode(x)


def main(args):
    input_dim = None

    # Grab resized photos
    X_resized_train, Y_resized_train, X_resized_test = [], [], []
    y_names = set()
    for file in os.listdir(POISON_PIC_LOCATION + RESIZED_POISONED_PIC_FILE_NAME):
        file_path = os.path.join(POISON_PIC_LOCATION + RESIZED_POISONED_PIC_FILE_NAME, file)
        if file_path.split(".")[-1] in ok_pic_formats:
            pic_name = file_path.split("/")[-1]
            pic_name = pic_name[:pic_name.find("nightshade")][:pic_name.find("resized")]
            y_names.add(pic_name)
            image = img.import_image(file_path)
            image = img.resize_image(image, args.new_width, args.new_height)
            Y_resized_train.append(image)
    
    for file in os.listdir(REGULAR_PIC_LOCATION + RESIZED_REGULAR_PIC_FILE_NAME):
        file_path = os.path.join(REGULAR_PIC_LOCATION + RESIZED_REGULAR_PIC_FILE_NAME, file)
        if file_path.split(".")[-1] in ok_pic_formats:
            pic_name = file_path.split("/")[-1]
            image = img.import_image(file_path)
            image = img.resize_image(image, args.new_width, args.new_height)
            input_dim = image.shape
            for y_name in y_names:
                if y_name in pic_name:
                    X_resized_train.append(image)
                    break
            else:
                X_resized_test.append(image)
    X_resized_train, Y_resized_train, X_resized_test = np.asarray(X_resized_train), np.asarray(Y_resized_train), np.asarray(X_resized_test)
    print(input_dim)
    
    # Grab padded photos
    # X_padded, Y_padded = [], []
    # for file in os.listdir(REGULAR_PIC_LOCATION + PADDED_REGULAR_PIC_FILE_NAME):
    #     file_path = os.path.join(REGULAR_PIC_LOCATION + PADDED_REGULAR_PIC_FILE_NAME, file)
    #     if file_path.split(".")[-1] in ok_pic_formats:
    #         pic_name = file_path.split("/")[-1]
    #         image = img.import_image(file_path)
    #         X_padded.append(X_padded)

    # for file in os.listdir(POISON_PIC_LOCATION + PADDED_POISONED_PIC_FILE_NAME):
    #     file_path = os.path.join(POISON_PIC_LOCATION + PADDED_POISONED_PIC_FILE_NAME, file)
    #     if file_path.split(".")[-1] in ok_pic_formats:
    #         pic_name = file_path.split("/")[-1]
    #         image = img.import_image(file_path)
    #         Y_padded.append(X_padded)

    exp_map = {
        # "ae_kl_only": {
        #     "add_reconstruction_loss": False,
        #     "variational": False
        # },
        # "vae_kl_only": {
        #     "add_reconstruction_loss": False, 
        #     "variational": True
        # },
        "ae_kl_and_rl": {
            "add_reconstruction_loss": True, 
            "variational": False
        },
        "vae_kl_and_rl": {
            "add_reconstruction_loss": True, 
            "variational": True
        }
    }
    
    
    for exp_type, params in exp_map.items():
        model = ConvAutoencoder(
                input_dim, intermediate_dim_scaling_rate=0.7, input_to_waist_ratio=4, 
                add_reconstruction_loss=params["add_reconstruction_loss"], variational=params["variational"]
            )
        if params["variational"]:
            optimizer = tf.keras.optimizers.Adam(1e-4)
            print(model.summary())
            for epoch in range(1, args.epochs + 1):
                for train_x in X_resized_train:
                    train_step(model, train_x, optimizer)

                loss = tf.keras.metrics.Mean()
                for test_x in X_resized_test:
                    loss(CVAE_loss(model, test_x))
                elbo = -loss.result()

        else:
            model.compile(
                optimizer='adam', loss=MeanSquaredError()
            )

            print(model.summary())
            model.fit(
                x=X_resized_train,
                y=Y_resized_train,
                epochs=args.epochs,
                validation_split=0.0
            )
        y_hat = model.predict(X_resized_test)

        save_path = HOMEMADE_POISON_PIC_LOCATION + RESIZED_POISONED_PIC_FILE_NAME
        for i, y in enumerate(y_hat):
            cv2.imwrite(save_path + exp_type + "/pic{}".format(i), y)


    # print(exp)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--new_height", type=int, help="height of pic to resize to", default=512)
    args.add_argument("-w", "--new_width", type=int, help="width of pic to resize to", default=512)

    args.add_argument("-e", "--epochs", type=int, help="# of epochs")
    main(args.parse_args())