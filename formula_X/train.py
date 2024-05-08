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
# PADDED_REGULAR_PIC_FILE_NAME = os.getenv("PADDED_REGULAR_PIC_FILE_NAME")

RESIZED_POISONED_PIC_FILE_NAME = os.getenv("RESIZED_POISONED_PIC_FILE_NAME")
# PADDED_POISONED_PIC_FILE_NAME = os.getenv("PADDED_POISONED_PIC_FILE_NAME")

MODEL_PATH = os.getenv("MODEL_PATH") # Save'em models

def nearest_lower_exponent_of_2(n):
    if n < 1:
        return 0
    power = 1
    while power <= n:
        power <<= 1  # -> power = power * 2
    return power >> 1  # Shift right to get the nearest lower power of 2 | binary byte shifting man


def reconstruction_loss(y_true, y_pred):
    # Choose an appropriate loss based on your data:
    return tf.reduce_mean(tf.square(y_true - y_pred))  # Example: Mean Squared Error

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
    def __init__(self, input_dim, h1_dim, intermediate_dim_scaling_rate, input_to_waist_ratio, variational: bool = True):
        super(Autoencoder, self).__init__()

        self.__input_dim = input_dim

        # higher to lower number, reverse this on the decoder side
        self.__h1_dim = nearest_lower_exponent_of_2(h1_dim)
        self.__hidden_layer_dimensions = [self.__h1_dim]
        self.__hidden_layer_dimensions.extend(
            [
                hidden_dim for hidden_dim in AutoEncoderHiddenLayerDimensionIterator(
                    current_dim=self.__h1_dim, input_dim=self.__input_dim[0], 
                    scaling_rate=intermediate_dim_scaling_rate, input_to_waist_ratio=input_to_waist_ratio
                )
            ]
        )

        print(self.__hidden_layer_dimensions)
        self.__latent_dim = self.__hidden_layer_dimensions[-1]

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

### This actually ends up denoising the pictures to remove the poison
class ConvAutoencoderGenerator():
    def __init__(self, input_dim, h1_dim, intermediate_dim_scaling_rate, input_to_waist_ratio,
                 activation: str,
                 add_max_pooling: bool = True, flat_waist: bool = False, variational: bool = True):
        self.__input_dim = input_dim

        # higher to lower number, reverse this on the decoder side
        h1_dim = nearest_lower_exponent_of_2(h1_dim)
        self.__hidden_layer_dimensions = [h1_dim]
        self.__hidden_layer_dimensions.extend(
            [
                hidden_dim for hidden_dim in AutoEncoderHiddenLayerDimensionIterator(
                    current_dim=h1_dim, input_dim=self.__input_dim[0], 
                    scaling_rate=intermediate_dim_scaling_rate, input_to_waist_ratio=input_to_waist_ratio
                )
            ]
        )

        # print(self.__hidden_layer_dimensions)
        self.__latent_dim = self.__hidden_layer_dimensions[-1]

        self.__activation = activation

        self.__variational = variational
        self.__add_max_pooling = add_max_pooling
        self.__flat_waist = flat_waist

    def sampling_fn(self, z_mean, z_log_var): # only used when variational is True
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.int_shape(z_mean)[1]
        epsilon = tf.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=z_mean.shape)
        return eps * tf.exp(z_log_var * .5) + z_mean

    def encode(self, x):
        if self.__add_max_pooling: # Add max pooling
            for hidden_layer_dim in self.__hidden_layer_dimensions:
                x = Conv2D(hidden_layer_dim, (3,3), activation=self.__activation, padding="same")(x)
                x = MaxPooling2D((2, 2), padding="same")(x)
        else:
            for hidden_layer_dim in self.__hidden_layer_dimensions:
                x = Conv2D(hidden_layer_dim, (3,3), activation=self.__activation, padding="same")(x) 

        if self.__flat_waist:
            x = Flatten()(x)
            # print(self.__hidden_layer_dimensions, self.__latent_dim)
            # x = Dense(self.__latent_dim *self.__latent_dim,  activation=self.__activation)(x)

        # print("###### Encoder Summary")
        # print(x.summary())
        if self.__variational: 
            z_mean, z_log_var = tf.split(x, num_or_size_splits=2, axis=1)
            return z_mean, z_log_var
        else:
            return x

    def decode(self, z):
        
        if self.__flat_waist:
            # z = InputLayer(input_shape=(self.__latent_dim,))(z)
            # z = Dense(self.__latent_dim *self.__latent_dim, activation=self.__activation)(z)
            # z = Dense(4*self.__latent_dim*4*self.__latent_dim*self.__latent_dim, activation=self.__activation)(z)
            if self.__add_max_pooling:
                factorization_count = len(self.__hidden_layer_dimensions)//2
                reshape_size = self.__input_dim[0]//np.power(factorization_count, 2)
            else:
                # factorization_count = len(self.__hidden_layer_dimensions)
                reshape_size = self.__input_dim[0]
            # print(self.__input_dim, factorization_count)
            
            print(reshape_size, self.__latent_dim, self.__hidden_layer_dimensions)
            z = Reshape(target_shape=(reshape_size, reshape_size, self.__latent_dim))(z)
            
        for hidden_layer_dim in reversed(self.__hidden_layer_dimensions):
            if self.__add_max_pooling:
                strides = 2
            else:
                strides=1
            z = Conv2DTranspose(hidden_layer_dim, (3,3), strides, activation=self.__activation, padding='same') (z)
        return z


    def call(self, inputs):
        # inputs = InputLayer(input_shape=self.__input_dim)(inputs)
        if self.__variational:
            x_mean, x_log_var = self.encode(inputs)
            x = self.reparameterize(x_mean, x_log_var)
        else:
            x = self.encode(inputs)
        y = self.decode(x)
        y = Conv2D(self.__input_dim[-1], (3, 3), activation='sigmoid', padding="same")(y) # Output layer
        return Model(inputs, y)


# TODO: variational stuff is not tested. - tbh might not need to yet
# this will prob eventually be some sort of a template code that can be reused - so parametrizing things well is pretty important
def main(args):
    input_dim = None

    # Grab resized photos
    regular_pic_train, poisoned_pics_train, regular_pics_test = [], [], []
    y_names = set()
    for file in os.listdir(POISON_PIC_LOCATION + RESIZED_POISONED_PIC_FILE_NAME):
        file_path = os.path.join(POISON_PIC_LOCATION + RESIZED_POISONED_PIC_FILE_NAME, file)
        if file_path.split(".")[-1] in ok_pic_formats:
            pic_name = file_path.split("/")[-1]
            pic_name = pic_name[:pic_name.find("nightshade")][:pic_name.find("resized")]
            if pic_name[-1] == "-":
                pic_name = pic_name[:-1]
            y_names.add(pic_name)
            image = img.import_image(file_path)
            image = img.resize_image(image, args.new_width, args.new_height)
            poisoned_pics_train.append(image)
    
    # x_train_pics = set()
    for file in os.listdir(REGULAR_PIC_LOCATION + RESIZED_REGULAR_PIC_FILE_NAME):
        file_path = os.path.join(REGULAR_PIC_LOCATION + RESIZED_REGULAR_PIC_FILE_NAME, file)
        if file_path.split(".")[-1] in ok_pic_formats:
            pic_name = file_path.split("/")[-1]
            image = img.import_image(file_path)
            image = img.resize_image(image, args.new_width, args.new_height)
            input_dim = image.shape
            for y_name in y_names:
                if y_name in pic_name:
                    # x_train_pics.add(pic_name)
                    regular_pic_train.append(image)
                    break
            else:
                regular_pics_test.append(image)
    
    del y_names
    # Normalize Data
    regular_pic_train, poisoned_pics_train, regular_pics_test = np.asarray(regular_pic_train), np.asarray(poisoned_pics_train), np.asarray(regular_pics_test)
    regular_pic_train = regular_pic_train/255
    poisoned_pics_train = poisoned_pics_train/255
    regular_pics_test = regular_pics_test/255
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

    
    # A Model is fully parametrized by - epochs, scaling rate, input to waist ratio, hidden_dim_1
    
    folder_name = HOMEMADE_POISON_PIC_LOCATION + RESIZED_POISONED_PIC_FILE_NAME + args.model_type + "/{}"

    model = ConvAutoencoderGenerator(
                input_dim, args.hidden_dim_1,
                intermediate_dim_scaling_rate=args.intermediate_dim_scaling_rate, 
                input_to_waist_ratio=args.input_to_waist_ratio, 
                add_max_pooling=args.add_max_pooling,
                flat_waist=args.flat_waist,
                variational=args.variational,
                activation=args.activation
        )
    
    model = model.call(Input(shape=input_dim))
    if args.variational:
        pass
        # optimizer = tf.keras.optimizers.Adam(1e-4)
        # print(model.summary())
        # for epoch in range(1, args.epochs + 1):
        #     for train_x in X_resized_train:
        #         train_step(model, train_x, optimizer)

        #     loss = tf.keras.metrics.Mean()
        #     for test_x in X_resized_test:
        #         loss(CVAE_loss(model, test_x))
        #     elbo = -loss.result()

    else:
        model.compile(
            optimizer='adam', loss=MeanSquaredError()
        )

        print(model.summary())
        for epoch_count in range(args.steps, args.epochs+1, args.steps):
            # print(epoch_count, x.shape, y.shape)
            if args.model_type == "antidote":
                model.fit(
                    x=poisoned_pics_train,
                    y=regular_pic_train,
                    initial_epoch=epoch_count - args.steps + 1,
                    epochs=epoch_count,
                    validation_split=0.2
                )
                y_hat = model.predict(poisoned_pics_train)*255
                y_hat = y_hat.astype(np.uint8)
            elif args.model_type == "poison":
                model.fit(
                    x=regular_pic_train,
                    y=poisoned_pics_train,
                    initial_epoch=epoch_count - args.steps + 1,
                    epochs=epoch_count,
                    validation_split=0.2
                )
                y_hat = model.predict(regular_pics_test)*255
                y_hat = y_hat.astype(np.uint8)
            else:
                raise ValueError("model type {} is nooot supported".format(args.model_type))


            model_name = "AE_activation={}_h1dim={}_epochs={}_intermediate_dim_scaling_rate={}_input_to_waist_ratio={}_add_max_pooling={}_flat_waist={}".format(
                    args.activation, args.hidden_dim_1, epoch_count, args.intermediate_dim_scaling_rate, args.input_to_waist_ratio, args.add_max_pooling, args.flat_waist
                    )
            
            model.save(MODEL_PATH + "/{}/{}.keras".format(args.model_type, model_name))

            # create directory if it doesn't exist (actually just overwriiiite)
            # Create the directory using os.mkdir()

            
            try:
                os.mkdir(folder_name.format(model_name))
                print(f"Directory '{folder_name.format(model_name)}' created successfully!")
            except FileExistsError:
                print(f"Directory '{folder_name.format(model_name)}' already exists.")
            except OSError as error:
                print(f"Error creating directory: {error}")

            for i, y_ in enumerate(y_hat):
                cv2.imwrite(folder_name.format(model_name) + "/pic{}.png".format(i), y_
                )


    # print(exp)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    model_config = args.add_argument_group("model_config")

    model_config.add_argument("-i", "--new_height", type=int, help="height of pic to resize to", default=512)
    model_config.add_argument("-w", "--new_width", type=int, help="width of pic to resize to", default=512)
    model_config.add_argument("-t", "--model_type", type=str, help="antidote or poison", default="antidote")

    hyperparams = args.add_argument_group("hyperparams")

    hyperparams.add_argument("-e", "--epochs", type=int, help="# of epochs", default=350)
    hyperparams.add_argument("-s", "--steps", type=int, help="# of epochs to run each before saving intermediate results", default=50)
    hyperparams.add_argument("-d", "--hidden_dim_1", type=int, help="", default=128)

    hyperparams.add_argument("-a", "--activation", type=str, help="linear or relu", default="linear")
    hyperparams.add_argument("-scale", "--intermediate_dim_scaling_rate", type=float, help="rate at which to scale down dim size", default=0.5)
    hyperparams.add_argument("-waist", "--input_to_waist_ratio", type=float, help="input/waist size", default=16)
    hyperparams.add_argument("-m", "--add_max_pooling", action="store_true", help="should u add max pooling on encoder?")
    hyperparams.add_argument("-f", "--flat_waist", action="store_true", help="should the waist be flattened?")
    hyperparams.add_argument("-v", "--variational", action="store_true", help="make it variational or nah?")

    main(args.parse_args())

