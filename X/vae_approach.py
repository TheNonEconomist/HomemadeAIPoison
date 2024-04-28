import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from keras import backend as K
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

class AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim_scaling_rate, input_to_waist_ratio, add_reconstruction_loss: bool = True, variational: bool = True):
        super(AutoEncoder, self).__init__()
        self.__latent_dim = nearest_lower_exponent_of_2(input_dim[0]) # size of layer right after the input layer

        # higher to lower number, reverse this on the decoder side
        self.__hidden_layer_dimensions = [self.__latent_dim]
        self.__hidden_layer_dimensions.extend(
            [
                hidden_dim for hidden_dim in AutoEncoderHiddenLayerDimensionIterator(
                    current_dim=self.__latent_dim, input_dim=input_dim[0], 
                    scaling_rate=latent_dim_scaling_rate, input_to_waist_ratio=input_to_waist_ratio
                )
            ]
        )

        self.__add_reconstruction_loss = add_reconstruction_loss
        self.__variational = variational

        # Encoder
        self.__input = Input(shape=input_dim) #Input Layer
        self.__encoder = [
            Dense(hidden_layer_dim, activation='relu') for hidden_layer_dim in self.__hidden_layer_dimensions
            ] 

        # Decoder
        self.__decoder = [
            Dense(hidden_layer_dim, activation='relu') for hidden_layer_dim in reversed(self.__hidden_layer_dimensions)
            ]
        self.__output = Dense(input_dim, activation='sigmoid')


    def call(self, inputs, training=False):
        # Encode
        z = self.__input(inputs)
        for encode_layer in self.__encoder:
            z = encode_layer(z)

        # Add Random Noise to make it Variational & Set Decoder
        if self.__variational:
            mean, log_var = tf.split(z, num_or_size_splits=2, axis=1)
            epsilon = tf.random.normal(shape=(tf.shape(mean)[0], self.__next_dim))
            z = mean + tf.exp(log_var * 0.5) * epsilon

        # Decode
        for decode_layer in self.__decoder:
            z = decode_layer(z)
        z = self.__output(z)
        
            
        
        # Loss Function
        kl_loss = -0.5 * tf.reduce_mean(log_var - tf.square(mean) - tf.exp(log_var) + 1)
        if self.__add_reconstruction_loss:
            reconstruction_loss = tf.mean(tf.square(z - x))
            self.add_loss(kl_loss + reconstruction_loss)
        else:
            self.add_loss(kl_loss)
        
        return z



def main(args):
    input_dim = None
    # Grab resized photos
    X_resized_train, Y_resized_train, X_resized_test = [], [], []
    y_names = set()
    for file in os.listdir(POISON_PIC_LOCATION + RESIZED_POISONED_PIC_FILE_NAME):
        file_path = os.path.join(POISON_PIC_LOCATION + RESIZED_POISONED_PIC_FILE_NAME, file)
        if file_path.split(".")[-1] in ok_pic_formats:
            pic_name = file_path.split("/")[-1]
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
            if pic_name in y_names:
                X_resized_train.append(image)
            else:
                X_resized_test.append(image)
    
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
        "ae_kl_only": {
            "add_reconstruction_loss": False,
            "variational": False
        },
        "vae_kl_only": {
            "add_reconstruction_loss": False, 
            "variational": True
        },
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
        model = AutoEncoder(
            input_dim, latent_dim_scaling_rate=0.7, input_to_waist_ratio=4, 
            add_reconstruction_loss=params["add_reconstruction_loss"], variational=params["variational"]
        )
        model.compile(optimizer='adam')
        model.fit(
            x=X_resized_train,
            y=Y_resized_train,
            epochs=5,
            verbose="auto",
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
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
    main(args.parse_args())