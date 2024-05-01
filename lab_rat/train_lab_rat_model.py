from X.image_resizing import *
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import argparse
import cv2


load_dotenv()

ok_pic_formats = {"jpg", "jpeg", "webp", "png"}
PIC_LOCATION = os.getenv("REGULAR_PIC_LOCATION")
MODEL_TRAIN_PATH = os.getenv("MODEL_TRAIN_PATH")


# TODO: What is dreambooth's image sizing layer?
# TODO: -> whatever it is, grab the images & 1) store their changed sizes, 2) store padded ones that retain original ratios and try training on both

def main(args):
    # Resize images first
    for file in os.listdir(PIC_LOCATION):
        file_path = os.path.join(PIC_LOCATION, file)
        if file_path.split(".")[-1] in ok_pic_formats:
            pic_name = file_path.split("/")[-1]
            image = import_image(file_path)

            # Resized
            save_path_resized = file_path[:-len(pic_name)] + args.resized_file_name + pic_name
            cv2.imwrite(save_path_resized, resize_image(image, args.new_width, args.new_height))

            # Padded
            save_path_padded = file_path[:-len(pic_name)] + args.padded_file_name + pic_name
            cv2.imwrite(save_path_padded, resize_and_pad_image(image, args.new_width, args.new_height))
    
    # Train model(s)
    os.popen(
        'accelerate launch {} \'
            '--with_prior_preservation \ '
            '--prior_loss_weight \ '
            '--class_data_dir={}\ '
            '--class_prompt=text prompt describing class\ '
            ''.format(MODEL_TRAIN_PATH)
    )
    
    # Resized
            
    # Padded


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-w", "--new_width", type=int, help="")
    args.add_argument("-h", "--new_height", type=int, help="")

    args.add_argument("-r", "--resized_file_name", type=str, help="name of file that stores resized images")
    args.add_argument("-p", "--padded_file_name", type=str, help="name of file that stores padded images")

    
    
    main(args.parse_args())

