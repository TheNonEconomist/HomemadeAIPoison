
import os
from image_resizing import *
from dotenv import load_dotenv
import argparse
import json

load_dotenv()

OK_PIC_FORMATS = {"jpg", "jpeg", "webp", "png"}
REGULAR_PIC_LOCATION = os.getenv("REGULAR_PIC_LOCATION")



def main(args):
    model = import_model(args.model_path)  ### TODO: Pseudocode

    pics, image_tagline = {}, {}
    for file in os.listdir(REGULAR_PIC_LOCATION):
        file_path = os.path.join(REGULAR_PIC_LOCATION, file)
        if file_path.split(".")[-1] in OK_PIC_FORMATS:
            pic_name = file_path.split("/")[-1]
            image = import_image(file_path)
            image = resize_image(image, args.new_width, args.new_height)
            # pics[pic_name] = image # Add pic to the dict
            image_tagline[pic_name] = [model.run(image)] # run our image captioning model here  #### TODO: PSEUDOCODE
            # TODO: figure out how to embed perhaps different permutation of neural nets here???

    # Save Tagline as a file
    with open(args.tagline_path, "w") as tagline_file:
        json.dump(image_tagline, tagline_file) 

def resize_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

def pad_image(image, new_width, new_height):
    return # TODO:
            


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("-m", "--model_path", type=str, help="Path to where the model is")
    args.add_argument("-t", "--tagline_path", type=str, help="Path to where to save the taglines")

    args.add_argument("-h", "--height", type=int, help="Height for images to be fed into the image-captioning network")
    args.add_argument("-w", "--width", type=int, help="Width for images to be fed into the image-captioning network")

    main(args.parse_args())
