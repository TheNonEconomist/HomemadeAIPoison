# from PIL import Image
import cv2
# import imagehash
from collections import defaultdict
import os
from PIL import Image

# PILLOW
# def load_image(file_path):
#     with Image.open(file_path) as img:
#         return np.asarray(img)

# def preprocess_image(img, target_size=(224, 224)):
#     img = img.resize(target_size)
#     img_array = np.array(img)
    
#     # Normalize the image array if required:
#     # img_array = img_array / 255.0

#     return img_array


# OpenCV
def import_image(input_path):
    image = cv2.imread(input_path)
    if image is None:
        print("Error: Image not found.")
        return
    else:
        return image

def resize_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def pad_image(image, new_width, new_height, pad_color=(0, 0, 0), borderType=cv2.BORDER_CONSTANT):
    """
    Pads an image to a specified width and height with a constant border value.

    Args:
        image: The image to be padded (numpy array).
        new_width: The desired width of the padded image.
        new_height: The desired height of the padded image.
        pad_color: The value to fill the padded area (tuple of 3 integers for BGR). Defaults to black (0, 0, 0).
        borderType: OpenCV border type for padding. Defaults to cv2.BORDER_CONSTANT.

    Returns:
        A padded image (numpy array).
    """

    height, width, channels = image.shape

    # Calculate the padding required
    top_pad = (new_height - height) // 2
    bottom_pad = new_height - height - top_pad
    left_pad = (new_width - width) // 2
    right_pad = new_width - width - left_pad

    return cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, borderType=borderType, value=pad_color)

def resize_and_pad_image(image, new_width, new_height, pad_color=[0, 0, 0]):
    """
    Some pictures might be exceed, so adjust them and then pad if we want to maintain the width to height ratio of the images
    """
    original_height, original_width = image.shape[:2]
    ratio_width = new_width / original_width
    ratio_height = new_height / original_height
    ratio = min(ratio_width, ratio_height)

     
    # compute dimensions that maintain the aspect ratio
    control_width, control_height = (int(original_width * ratio), int(original_height * ratio))
    resized_image = resize_image(image, control_width, control_height)

    return pad_image(resized_image, new_width, new_height)

def export_image(image, output_path):
    cv2.imwrite(output_path, image)


# def find_duplicates(images_directory): # grab hash of each image and store their path
#     hashes = defaultdict(list)
#     # Loop through image files
#     for image_filename in os.listdir(images_directory):
#         if image_filename.endswith(("jpg", "jpeg", "webp", "png")):
#             image_path = os.path.join(images_directory, image_filename)
#             # Open and convert image to grayscale
#             # image = import_image(image_path)
#             image = Image.open(image_path)

#             # Use pHash
#             h = imagehash.phash(image)

#             # Append image path to hash entry
#             hashes[h].append(image_path)

#     # Identify duplicates (entries with more than one path)
#     duplicates = {hash_val: paths for hash_val, paths in hashes.items() if len(paths) > 1}
#     return duplicates

def handle_duplicates(duplicates):
    for hash_val, paths in duplicates.items():
        print(f"Duplicate images for hash {hash_val}:")
        # Keep the first image, remove others
        for path in paths[1:]:
            print(f"Removing {path}")
            os.remove(path)  # Uncomment this to actually delete the files