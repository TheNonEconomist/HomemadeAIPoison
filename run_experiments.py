import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import argparse
import cv2

####### 
load_dotenv()

ACCEPTED_PIC_FORMATS = {"jpg", "jpeg", "webp", "png"}
POISONED_PIC_LOCATION = os.getenv("NIGHTSHADE_PIC_LOCATION")
REGULAR_PIC_LOCATION = os.getenv("REGULAR_PIC_LOCATION")
########





def main(args):
    pass



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-h", "--height", help="image height", type=int)
    args.add_argument("-w", "--width", help="image width", type=int)

    main(args.parse_args())
