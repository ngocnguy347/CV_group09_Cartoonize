import os
import io
import uuid
import sys
import yaml
import traceback
import argparse
from PIL import Image

with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

sys.path.insert(0, './white_box_cartoonizer/')

import cv2
from flask import Flask, render_template, make_response, flash
import flask
from PIL import Image
import numpy as np
import skvideo.io

from cartoonize import WB_Cartoonize


## Init Cartoonizer and load its weights 
wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

# def convert_bytes_to_image(img_bytes):
#     """Convert bytes to numpy array
#     Args:
#         img_bytes (bytes): Image bytes read from flask.
#     Returns:
#         [numpy array]: Image numpy array
#     """
    
#     pil_image = Image.open(io.BytesIO(img_bytes))
#     if pil_image.mode=="RGBA":
#         image = Image.new("RGB", pil_image.size, (255,255,255))
#         image.paste(pil_image, mask=pil_image.split()[3])
#     else:
#         image = pil_image.convert('RGB')
    
#     image = np.array(image)
    
#     return image

def prepare_image(image_path):
    pil_image =  Image.open(image_path)
    pil_image.load()

    if pil_image.mode=="RGBA":
        image = Image.new("RGB", pil_image.size, (255,255,255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    
    image = np.array(image)
    
    return image

def cartoonize(image_folder, cartoon_folder):

    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)
        # img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        image = prepare_image(file_path)
        cartoon_image = wb_cartoonizer.infer(image)
        cv2.imwrite(os.path.join(cartoon_folder, file_name),cartoon_image)



if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        parser.add_argument('-i', '--image_folder', type=str,
                            help='directory of input images')

        parser.add_argument('-o', '--cartoon_folder', type=str,
                            help='directory of output cartoon images')
       
        args = parser.parse_args()
        cartoonize(args.image_folder, args.cartoon_folder)


