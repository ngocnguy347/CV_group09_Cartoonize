import os
import io
import sys
import yaml
with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)
sys.path.insert(0, './white_box_cartoonizer/')
import cv2
from PIL import Image
import numpy as np
from cartoonize import WB_Cartoonize

from PIL import Image
import glob

wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

def convert_bytes_to_image(img_bytes):
    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode=="RGBA":
        image = Image.new("RGB", pil_image.size, (255,255,255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    image = np.array(image)
    return image

with open("aaa.jpeg", "rb") as image: # đổi input ảnh ở đây
    f = image.read()
    img = bytearray(f)
    image = convert_bytes_to_image(img)
    cartoon_image = wb_cartoonizer.infer(image)            
    cv2.imwrite("cartoonized_img_name2.jpg", cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR)) # đổi output ảnh ở đây
                    
               