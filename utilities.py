__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import cv2
from PIL import ImageDraw


get_base_name = lambda fn: '.'.join(fn.split('/')[-1].split('.')[:-1]) 

def show_results_break(np_image, regions):
    for (x, y, w, h) in regions:
        r = x + w
        b = y + h
        cv2.rectangle(np_image, (x,y), (r,b), (0,255,0), 2)
    cv2.imshow('test', np_image)
    return (cv2.waitKey() == 27)


TEST_FONT = '5'

def get_font_size(font):
    return max(font.getsize(TEST_FONT))

def annotate_recognition(im, regions, labels, font, color=255):
    clone = im.copy()
    draw = ImageDraw.Draw(clone)
    size = get_font_size(font)
    for idx, (x, y, w, h) in enumerate(regions):
        draw.text(
            (x+w-size, y+h-size), str(labels[idx]), font=font, fill=color)
    return clone