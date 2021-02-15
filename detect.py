__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import argparse
import os
import cv2
import numpy as np
from utilities import get_base_name, show_results_break
from PIL import Image, ImageDraw
import dlib

class Detector:
    def __init__(self, detector_model_files, opencv=True, resize_factor=1.):
        self._resize_factor = resize_factor
        self._opencv = opencv
        if opencv:
            self._detector = cv2.CascadeClassifier(detector_model_files[0])
        else:
            self._simple_detectors = []
            for filename in detector_model_files:
                dlib_detector = dlib.simple_object_detector(filename)
                self._simple_detectors.append(dlib_detector)
            self._multidetector = dlib.simple_object_detector(self._simple_detectors)

    def __call__(self, np_image):
        if self._opencv:
            if len(np_image.shape) > 3 and np_image.shape[2] > 1:
                np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
            digits = self._detector.detectMultiScale(np_image, minSize=(6, 6), maxSize=(48, 48))
        else:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
            if self._resize_factor != 1.:
                np_image = cv2.resize(np_image, dsize=None, fx=self._resize_factor, fy=self._resize_factor)
            
            #for d in self._simple_detectors:
            #    r = d(np_image)
            #    print('s', r)
            boxes = self._multidetector(np_image)
            #print("Multi result:", boxes)
        
            #dlib_boxes, dlib_scores, dlib_inds = self._multidetector.run_multiple(self._simple_detectors, np_image)
            #print("Run multiple", dlib_boxes, dlib_scores, dlib_inds)
        
            digits = [(b.left(), b.top(), b.right(), b.bottom()) for b in boxes]

            digits = [ (int(b[0]/self._resize_factor), 
                        int(b[1]/self._resize_factor),
                        int(b[2]/self._resize_factor),
                        int(b[3]/self._resize_factor)) for b in digits]

        return digits


def detect(im, xml):
    digit_cascade = cv2.CascadeClassifier(xml)
    digits = digit_cascade.detectMultiScale(im)
    return digits

def annotate_detection(im, regions, color=128):
    clone = im.copy()
    draw = ImageDraw.Draw(clone)
    for (x, y, w, h) in regions:
        draw.rectangle((x, y, x+w, y+h), outline=color)
    return clone


def crop_detection(im, regions):
    return [im.crop((x, y, x+w, y+h)) for (x, y, w, h) in regions]

def parse_filename(filename):
    name = os.path.split(filename)[-1].split('.')[0]
    if 'score' in filename or 'shotclock' in filename:
        return name.split('_')[0]
    if 'colon' in name or 'dot' in name:
        name = name.split('_')[0]
        name = name.replace('colon',':').replace('dot','.')
    else:
        name = ' '.join(name.split('_')[:2])
    return name

def parse_args():
    parser = argparse.ArgumentParser("Digits detector")
    parser.add_argument('-i', '--images', type=str, nargs='+', help='Images to test/train on')
    parser.add_argument('-m', '--model',  type=str, nargs='+', required=True, help='Model path(-s) to load/save')
    parser.add_argument('-d', '--display', action='store_true', help='Display test results')
    parser.add_argument('-is', '--ignore_signs', action='store_true', help="Ignore colons and dots")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    detector = Detector(args.model, opencv=False, resize_factor=3.) #2.1)
    TPs, FPs = 0, 0
    for image_fn in args.images:
        name = parse_filename(image_fn)
        if args.ignore_signs: name = name.replace(':','').replace('.','').replace(' ','')
        objects_amount = 1 # len(name)
        np_image = cv2.imread(image_fn)
        assert(np_image is not None),f"Cannot read image {image_fn}"
        image = np.zeros_like(np_image)
        image = cv2.resize(image, dsize=(24,48)) #, fx=1.3, fy=1.3)
        image[2:np_image.shape[0]+2,1:np_image.shape[1]+1,:] = np_image
        digit_regions = detector(image)
        if len(digit_regions) == objects_amount: TPs += 1
        else: FPs += 1
        print(f"{name} --> {len(digit_regions)}")
        if args.display:
            if show_results_break(np_image, digit_regions): break
    TPrate = TPs / float(TPs + FPs) if TPs+FPs>0 else None
    print(f"Results: TP rate {TPrate}, TPs {TPs}, FP {FPs}")    
