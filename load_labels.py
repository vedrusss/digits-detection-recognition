__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import struct
import numpy as np
from PIL import Image
import argparse
import cv2
from os.path import join as opj

def load_data(data_lst, data_root, verbose=False):
    images = []
    labels = []
    lines = open(data_lst).read().splitlines()
    for line in lines:
        path, label = line.split()
        images.append(opj(data_root, path))
        labels.append(label)
    if verbose:
        print("Data stats:")
        for l in sorted(list(set(labels))):
            amount = len([el for el in labels if el == l])
            print(f"{l} : {amount}")
    return images, labels

#def resize_keep_aspect_ratio(image, size):

def encode_labels(labels):
    encoded = []
    mapping = {}
    existing = []
    for l in labels:
        if not l in existing:
            existing.append(l)
        ind = existing.index(l)
        encoded.append(ind)
        if not ind in mapping:
            mapping[ind] = l
    return encoded, mapping

def prepare_image(filename, size=(28, 28)):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, size)
    return image

def get_labels(file):
    magic, num = struct.unpack(">II", file.read(8))
    if magic != 2049:
        raise ValueError('Magic number mismatch, expected 2049,' +
                         ' got %d' % magic)

    return np.fromfile(file, dtype=np.int8), num


def get_images(file):
    magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
    if magic != 2051:
        raise ValueError('Magic number mismatch, expected 2051,' +
                         ' got %d' % magic)
    images = np.fromfile(file, dtype=np.uint8).reshape(num, rows * cols)
    return images, num, rows, cols


def get_data(label_filename, image_filename):
    with open(label_filename, 'rb') as label_file:
        labels, num_labels = get_labels(label_file)

    with open(image_filename, 'rb') as image_file:
        images, num_images, rows, cols = get_images(image_file)

    if num_labels != num_images:
        print('[WARNING]: Number of images and labels mismatch')

    return images, labels, num_labels, rows, cols

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("label_file", type=str)
    parser.add_argument("image_file", type=str)

    args = parser.parse_args()

    images, labels, num, rows, cols = get_data(args.label_file,
                                               args.image_file)
    print('First:', labels[0])
    Image.fromarray(images[0].reshape(rows, cols)).show()
    print('Last:', labels[-1])
    Image.fromarray(images[-1].reshape(rows, cols)).show()
    print('Length', len(labels))
