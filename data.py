__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import cv2
from math import ceil
import numpy as np
from os.path import join as opj

def encode_labels(labels):
    unique_labels = set(labels)
    mapping_from_labels = {}
    mapping_to_labels = {}
    for ind, label in enumerate(unique_labels):
        mapping_from_labels[label] = ind
        mapping_to_labels[ind] = label
    encoded = [mapping_from_labels[label] for label in labels]
    return encoded, mapping_from_labels, mapping_to_labels

def label_to_binary(label, vector_size):
    encoded = [0] * vector_size
    encoded[label] = 1
    return encoded

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

def prepare_image(filename, size, keep_aspect_ratio):
    image = cv2.imread(filename)
    if image is None:
        raise Exception(f'Cannot read image {filename}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if keep_aspect_ratio:
        image = resize_keep_aspect_ratio(image, size)
    else:
        image = cv2.resize(image, size)
    return image


def preprocess_for_opencv(images, deskew_size=28, hist_bin_n=16):
    def deskew(img):
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*deskew_size*skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (deskew_size, deskew_size), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img

    def hog(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        # quantizing binvalues in (0...16)
        bins = np.int32(hist_bin_n * ang / (2 * np.pi))
        bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), hist_bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)     # hist is a 64 bit vector
        return hist
    
    hogdata = [hog(deskew(im)) for im in images]
    print(np.float32(hogdata).shape, np.float32(hogdata).reshape(-1, hist_bin_n * 4).shape)
    return np.float32(hogdata).reshape(-1, hist_bin_n * 4)

def resize_keep_aspect_ratio(image, size):
    input_w, input_h = image.shape[1], image.shape[0]
    target_w, target_h = size[0], size[1]
    h = ceil(target_w * input_h / float(input_w))
    w = ceil(target_h * input_w / float(input_h))
    if h <= target_h: w = target_w
    else: h = target_h
    image = cv2.resize(image, (int(w), int(h)))

    delta_w = target_w - w
    delta_h = target_h - h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    #top = 0
    #left = 0
    #bottom = int(target_h - h);
    #right = int(target_w - w);
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(1, 1, 1))
    return image
