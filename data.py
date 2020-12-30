__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import cv2
import numpy as np
from os.path import join as opj

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

def prepare_image(filename, size):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    return np.float32(hogdata).reshape(-1, 64)