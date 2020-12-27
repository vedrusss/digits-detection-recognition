__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import cv2
import numpy as np
from sklearn import svm
import pickle

svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR, svm_type=cv2.ml.SVM_C_SVC)
SAMPLE_SIZE = (28, 28)
SZ = 28
bin_n = 16  # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


class Recognizer:
    def __init__(self, fw_type, model_path=None, train_data=None):
        self._recognizer = self._init_cv(train_data) if fw_type == 'opencv' else \
                           self._init_sk(model_path, train_data)
        self.__call__ = self._predict_cv if fw_type == 'opencv' else self._predict_sk

    def _init_cv(self, train_data):
        svc = cv2.ml.SVM_create()
        svc.setType(svm_params['svm_type'])
        svc.setKernel(svm_params['kernel_type'])
        rows = SAMPLE_SIZE[0]
        cols = SAMPLE_SIZE[1]
        traindata = preprocess(train_data['images'], rows, cols)
        #responses = np.float32(labels[:, None])
        responses = np.array(train_data['labels'], dtype=np.int)# np.int(labels[:, None]) #[:, None])
        svc.train(traindata, cv2.ml.ROW_SAMPLE, responses)
        return svc

    def _init_sk(self, model_path, train_data):
        if train_data:
            print('Training sklearn SVM digits recognizer...')
            svc = svm.SVC(kernel='linear')
            svc.fit(train_data['images'], train_data['labels'])
            with open(model_path, 'wb') as f:
                f.write(pickle.dumps(svc))
        else:
            print("Loading sklearn SVM digits recognizer")
            with open(model_path, 'rb') as f:
                svc = pickle.loads(f.read())
        return svc

    def _predict_cv(self, imgs):
        test = [np.float32(i.resize(SAMPLE_SIZE)).ravel() for i in imgs]
        rows = SAMPLE_SIZE[0]
        cols = SAMPLE_SIZE[1]
        testdata = preprocess(test, rows, cols).reshape(-1, bin_n * 4)
        res = self._recognizer.predict(testdata)[1]
        labels = res.astype(np.uint8).ravel()
        return labels

    def _predict_sk(self, imgs):
        test = [np.float32(i.resize(SAMPLE_SIZE)).ravel() for i in imgs]
        labels = self._recognizer.predict(test)
        return labels




from PIL import ImageDraw
from load_labels import get_data



TEST_FONT = '5'


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n * ang / (2 * np.pi))
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n)
             for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

def preprocess(images, rows, cols):
    deskewed = [deskew(im.reshape(rows, cols)) for im in images]
    hogdata = [hog(im) for im in deskewed]
    return np.float32(hogdata).reshape(-1, 64)


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


if __name__ == '__main__':
    LABEL_FILE = 'MNIST/train-labels-idx1-ubyte'
    IMAGE_FILE = 'MNIST/train-images-idx3-ubyte'
    TRAIN_SIZE = 10000

    images, labels, num, rows, cols = get_data(LABEL_FILE, IMAGE_FILE)
    rec = Recognizer('sk', model_path='svc_sk.pkl', 
        train_data={'images':images[:TRAIN_SIZE], 'labels':labels[:TRAIN_SIZE]})
    print(rec)
    rec = Recognizer('sk', model_path='svc_sk.pkl')
    print(rec)
    rec = Recognizer('opencv', model_path='svc_opencv.pkl',
        train_data={'images':images[:TRAIN_SIZE], 'labels':labels[:TRAIN_SIZE]})
    print(rec)

"""


    print('Training OpenCV SVM digits recognizer...')
    svc1 = cvtrain(images[:TRAIN_SIZE], labels[:TRAIN_SIZE], num, rows, cols)
    fs = cv2.FileStorage('svc1.bin', flags=1)
    fs.write(name='classifier',val=pickle.dumps(svc1))
    fs.release()
    #svc1.save('svc1.bin')

    

def cvtrain(images, labels, num, rows, cols):
    svc = cv2.ml.SVM_create()
    svc.setType(svm_params['svm_type'])
    svc.setKernel(svm_params['kernel_type'])
    traindata = preprocess(images, rows, cols)
    #responses = np.float32(labels[:, None])
    responses = np.array(labels, dtype=np.int)# np.int(labels[:, None]) #[:, None])
    svc.train(traindata, cv2.ml.ROW_SAMPLE, responses)
    return svc


def sktrain(images, labels):
    svc = svm.SVC(kernel='linear')
    svc.fit(images, labels)
    return svc

"""