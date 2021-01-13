__author__ = "Alexey Antonenko, vedrusss@gmail.com"

from torchvision import models
import torch

import os
import cv2
import numpy as np
from sklearn import svm
import pickle
import data
from evaluate import evaluate_classifier
from nnLlayer import LLayerNN
from classifier_model import initialize_cnn_model, train_cnn_model, test_cnn_model, save_cnn_model


#svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR, svm_type=cv2.ml.SVM_C_SVC)
svm_params = dict(kernel_type=cv2.ml.SVM_INTER, svm_type=cv2.ml.SVM_C_SVC)
SZ = 24
SAMPLE_SIZE = (SZ, 2*SZ)  #  w, h
deskew_size=32
hist_bin_n=32


class Recognizer:
    def __init__(self, fw_type, model_path=None, train_data=None):
        if fw_type != 'cnn':
            if train_data:
                elabels, self._mapping_from_labels, self._mapping_to_labels = \
                    data.encode_labels(train_data['labels'])
            else:
                elabels = None
                self._mapping_to_labels = {}
                self._mapping_from_labels = {}
                images = None
            if fw_type == 'opencv':
                self._recognizer = self._init_cv(images, elabels)
                self.predict = self._predict_cv
            elif fw_type == 'sk':
                self._recognizer = self._init_sk(model_path, images, elabels)
                self.predict = self._predict_sk
            elif fw_type == 'snn':
                self._recognizer = self._init_snn(model_path, images, elabels)
                self.predict = self._predict_snn
        else:  #  fw_type == 'cnn':
            self._recognizer = self._init_cnn(model_path, train_data)
            self.predict = self._predict_cnn            

    def _init_cv(self, image_files, labels):
        svc = cv2.ml.SVM_create()
        svc.setType(svm_params['svm_type'])        
        svc.setKernel(svm_params['kernel_type'])
        svc.setC(0.00001)
        if image_files and labels:
            traindata = data.preprocess_for_opencv(image_files, deskew_size, hist_bin_n)
            responses = np.array(labels, dtype=np.int)
            svc.train(traindata, cv2.ml.ROW_SAMPLE, responses)
            print("C: ", svc.getC())
        return svc

    def _init_cnn(self, model_path, train_data):
        if train_data:
            num_labels = len(os.listdir(train_data[0]))
            cnn, _ = initialize_cnn_model(num_labels=num_labels)
            self._mapping_to_labels = train_cnn_model(cnn, train_data[0], train_data[1])
            save_cnn_model(cnn, self._mapping_to_labels, model_path)
        else:
            cnn, self._mapping_to_labels = initialize_cnn_model(model_path=model_path)
        self._mapping_from_labels = {v:k for k,v in self._mapping_to_labels.items()}
        return cnn

    def _init_snn(self, model_path, image_files, labels):
        if train_data:
            print('Training shallow NN..')
            labels = train_data['labels']
            num_classes = len(set(labels))
            labels = [data.label_to_binary(label, num_classes) for label in labels]
            layers = ({'activation' : 'Tanh'   , 'size' : 24},
                      {'activation' : 'Tanh'   , 'size' : 12},
                      {'activation' : 'Tanh'   , 'size' :  8},
                      {'activation' : 'Sigmoid', 'size' :  6})
            traindata = data.preprocess_for_opencv(image_files, deskew_size, hist_bin_n).T
            responses = np.array(labels, dtype=np.float).T
            shallowNN = LLayerNN(traindata.shape[0], layers, num_classes)
            traindata = shallowNN.normalize(traindata)
            shallowNN.train(traindata, responses, minibatchSize=traindata.shape[1], epochsNum = 50000)
            shallowNN.save(model_path)
        else:
            print('Loading shallow NN')
            shallowNN = LLayerNN()
            shallowNN.load(model_path)
        return shallowNN

    def _init_sk(self, model_path, image_files, labels):
        if train_data:
            print('Training sklearn SVM digits recognizer...')
            svc = svm.SVC(kernel='rbf', C=1.0, gamma='scale')  #  'linear'  kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
            #traindata = [image.ravel() for image in train_data['images']]
            traindata = data.preprocess_for_opencv(image_files, deskew_size, hist_bin_n)
            svc.fit(traindata, labels)
            print("C: ", svc.C)
            with open(model_path, 'wb') as f:
                f.write(pickle.dumps(svc))
        else:
            print("Loading sklearn SVM digits recognizer")
            with open(model_path, 'rb') as f:
                svc = pickle.loads(f.read())
        return svc

    def _predict_cv(self, imgs):
        testdata = data.preprocess_for_opencv(imgs, deskew_size, hist_bin_n).reshape(-1, hist_bin_n * 4)
        res = self._recognizer.predict(testdata)[1]
        labels = res.astype(np.uint8).ravel()
        return labels

    def _predict_cnn(self, imgs):
        return test_cnn_model(self._recognizer, imgs)

    def _predict_snn(self, imgs):
        testdata = data.preprocess_for_opencv(imgs, deskew_size, hist_bin_n).reshape(-1, hist_bin_n * 4).T
        testdata = self._recognizer.normalize(testdata)
        res = self._recognizer.predict(testdata)
        labels = [np.argmax(res[:,i]) for i in range(res.shape[1])]
        #labels = res.astype(np.uint8).ravel()
        return labels

    def _predict_sk(self, imgs):
        #testdata = [image.ravel() for image in imgs]
        testdata = data.preprocess_for_opencv(imgs, deskew_size, hist_bin_n).reshape(-1, hist_bin_n * 4)
        labels = self._recognizer.predict(testdata)
        return labels

    @property
    def mapping_to_labels(self):
        return self._mapping_to_labels


if __name__ == '__main__':
    import sys

    use_trained = True
    data_stats = True
    keep_aspect_ratio = False

    if len(sys.argv) < 4:
        cases = {1: {'train_lst':'train', 'test_lst':'test'},
                 2: {'train_lst':'test',  'test_lst':'train'},
                 3: {'train_lst':'train_12345colonempty', 'test_lst':'test_12345colonempty'},
                 4: {'train_lst':'train_12345empty', 'test_lst':'test_12345empty'}}
        
        chosen_case = 1
        framework = "cnn" #  "opencv" "sk" "snn" "cnn"
        data_root = '/data/tasks/ocr_pipeline/calculator_font/digits_and_signs/digits_only'
        data_name = os.path.split(data_root)[-1]
        train_lst, test_lst = cases[chosen_case]['train_lst'], cases[chosen_case]['test_lst']
        
        train_dir, test_dir = train_lst.split('_')[0], test_lst.split('_')[0]
        train_data_root = f'{data_root}/{train_dir}'
        test_data_root = f'{data_root}/{test_dir}'
        train_data_lst = f'{data_root}/{train_lst}.lst'
        test_data_lst = f'{data_root}/{test_lst}.lst'
        model_path = f'digits_{framework}_{train_lst}_{data_name}'
        if framework == 'cnn': model_path += '.pth'
        else: model_path += '.pkl'
        print(f"Using default launch parameters.\n You may specify them by {sys.argv[0]} <framework> <train_data_root> <test_data_root> <train_list> <test_list> <model_path>")
    else:
        framework = sys.argv[1]
        train_data_root = sys.argv[2]
        test_data_root = sys.argv[3]
        train_data_lst = sys.argv[4]
        test_data_lst = sys.argv[5]
        model_path = sys.argv[6]

    print(f"Running with parameters:\n {train_data_root}\n {test_data_root}\n {train_data_lst}\n {test_data_lst}\n {model_path}")
    
    # Load data and train the model
    files, labels = data.load_data(data_lst=train_data_lst, data_root=train_data_root, verbose=data_stats)
    images = [data.prepare_image(fn, SAMPLE_SIZE, keep_aspect_ratio) for fn in files]
    if framework == 'cnn':
        train_data = None if use_trained else (train_data_root, test_data_root)
        rec = Recognizer(framework, model_path=model_path, train_data=train_data)
    else:
        train_data = None if use_trained else {'images':images, 'labels':labels}
        rec = Recognizer(framework, model_path=model_path, train_data=train_data)
    print(f"Trained/loaded/saved from/to {model_path}\n")

    print("Mapping id to label:", rec.mapping_to_labels)

    print("Evaluation against TRAIN set")
    predictions = [rec.mapping_to_labels[id] for id in rec.predict(images)]
    evaluate_classifier(predictions, labels)

    #  Load test data and test the model
    test_files, test_labels = data.load_data(data_lst=test_data_lst, data_root=test_data_root, verbose=data_stats)
    test_images = [data.prepare_image(fn, SAMPLE_SIZE, keep_aspect_ratio) for fn in test_files]
    print("Evaluation against TEST set")
    predictions = [rec.mapping_to_labels[id] for id in rec.predict(test_images)]
    evaluate_classifier(predictions, test_labels, False)
