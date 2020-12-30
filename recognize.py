__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import cv2
import numpy as np
from sklearn import svm
import pickle
import data
from evaluate import evaluate_classifier


svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR, svm_type=cv2.ml.SVM_C_SVC)
SZ = 16
SAMPLE_SIZE = (2*SZ, SZ)
deskew_size=32
hist_bin_n=16


class Recognizer:
    def __init__(self, fw_type, model_path=None, train_data=None):
        self._recognizer = self._init_cv(train_data) if fw_type == 'opencv' else \
                           self._init_sk(model_path, train_data)
        self.predict = self._predict_cv if fw_type == 'opencv' else self._predict_sk
    
    def _init_cv(self, train_data):
        svc = cv2.ml.SVM_create()
        svc.setType(svm_params['svm_type'])
        svc.setKernel(svm_params['kernel_type'])
        rows = SAMPLE_SIZE[0]
        cols = SAMPLE_SIZE[1]
        traindata = data.preprocess_for_opencv(train_data['images'], deskew_size, hist_bin_n)
        responses = np.array(train_data['labels'], dtype=np.int)
        svc.train(traindata, cv2.ml.ROW_SAMPLE, responses)
        return svc

    def _init_sk(self, model_path, train_data):
        if train_data:
            print('Training sklearn SVM digits recognizer...')
            svc = svm.SVC(kernel='linear')
            features = [image.ravel() for image in train_data['images']]
            svc.fit(features, train_data['labels'])
            with open(model_path, 'wb') as f:
                f.write(pickle.dumps(svc))
        else:
            print("Loading sklearn SVM digits recognizer")
            with open(model_path, 'rb') as f:
                svc = pickle.loads(f.read())
        return svc

    def _predict_cv(self, imgs):
        rows = SAMPLE_SIZE[0]
        cols = SAMPLE_SIZE[1]
        testdata = data.preprocess_for_opencv(imgs, deskew_size, hist_bin_n).reshape(-1, hist_bin_n * 4)
        res = self._recognizer.predict(testdata)[1]
        labels = res.astype(np.uint8).ravel()
        return labels

    def _predict_sk(self, imgs):
        features = [image.ravel() for image in imgs]
        labels = self._recognizer.predict(features)
        return labels


if __name__ == '__main__':
    framework = "sk" #  "opencv"
    use_trained = False

    data_root='/data/tasks/ocr_pipeline/calculator_font/digits_and_signs'
    train_data_lst='/data/tasks/ocr_pipeline/calculator_font/digits_and_signs/train-train.lst'
    test_data_lst='/data/tasks/ocr_pipeline/calculator_font/digits_and_signs/train-test.lst'
    
    # Load data and train the model
    files, labels = data.load_data(data_lst=train_data_lst, data_root=data_root, verbose=True)
    images = [data.prepare_image(fn, SAMPLE_SIZE) for fn in files]
    elabels, mapping = data.encode_labels(labels)
    train_data = None if use_trained else {'images':images, 'labels':elabels}
    rec = Recognizer(framework, model_path=f'digits_{framework}.pkl', train_data=train_data)
    print("Trained\n")

    #  Load test data and test the model
    test_files, test_labels = data.load_data(data_lst=test_data_lst, data_root=data_root, verbose=True)
    test_images = [data.prepare_image(fn, SAMPLE_SIZE) for fn in test_files]
    test_elabels, test_mapping = data.encode_labels(test_labels)
    predictions = rec.predict(test_images)
    for prediction, elabel, glabel in zip(predictions, test_elabels, test_labels):
        label = test_mapping[prediction]
        #print(f"prediction: {prediction}, elabel: {elabel} || label: {label}, glabel: {glabel}")
    tps, integral_tps = evaluate_classifier(predictions, test_elabels)
    print("Evaluation results:")
    for elabel in sorted(tps.keys()):
        label = test_mapping[elabel]
        print(f"{label} : {tps[elabel]}")
    print(f"Integral TPS: {integral_tps}")
