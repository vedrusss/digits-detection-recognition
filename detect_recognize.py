__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import argparse
import cv2
from classifier_model import initialize_cnn_model, test_cnn_model
from detect import Detector

import gi
gi.require_version('Gtk', '3.0')

def get_crop(np_image, ltrb):
    l, t, r, b = ltrb
    if l < 0: l = 0
    if t < 0: t = 0
    if r >= np_image.shape[1]: r = np_image.shape[1] - 1
    if b >= np_image.shape[0]: b = np_image.shape[0] - 1
    return np_image[t:b, l:r] if len(np_image.shape) < 3 else np_image[t:b, l:r, :]

class DigitsDetector:
    def __init__(self, detector_model_files, classifier_model_file):
        self._detector = Detector(detector_model_files, opencv=False, resize_factor=2.)
        self._classifier, self._mapping_to_labels = initialize_cnn_model(model_path=classifier_model_file)

    def __call__(self, np_image):
        boxes = self._detector(np_image)
        crops = [get_crop(np_image, box) for box in boxes]
        classifications = test_cnn_model(self._classifier, crops)
        return [{'box':b, 'label':c[1], 'score':c[2]} for b, c in zip(boxes, classifications)]

def parse_args():
    parser = argparse.ArgumentParser("Digits/classifier detector")
    parser.add_argument('-i', '--images', type=str, nargs='+', help='Images to test/train on')
    parser.add_argument('-dm', '--detector_models',  type=str, nargs='+', required=True, help='Detector model paths')
    parser.add_argument('-cm', '--classifier_model', type=str, required=True, help="Classifier model path")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    digits_detector = DigitsDetector(args.detector_models, args.classifier_model)
    cv2.namedWindow('window')
    for image_filename in args.images:
        image = cv2.imread(image_filename)
        if image is None:
            print(f"Cannot read image {image_filename}")
            continue
        print(f"\t{image_filename}")
        cv2.imshow('window', image)
        detections = digits_detector(image)
        for det in detections:
            print(f"{det['label']} {det['score']} {det['box']}")
        if cv2.waitKey() == 27:
            break
    cv2.destroyAllWindows()
