__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import argparse
import cv2
import os
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
    def __init__(self, detector_model_files, classifier_model_file, resize_factor=2.):
        self._detector = Detector(detector_model_files, opencv=False, resize_factor=resize_factor)
        self._classifier, self._mapping_to_labels = initialize_cnn_model(model_path=classifier_model_file)

    def __call__(self, np_image):
        boxes = self._detector(np_image)
        crops = [get_crop(np_image, box) for box in boxes]
        classifications = test_cnn_model(self._classifier, crops)
        return [{'box':b, 'label':c[1], 'score':c[2]} for b, c in zip(boxes, classifications)]

def compose_string(detections, align_horizontal=True, threshold=0.):
    ind = 0 if align_horizontal else 1
    dets = {d['box'][ind]:d['label'] for d in detections if d['score'] > threshold}
    s = ''
    for k in sorted(dets.keys()):
        s += dets[k]
    return s.strip()

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
    parser = argparse.ArgumentParser("Digits/classifier detector")
    parser.add_argument('-i', '--images', type=str, nargs='+', help='Images to test/train on')
    parser.add_argument('-dm', '--detector_models',  type=str, nargs='+', required=True, help='Detector model paths')
    parser.add_argument('-cm', '--classifier_model', type=str, required=True, help="Classifier model path")
    parser.add_argument('-cs', '--compose_string', action='store_true', help="Compose string")
    parser.add_argument('-is', '--ignore_signs', action='store_true', help="Ignore colons and dots")
    parser.add_argument('-d',  '--display', action='store_true', help="Display process")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    digits_detector = DigitsDetector(args.detector_models, args.classifier_model, resize_factor=2.1)
    if args.display: cv2.namedWindow('window')
    TPs, FPs = 0, 0
    for image_filename in args.images:
        image = cv2.imread(image_filename)
        if image is None:
            print(f"Cannot read image {image_filename}")
            continue
        print(f"\t{image_filename}")
        if args.display: cv2.imshow('window', image)
        detections = digits_detector(image)
        for det in detections:
            print(f"{det['label']} {det['score']} {det['box']}")
        if args.compose_string:
            name = parse_filename(image_filename)
            if args.ignore_signs: name = name.replace(':','').replace('.','').replace(' ','')
            det_string = compose_string(detections)
            print(f"{name}\t-->\t{det_string}")
            if det_string == name: TPs += 1
            else: FPs += 1
        if args.display: 
            if cv2.waitKey() == 27:
                break
    TPrate = TPs / float(TPs + FPs) if TPs+FPs>0 else None
    print(f"Results: TP rate {TPrate}, TPs {TPs}, FP {FPs}")    
    if args.display: cv2.destroyAllWindows()
