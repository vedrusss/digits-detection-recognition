__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import argparse
import os
import json
import cv2
import numpy as np
from collections import defaultdict
from detect import Detector
from detect_recognize import DigitsDetector

base_name = lambda path: '.'.join(os.path.split(path)[-1].split('.')[:-1])
scan_files = lambda folder: {base_name(name): os.path.join(folder, name)
                             for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))}

def compare(detections, gt_objects):
    TPs, FNs = defaultdict(int), defaultdict(int)
    for obj in gt_objects:
        label, ltrb = obj['label'], obj['box']
        boxes = [d['box'] for d in detections if d['label'] is None or d['label'] == label]
        if has_intersection(ltrb, boxes):
            TPs[label] += 1
        else:
            FNs[label] += 1
    FPs = defaultdict(int)
    for d in detections:
        label, box = d['label'], d['box']
        gt_boxes = [obj['box'] for obj in gt_objects if label is None or obj['label'] == label]
        if label is None:
            label = 'none'
        if not has_intersection(box, gt_boxes):
            FPs[label] += 1
    return TPs, FNs, FPs

def evaluate_detector(detector, test_data, and_recognizer=False, stop=None):
    all_TPs, all_FNs, all_FPs = defaultdict(int), defaultdict(int), defaultdict(int)
    i = 0
    for im_path, ann_path in test_data:
        image = cv2.imread(im_path)
        assert(image is not None), f"Cannot read {im_path}"
        objects = parse_annotation(ann_path)
        detections = detector(image)
        if not and_recognizer:
            detections = [{'box':b, 'label':None, 'score':None} for b in detections]
        TPs, FNs, FPs = compare(detections, objects)
        for label in TPs.keys():
            all_TPs[label] += TPs[label]
            all_FNs[label] += FNs[label]
        for label in FPs.keys():
            all_FPs[label] += FPs[label]
        i += 1
        if stop and i > stop: break
    
    pres, recs, f1ss = defaultdict(int), defaultdict(int), defaultdict(int)
    for label, fps in all_FPs.items():
        tps = all_TPs[label] if label in all_TPs else sum(all_TPs.values())
        fns = all_FNs[label] if label in all_FNs else sum(all_FNs.values())
        detections_amount = tps + fps
        gt_objects_amount = tps + fns
        pres[label], recs[label], f1ss[label] = pre_rec_f1s(tps, fns, fps)
    tps = sum(all_TPs.values())
    fns = sum(all_FNs.values())
    fps = sum(all_FPs.values())
    i_pre, i_rec, i_f1s = pre_rec_f1s(tps, fns, fps)
    stats = {}
    stats['per_label'] = (pres, recs, f1ss)
    stats['integral'] = (i_pre, i_rec, i_f1s)
    results = (all_TPs, all_FNs, all_FPs)
    return stats, results

def pre_rec_f1s(tps, fns, fps):
    all_dets = tps + fps
    gt_positives = tps + fns
    pre = tps / float(all_dets) if all_dets > 0 else None
    rec = tps / float(gt_positives) if gt_positives > 0 else None
    f1s = 2. * pre * rec / (pre + rec) if pre and rec else None
    return pre, rec, f1s

def has_intersection(box, boxes, iou_threhold=0.6):
    for b in boxes:
        IoU = iou(box, b)
        #print(box, b, IoU, iou_threhold)
        if IoU >= iou_threhold:
            return True
    return False

def iou(b1, b2):
    l, t = max(b1[0], b2[0]), max(b1[1], b2[1])
    r, b = min(b1[2], b2[2]), min(b1[3], b2[3])
    if l >= r or t >= b: 
        return 0.0 # no intersection at all
    interArea = max(0, r-l+1) * max(0, b-t+1)
    b1Area = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)
    b2Area = (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1)
    return interArea / float(b1Area + b2Area - interArea)


def parse_annotation(filepath):
    d = json.load(open(filepath))
    return d.get('objects', [])

def parse_args():
    parser = argparse.ArgumentParser("Tool to create data for digits detector")
    parser.add_argument('-i', '--images', type=str, required=True, 
                        help='Folders with digit images')
    parser.add_argument('-a', '--annotations', type=str, required=True, 
                        help='Folder annotations')
    parser.add_argument('-dm','--detector_models', type=str, nargs='+',
                        help='Detector model file(-s)')
    parser.add_argument('-cm','--classifier_model', type=str, default=None,
                        help="Specify classifier model to evaluate the whole pipeline")
    parser.add_argument('-o', '--output', type=str, default=None,
                        help="Where to store montages for analysis")
    return parser.parse_args()

def main(args):
    image_files = scan_files(args.images)
    annotations = scan_files(args.annotations)
    test_data = [[im_path, annotations[name]] for name, im_path in image_files.items() if name in annotations]
    
    opencv_model = False # args.detector_models.endswith('.xml')
    stop = None
    if args.classifier_model:
        detector = DigitsDetector(args.detector_models, args.classifier_model)
        stats, results = evaluate_detector(detector, test_data, True, stop)
    else:
        detector = Detector(args.detector_models, opencv_model, resize_factor=2.)
        stats, results = evaluate_detector(detector, test_data, False, stop)
    #  print stats
    print("--- 'per_label' ---")
    metrics = stats['per_label']
    for label in metrics[0].keys():
        print(f"{label}\tprecision: {round(metrics[0][label],3)}, recall: {round(metrics[1][label],3)}, f1 score: {round(metrics[2][label],3)}")
    print("--- 'integral' ---")
    metrics = stats['integral']
    print(f"precision: {round(metrics[0],3)}, recall: {round(metrics[1],3)}, f1 score: {round(metrics[2],3)}")

if __name__ == "__main__":
    main(parse_args())
