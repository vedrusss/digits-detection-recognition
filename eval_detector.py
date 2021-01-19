__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import argparse
import os
import json
import cv2
import numpy as np
from collections import defaultdict
from detect import Detector

base_name = lambda path: '.'.join(os.path.split(path)[-1].split('.')[:-1])
scan_files = lambda folder: {base_name(name): os.path.join(folder, name)
                             for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))}

def compare(boxes, gt_objects):
    to_ltrb = lambda xywh: (xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3])
    TPs, FNs = defaultdict(int), defaultdict(int)
    for obj in gt_objects:
        label, ltrb = obj['label'], obj['box']
        if has_intersection(ltrb, [to_ltrb(box) for box in boxes]):
            TPs[label] += 1
        else:
            FNs[label] += 1
    gt_boxes = [obj['box'] for obj in gt_objects]
    FPs = 0
    for xywh in boxes:
        if not has_intersection(to_ltrb(xywh), gt_boxes):
            FPs += 1
    return TPs, FNs, FPs

def evaluate_detector(detector, test_data):
    all_TPs, all_FNs = defaultdict(int), defaultdict(int)
    all_FPs = 0
    i = 0
    for im_path, ann_path in test_data:
        image = cv2.imread(im_path)
        assert(image is not None), f"Cannot read {im_path}"
        objects = parse_annotation(ann_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = detector(image)
        TPs, FNs, FPs = compare(boxes, objects)
        for label in TPs.keys():
            all_TPs[label] += TPs[label]
            all_FNs[label] += FNs[label]
        all_FPs += FPs
        i += 1
        if i > 100:
            break
    
    pres, recs, f1ss = defaultdict(int), defaultdict(int), defaultdict(int)
    for label in all_TPs.keys():
        pres[label], recs[label], f1ss[label] = pre_rec_f1s(all_TPs[label], all_FNs[label], all_FPs)
    i_pre, i_rec, i_f1s = pre_rec_f1s(sum(all_TPs.values()), sum(all_FNs.values()), all_FPs)
    stats = {}
    stats['per_label'] = (pres, recs, f1ss)
    stats['integral'] = (i_pre, i_rec, i_f1s)
    results = (all_TPs, all_FNs, all_FPs)
    return stats, results
    
def pre_rec_f1s(tps, fns, fps):
    pre = tps / float(tps + fps) if tps + fps > 0 else None
    rec = tps / float(tps + fns) if tps + fns > 0 else None
    f1s = 2. * pre * rec / (pre + rec) if pre and rec else None
    return pre, rec, f1s

def has_intersection(box, boxes, iou_threhold=0.3):
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
    parser.add_argument('-dm','--detector_model', type=str, default='models/cascade.xml',
                        help='Detector model file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help="Where to store montages for analysis")
    return parser.parse_args()

def main(args):
    image_files = scan_files(args.images)
    annotations = scan_files(args.annotations)
    test_data = [[im_path, annotations[name]] for name, im_path in image_files.items() if name in annotations]
    assert(os.path.isfile(args.detector_model)),f"Cannot find specified model {args.detector_model}"
    detector = Detector(args.detector_model)
    stats, results = evaluate_detector(detector, test_data)
    #  print stats
    print("--- 'per_label' ---")
    metrics = stats['per_label']
    for label in metrics[0].keys():
        print(f"{label}\tprecision: {metrics[0][label]}, recall: {metrics[1][label]}, f1 score: {metrics[2][label]}")
    print("--- 'integral' ---")
    metrics = stats['integral']
    print(f"{label}\tprecision: {metrics[0]}, recall: {metrics[1]}, f1 score: {metrics[2]}")

if __name__ == "__main__":
    main(parse_args())
