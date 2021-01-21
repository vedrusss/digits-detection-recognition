__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import os, argparse
import cv2
import json

def create_negatives(images_root, dst_folder, target_size=(96, 96)):
    image_files = get_entity(images_root)
    dst_img_folder = 'img'
    dst_list_file = os.path.join(dst_folder, 'bg.txt')
    f = open(dst_list_file, 'w')
    amount = 0
    for i, fn in enumerate(image_files):
        im = cv2.imread(fn)
        if im is None: continue
        im = cv2.resize(im, target_size)
        info_line = os.path.join(dst_img_folder, f'{i+1}.jpg')
        os.makedirs(os.path.join(dst_folder, dst_img_folder), exist_ok=True)
        cv2.imwrite(os.path.join(dst_folder, info_line), im)
        f.write(f'{info_line}\n')
        amount += 1
    f.close()
    print(f"Negative samples information ({amount} of samples) stored to {dst_folder}")

def create_positives(images_root, annotations_root, dst_folder):
    samples = scan_data(images_root, annotations_root)
    dst_img_folder = 'img'
    dst_list_file = os.path.join(dst_folder, 'info.dat')
    f = open(dst_list_file, 'w')
    amount = 0
    for i, (image_fn, annotation_fn) in enumerate(samples):
        objects = json.load(open(annotation_fn)).get('objects', [])
        if len(objects) < 1: continue
        objects_s = str(len(objects))
        for obj in objects:
            l, t, r, b = obj['box']
            w = r - l
            h = b - t
            objects_s += f' {l} {t} {w} {h}'
        im = cv2.imread(image_fn)
        if im is None: continue
        info_line = os.path.join(dst_img_folder, f'{i+1}.jpg')
        os.makedirs(os.path.join(dst_folder, dst_img_folder), exist_ok=True)
        cv2.imwrite(os.path.join(dst_folder, info_line), im)
        f.write(f'{info_line} {objects_s}\n')
        amount += 1
    f.close()
    print(f"Positive samples information ({amount} of samples) stored to {dst_folder}") 

def get_base_name(path):
    fn = os.path.split(path)[-1]
    return '.'.join(fn.split('.')[:-1])

def get_entity(folder):
    return [os.path.join(folder, name) for name in os.listdir(folder)]

def scan_data(images_root, annotations_root):
    image_files = get_entity(images_root) 
    samples = []
    for fn in image_files:
        annotation_path = os.path.join(annotations_root, get_base_name(fn) + '.json')
        if not os.path.exists(annotation_path):
            continue
        samples.append((fn, annotation_path))
    return samples

def parse_args():
    parser = argparse.ArgumentParser("Tool to prepare positive and negative train sets for opencv")
    parser.add_argument('-d', '--dataset_root', type=str, required=True, help='Dataset root folder')
    parser.add_argument('-a', '--annotations_root', type=str, default=None,
        help='Annotations root folder (provide to create positive samples)')
    parser.add_argument('-o', '--output', type=str, required=True, help='Folder to store results')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    if args.annotations_root:
        create_positives(args.dataset_root, args.annotations_root, args.output)
    else:
        create_negatives(args.dataset_root, args.output)
