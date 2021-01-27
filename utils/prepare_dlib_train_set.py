__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import argparse
import os
import json
import cv2
from random import randint

def base_name(path):
    return os.path.split(path)[-1]

def get_images(folder):
    return {name:os.path.join(folder, name) for name in os.listdir(folder) 
            if os.path.isfile(os.path.join(folder, name))}

def intersects(box, boxes):
    inside_rect = lambda x, y, l, t, r, b: (l <= x <= r) and (t <= y <= b)
    for (l, t, r, b) in boxes:
        if any([inside_rect(x, y, l, t, r, b) for (x, y) in [(box[0], box[1]),
                                                             (box[2], box[1]),
                                                             (box[2], box[3]),
                                                             (box[0], box[3])]]):
            return True
    return False

def pick_up_position(dst_hw, src_hw, exclude_areas, verbose=False):
    while True:
        l = randint(0, dst_hw[1] - 1 - src_hw[1])
        r = l + src_hw[1]
        t = randint(0, dst_hw[0] - 1 - src_hw[0])
        b = t + src_hw[0]
        box = (l, t, r, b)
        if not intersects(box, exclude_areas):
            break
    if verbose:
        print(f"picked box {box}")
    return box

def pick_up_read_one(images_dict, target_size=None):
    keys = list(images_dict.keys())
    key  = randint(0, len(keys)-1)
    path = images_dict[keys[key]]
    image = cv2.imread(path)
    assert(image is not None),f"Cannot read image {path}"
    if target_size:
        image = cv2.resize(image, tuple(target_size))
    return image

def pick_up_read_set(labels_images_dict, amount, target_size):
    labels = list(labels_images_dict.keys())
    res_set = []
    while amount > 0:
        label = labels[randint(0, len(labels)-1)]
        image = pick_up_read_one(labels_images_dict[label], target_size)
        res_set.append((label, image))
        amount -= 1
    return res_set

def parse_args():
    parser = argparse.ArgumentParser("Tool to create train data for dlib based digits detector")
    parser.add_argument('-if', '--input_foreground', type=str, nargs='+', required=True, 
                        help='Folders with digits and signs to be detected')
    parser.add_argument('-ib', '--input_background', type=str, required=True, 
                        help='Folder with empty images to be used as background')
    parser.add_argument('-a', '--amount', type=int, default=5, 
                        help='Amount of foreground images to be used per generated image')
    parser.add_argument('-n', '--num_images', type=int, default=5000, 
                        help="Amount of images to be generated")
    parser.add_argument('-s', '--size', type=int, nargs=2, default=[24, 48],
         help="Target size for positive samples (24 and 48 by default)")
    parser.add_argument('-o', '--output', type=str, required=True, help="Folder to images to")
    return parser.parse_args()

def main(args):
    foregrounds = {base_name(path) : get_images(path) for path in args.input_foreground}
    background = get_images(args.input_background)
    dst_folder_root = args.output
    dst_folder_imgs = os.path.join(dst_folder_root, 'images')
    dst_folder_anns = os.path.join(dst_folder_root, 'annotations')
    os.makedirs(dst_folder_imgs, exist_ok=True)
    os.makedirs(dst_folder_anns, exist_ok=True)
    amount = args.num_images
    amount_per_image = args.amount

    while amount > 0:
        foreground_images = pick_up_read_set(foregrounds, amount_per_image, args.size)
        background_image = pick_up_read_one(background)
        b_height = int(2 * sum([img[1].shape[0] for img in foreground_images]))
        b_width  = int(2 * sum([img[1].shape[1] for img in foreground_images]))
        background_image = cv2.resize(background_image, (b_width, b_height), interpolation=cv2.INTER_CUBIC)
        locations = []
        for (label, img) in foreground_images:
            occupied = [location[1] for location in locations]
            #print(background_image.shape, img.shape)
            l, t, r, b = pick_up_position(background_image.shape[:2], img.shape[:2], occupied)
            #print(l,t,r,b)
            background_image[t:b, l:r, :] = img[:, :, :]
            locations.append((label, [l, t, r, b]))
        dst_img_path = os.path.join(dst_folder_imgs, f'{amount}.png')
        dst_ann_path = os.path.join(dst_folder_anns, f'{amount}.json')
        cv2.imwrite(dst_img_path, background_image)
        locations = {'objects': [{'label': l[0], 'box': l[1]} for l in locations]}
        json.dump(locations, open(dst_ann_path, 'w'))
        print(f"Created {base_name(dst_img_path)} + {base_name(dst_ann_path)}\t{amount} left")
        amount -= 1

if __name__ == "__main__":
    main(parse_args())
