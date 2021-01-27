__author__ = "Alexey Antonenko, vedrusss@gmail.com"

import os, argparse
import cv2
import dlib
import json
import time

base_name = lambda path: '.'.join(os.path.split(path)[-1].split('.')[:-1])
scan_files = lambda folder: {base_name(name): os.path.join(folder, name)
                             for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))}

def check_folder(path):
    folder = os.path.split(path)[0]
    os.makedirs(folder, exist_ok=True)
    return path

def load_data(folder):
    image_files = scan_files(os.path.join(folder, 'images'))
    annotations = scan_files(os.path.join(folder, 'annotations'))
    images, boxes = [], []
    for name, im_path in image_files.items():
        if not name in annotations:
            continue
        img = cv2.imread(im_path)
        if img is None:
            continue
        objects = json.load(open(annotations[name])).get('objects', [])
        sample_boxes = [dlib.rectangle(left=obj['box'][0], top=obj['box'][1], right=obj['box'][2], bottom=obj['box'][3])
                 for obj in objects]
        images.append(img)
        boxes.append(sample_boxes)

        #if len(images) > 2000: break
    assert(len(images)), "No data found for training"
    return images, boxes


def train(images, boxes, filename):
    # Initialize object detector Options
    options = dlib.simple_object_detector_training_options()
    # I'm disabling the horizontal flipping, becauase it confuses the detector if you're training on few examples
    # By doing this the detector will only detect left or right hand (whichever you trained on).
    options.add_left_right_image_flips = False
    # Set the c parameter of SVM equal to 5
    # A bigger C encourages the model to better fit the training data, it can lead to overfitting.
    # So set an optimal C value via trail and error.
    options.C = 0.1

    options.num_threads = 1
    options.be_verbose = True
    #options.detection_window_size = 1152
    
    # You can start the training now
    print("Starting training")
    st = time.time()
    detector = dlib.train_simple_object_detector(images, boxes, options)
    # Print the Total time taken to train the detector
    print('Training Completed, Total Time taken: {:.2f} seconds'.format(time.time() - st))

    # Save trained model
    detector.save(check_folder(filename))

    # Eval detector against train set
    metrics = dlib.test_simple_object_detector(images, boxes, detector)
    print("Training Metrics: {}".format(metrics))

    return detector

def parse_args():
    parser = argparse.ArgumentParser("DLIB digits detection training")
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Root folder with images and annotations subfolders')
    parser.add_argument('-e', '--eval_dataset', type=str, default=None,
                        help='Root folder with images and annotations subfolders of test dataset')
    parser.add_argument('-m', '--model',  type=str, required=True, help='Model path to load/save')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    images, boxes = load_data(args.dataset)
    detector = train(images, boxes, args.model)

    if args.eval_dataset:
        images, boxes = load_data(args.eval_dataset)
        metrics = dlib.test_simple_object_detector(images, boxes, detector)
        print("Testing Metrics: {}".format(metrics))
