__author__ = "Alexey Antonenko, vedrusss@gmail.com"
import argparse
from os import listdir, makedirs
from os.path import join as opj
from os.path import split as ops
from os.path import isdir, isfile
from random import shuffle

def save(lst, dst):
    folder, _ = ops(dst)
    makedirs(folder, exist_ok=True)
    with open(dst, 'w') as f:
        for label, names in lst.items():
            for name in names:
                f.write(f"{name} {label}\n")

def split_ext(path):
    parts = path.split('.')
    ext = parts[-1]
    return '.'.join(parts[:-1]), ext

def main(dataset_root, split, output):
    get_folder_name = lambda path: ops(path)[-1] if ops(path)[-1] else ops(path)[-2]
    scan_files = lambda folder: [opj(get_folder_name(folder), name) for name in listdir(folder) \
        if isfile(opj(folder, name))]
    subfolders = {name : scan_files(opj(dataset_root, name)) \
        for name in listdir(dataset_root) if isdir(opj(dataset_root, name))}
    print(f"Found {len(subfolders)} subfolders:")
    train, test = {}, {}
    for folder in sorted(subfolders.keys()):
        paths = subfolders[folder]
        print(f"{folder} : {len(paths)}")
        if split:
            shuffle(paths)
            part = int(split * len(paths) / 100.)
            train[folder] = paths[:part]
            test[folder] = paths[part:]
        else:
            train[folder] = paths
    basename, ext = split_ext(output)
    if split:
        save(train, basename + '-train.' + ext)
        save(test, basename + '-test.' + ext)
    else:
        save(train, basename + '.' + ext)

def parse_args():
    parser = argparse.ArgumentParser("Tool set up list of data with groundtruth labels")
    parser.add_argument('-d', '--dataset_root', type=str, required=True,
        help='Dataset root folder (must contain subfolders with images. Each subfolder is a label)')
    parser.add_argument('-s', '--split', type=int, default=None,
        help='Specify train part (percents) if required to split into train/test')
    parser.add_argument('-o', '--output', type=str, required=True, help='List filename')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.dataset_root, args.split, args.output)
