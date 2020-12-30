import os, argparse
import cv2
from random import shuffle
import uuid

def display_frames(cap, fnum):
    win = 'win'
    cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
    res, frame = cap.read()
    while res:
        cv2.imshow(win, frame)
        if cv2.waitKey() == 27: break
        res, frame = cap.read()

def extract_save_crops(cap, box, splits, frame_nums, dst_folder):
    extracted = 0
    for fnum in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        res, frame = cap.read()
        if not res: continue

        crop = frame[box[1]:box[3], box[0]:box[2], :]
        dst_crop, uid = make_dst_crop_name(dst_folder, splits, box[0])
        cv2.imwrite(dst_crop, crop)
        if splits:
            x0 = 0
            for x in splits:
                x1 = x - box[0]
                s_crop = crop[:, x0:x1, :]
                x0 = x1
                dst_split = make_dst_split_name(dst_folder, 'splits', x1, uid)
                cv2.imwrite(dst_split, s_crop)
        extracted += 1
    return extracted

def frames_to_extract(frames_amount, max_num_to_extract):
    frames = list(range(frames_amount))
    if max_num_to_extract is None or frames_amount <= max_num_to_extract:
        return frames
    shuffle(frames)
    return frames[:max_num_to_extract]

def make_dst_crop_name(folder, suffixies, a):
    os.makedirs(folder, exist_ok=True)
    uid = str(uuid.uuid4()).split('-')[-1]
    name = uid
    if suffixies:
        for s in suffixies: name += "_{}".format(s - a)
    name += ".png"
    return os.path.join(folder, name), uid

def make_dst_split_name(folder, subfolder, suffix, uid):
    os.makedirs(os.path.join(folder, subfolder), exist_ok=True)
    name = "{}_{}.png".format(uid, suffix)
    return os.path.join(folder, subfolder, name)

def parse_args():
    parser = argparse.ArgumentParser("Tool to extract specified regions from the video frames")
    parser.add_argument('-i', '--input_video', type=str, required=True, help='Video to parse')
    parser.add_argument('-b', '--box', type=int, nargs=4, required=True, 
                        help='Box left, top, right, bottom coordinates to crop from the frame')
    parser.add_argument('-s', '--splits', type=int, nargs=4, default=None, help='Split crop by X coordinates')
    parser.add_argument('-n', '--num_frames', type=int, default=None, help="Amount of frames to extract (all by default)")
    parser.add_argument('-o', '--output', type=str, required=True, help="Folder to save crops")
    parser.add_argument('-d', '--display', type=int, default=None, help="Just display frames from specified number")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    box = args.box
    assert(box[0] >=0 and box[1] >= 0 and box[2] > 0 and box[3] > 0),\
        "Box corners must be greater than zero"
    cap = cv2.VideoCapture(args.input_video)
    assert(cap),"Cannot load video " + args.input_video
    if args.display:
        display_frames(cap, args.display)
        quit()
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    assert(box[0] < w-1 and box[1] < h-1 and box[2] < w and box[3] < h),\
        "Box corners must be less than frame width x height ({} x {})".format(w, h)
    frames_amount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_nums = frames_to_extract(frames_amount, args.num_frames)
    assert(len(frame_nums)),"No frames to be extracted"
    amount = extract_save_crops(cap, box, args.splits, frame_nums, args.output)
    print("{} crops from {} have been extracted and saved to {}".format(amount, frames_amount, args.output))
