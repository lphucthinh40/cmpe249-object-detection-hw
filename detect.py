import argparse
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

IMAGE_EXT = ['.png', '.jpg']
VIDEO_EXT = ['.mp4']

def main(opt):    
    if opt.model == 'yolov5':
        sys.path.insert(0, './yolov5')
        from yolov5.yolov5_detector import Yolov5Detector
        model = Yolov5Detector()
    elif opt.model == 'detectron2':
        sys.path.insert(0, './detectron')
        from detectron.detectron2_detector import Detectron2Detector
        model = Detectron2Detector()
    else:
        print('unsupported model selection!')
        exit(1)
    ext = os.path.splitext(opt.path)[1]
    if ext in IMAGE_EXT:
        model.detectImage(opt.path)
    elif ext in VIDEO_EXT:
        model.detectVideo(opt.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--model', type=str, default= 'yolov5', choices=['yolov5', 'detectron2'], help='pick detector model')
    opt = parser.parse_args()
    main(opt)
