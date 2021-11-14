import json

from tqdm import tqdm
from utils.general import download, Path


def argoverse2yolo(set):
    labels = {}
    a = json.load(open(set, "rb"))
    for annot in tqdm(a['annotations'], desc=f"Converting {set} to YOLOv5 format..."):
        img_id = annot['image_id']
        img_name = a['images'][img_id]['name']
        img_label_name = img_name[:-3] + "txt"

        cls = annot['category_id']  # instance class id
        x_center, y_center, width, height = annot['bbox']
        x_center = (x_center + width / 2) / 1920.0  # offset and scale
        y_center = (y_center + height / 2) / 1200.0  # offset and scale
        width /= 1920.0  # scale
        height /= 1200.0  # scale

        img_dir = set.parents[2] / 'Argoverse-1.1' / 'labels' / a['seq_dirs'][a['images'][annot['image_id']]['sid']]
        if not img_dir.exists():
            img_dir.mkdir(parents=True, exist_ok=True)

        k = str(img_dir / img_label_name)
        if k not in labels:
            labels[k] = []
        labels[k].append(f"{cls} {x_center} {y_center} {width} {height}\n")

    for k in labels:
        with open(k, "w") as f:
            f.writelines(labels[k])


# Download
dir = Path('../datasets/Argoverse')  # dataset root dir
urls = ['https://argoverse-hd.s3.us-east-2.amazonaws.com/Argoverse-HD-Full.zip']
download(urls, dir=dir, delete=False)

# Convert
annotations_dir = 'Argoverse-HD/annotations/'
(dir / 'Argoverse-1.1' / 'tracking').rename(dir / 'Argoverse-1.1' / 'images')  # rename 'tracking' to 'images'
for d in "train.json", "val.json":
    argoverse2yolo(dir / annotations_dir / d)  # convert VisDrone annotations to YOLO labels
