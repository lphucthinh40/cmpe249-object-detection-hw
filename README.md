# cmpe249-object-detection-hw
### Setup Environment
```
pip install -r requirements.txt
# install detectron2
pip install pyyaml==5.1
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
### Inference with Yolov5
```
# download weight files
https://drive.google.com/file/d/1jKbR9dt-8ApMYk0d7ljaGSWa1UU0Tk4w/view?usp=sharing
# run inference
python detect.py --model yolov5 --path test/test-image-1.jpg
(or for video: python detect.py --model yolov5 --path test/bdd100k-demo-vid-1.mp4)
```
### Inference with Detectron2
```
# download weight files
https://drive.google.com/file/d/1pPswBNWX0hDUwrC8v5qvdKE6UrkcusLd/view?usp=sharing
# run inference
python detect.py --model detectron2 --path test/test-image-1.jpg
(or for video: python detect.py --model detectron2 --path test/bdd100k-demo-vid-1.mp4)
```
