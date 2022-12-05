import sys
sys.path.append("../")
from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from example.utils import create_dir, process_label
from keras import backend as K
from models.yolov3 import YOLOv3_Darknet53
from PIL import Image
from tog.attacks import *
import os
import cv2
K.clear_session()

weights = '../model_weights/YOLOv3_Darknet53.h5'  # TODO: Change this path to the victim model's weights

detector = YOLOv3_Darknet53(weights=weights)

eps = 8 / 255.       # Hyperparameter: epsilon in L-inf norm
eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
n_iter = 10          # Hyperparameter: number of attack iterations

# inputpath = 'video2frame_Z/'
# inputpath = 'adversarial_frame_Z/'
inputpath = 'gaussian_frame_W/'
bbox_path = "bbox_gaussian_W/"
create_dir(bbox_path)
files = [f for f in os.listdir(inputpath) if os.path.isfile(os.path.join(inputpath, f))]
files.sort(key=lambda x: int(x[:-4]))

for i in range(len(files)):
    input_img = Image.open(inputpath + files[i])

    x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
    detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)

    bboxes = []
    for d_result in detections_query:
        x_min = d_result[-4]
        y_min = d_result[-3]
        x_max = d_result[-2]
        y_max = d_result[-1]
        s = d_result[1]
        l = process_label(d_result[0])
        bboxes.append(np.asarray([x_min, y_min, x_max, y_max, s, l]))
    np.save(bbox_path + files[i][:-4] + '.npy', bboxes)
