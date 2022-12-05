'''
take the adversarial samples as benchmark, to calculate the mAP metrics
'''
import sys
sys.path.append("../")

import os
import numpy as np
from PIL import Image

from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from example.utils import create_dir, process_label
from keras import backend as K
from models.yolov3 import YOLOv3_Darknet53
from accuracy_metrics.mAP import voc_eval
from PIL import Image, ImageFilter
from tog.attacks import *
import os
import cv2
K.clear_session()

weights = '../model_weights/YOLOv3_Darknet53.h5'  # TODO: Change this path to the victim model's weights
detector = YOLOv3_Darknet53(weights=weights)

quality = 90

adv = "adv"
gtpath = '../example/bbox_{}_W/'.format(adv)
gtimgpath = '../example/{}_frame_W/'.format(adv)
testpath = "../example/bbox_{}_compression_W/".format(adv)
testimgpath = "../example/{}_compression_frame_W/".format(adv)

create_dir(testpath)
create_dir(testimgpath)

files = [f for f in os.listdir(gtimgpath) if os.path.isfile(os.path.join(gtimgpath, f))]
files.sort(key=lambda x: int(x[:-4]))

#TODO: define based on wrong testcase
for i in range(len(files)):
    input_img = Image.open(gtimgpath + files[i])
    input_img.save(testimgpath + files[i], quality=quality)
    input_img = Image.open(testimgpath + files[i])

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
    np.save(testpath + files[i][:-4] + '.npy', bboxes)

mAP = []
# 计算每个类别的AP
for i in [0, 2]:
    rec, prec, ap = voc_eval(testpath, gtpath, i,  "./" )
    print("{} :	 {} ".format(i, ap))
    mAP.append(ap)

mAP = tuple(mAP)

# 输出总的mAP
print("mAP :	 {}".format( float( sum(mAP)/len(mAP)) ))





