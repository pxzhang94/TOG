import sys
sys.path.append("../")
from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from example.utils import create_dir
from keras import backend as K
from models.yolov3 import YOLOv3_Darknet53
from PIL import Image
from tog.attacks import *
import os
import cv2
K.clear_session()

weights = '../model_weights/YOLOv3_Darknet53.h5'  # TODO: Change this path to the victim model's weights

detector = YOLOv3_Darknet53(weights=weights)

def process_label(olabel):
    if olabel == 0:
        nlabel = 4
    if olabel == 1:
        nlabel = 1
    if olabel == 2:
        nlabel = 14
    if olabel == 3:
        nlabel = 8
    if olabel == 4:
        nlabel = 39
    if olabel == 5:
        nlabel = 5
    if olabel == 6:
        nlabel = 2
    if olabel == 7:
        nlabel = 15
    if olabel == 8:
        nlabel = 56
    if olabel == 9:
        nlabel = 19
    if olabel == 10:
        nlabel = 60
    if olabel == 11:
        nlabel = 16
    if olabel == 12:
        nlabel = 17
    if olabel == 13:
        nlabel = 3
    if olabel == 14:
        nlabel = 0
    if olabel == 15:
        nlabel = 58
    if olabel == 16:
        nlabel = 18
    if olabel == 17:
        nlabel = 57
    if olabel == 18:
        nlabel = 6
    if olabel == 19:
        nlabel = 62
    return nlabel

eps = 8 / 255.       # Hyperparameter: epsilon in L-inf norm
eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
n_iter = 10          # Hyperparameter: number of attack iterations

# inputpath = 'video2frame_Z/'
# inputpath = 'adversarial_frame_Z/'
inputpath = 'haze_frame_W/'
create_dir("bbox_haze_W/")
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
    # np.save("bbox_test_Z/" + files[i][:-4] + '.npy', bboxes)
    np.save("bbox_haze_W/" + files[i][:-4] + '.npy', bboxes)

    # # Generation of the adversarial example
    # x_adv_vanishing = tog_vanishing(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
    # detections_adv_vanishing = detector.detect(x_adv_vanishing, conf_threshold=detector.confidence_thresh_default)
    # print(detections_adv_vanishing)

    # adv_x = (x_adv_vanishing[0]*255.).astype(np.uint8)
    #
    # cv2.imwrite(outputpath + "/%#05d.jpg" % (i + 1), adv_x[...,::-1])

# # Visualizing the detection results on the adversarial example and compare them with that on the benign input
# detections_adv_vanishing = detector.detect(x_adv_vanishing, conf_threshold=detector.confidence_thresh_default)
# visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
#                       'TOG-vanishing': (x_adv_vanishing, detections_adv_vanishing, detector.model_img_size, detector.classes)})