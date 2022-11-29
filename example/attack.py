import sys
sys.path.append("../")
from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
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

inputpath = 'video2frame_W/'
outputpath = "adversarial_frame_W/"
# inputpath = 'video2frame_Z/'
# outputpath = "adversarial_frame_Z/"
files = [f for f in os.listdir(inputpath) if os.path.isfile(os.path.join(inputpath, f))]
files.sort(key=lambda x: int(x[:-4]))

for i in range(len(files)):
    input_img = Image.open(inputpath + files[i])

    x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)

    # Generation of the adversarial example
    x_adv_vanishing = tog_vanishing(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)

    adv_x = (x_adv_vanishing[0]*255.).astype(np.uint8)

    cv2.imwrite(outputpath + "/%#05d.jpg" % (i + 1), adv_x[...,::-1])

# # Visualizing the detection results on the adversarial example and compare them with that on the benign input
# detections_adv_vanishing = detector.detect(x_adv_vanishing, conf_threshold=detector.confidence_thresh_default)
# visualize_detections({'Benign (No Attack)': (x_query, detections_query, detector.model_img_size, detector.classes),
#                       'TOG-vanishing': (x_adv_vanishing, detections_adv_vanishing, detector.model_img_size, detector.classes)})