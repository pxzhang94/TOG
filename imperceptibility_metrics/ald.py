import sys
sys.path.append("../")

import numpy as np
import os
from PIL import Image
import cv2

def distortion_measure(adv, ori, p):
    # change to gray pic
    # if 3 == len(adv.shape):
    #     adv = adv * np.asarray([0.3, 0.59, 0.11])
    #     adv = adv.sum(axis=2)
    #     ori = ori * np.asarray([0.3, 0.59, 0.11])
    #     ori = ori.sum(axis=2)
    adv = adv.reshape(-1)
    ori = ori.reshape(-1)

    distance = 0
    ori_distance = 0
    if '0' == p:
        distance = distance + sum([int(i) for i in adv != ori])
        ori_distance = ori_distance + len(adv)
    elif '1' == p:
        distance = distance + sum(abs(adv - ori))
        ori_distance = ori_distance + sum(ori)
    elif '2' == p:
        distance = distance + sum((adv - ori) ** 2) ** 0.5
        ori_distance = ori_distance + sum(ori ** 2) ** 0.5
    elif 'inf' == p:
        distance = distance + max(abs(adv - ori))
        ori_distance = ori_distance + max(ori)
    return 1.0 * distance / ori_distance


p = '2'
testpath = '../example/test_frame_W/'
advpath = "../example/gaussian_frame_W/"

files = [f for f in os.listdir(testpath) if os.path.isfile(os.path.join(testpath, f))]
files.sort(key=lambda x: int(x[:-4]))

distortion = 0
count = 0
#TODO: define based on wrong testcase
for i in range(len(files)):
    if os.path.exists(advpath  + files[i]):
        # testimg = np.asarray(Image.open(testpath  + files[i]))
        # advimg = np.asarray(Image.open(advpath  + files[i]))
        testimg = cv2.imread(testpath + files[i])
        testimg = cv2.cvtColor(testimg, cv2.COLOR_BGR2GRAY)
        advimg = cv2.imread(advpath  + files[i])
        advimg = cv2.cvtColor(advimg, cv2.COLOR_BGR2GRAY)
        distortion = distortion + distortion_measure(advimg.astype("float"), testimg.astype("float"), p)
        count += 1

result = distortion / count
print('average L-%s distortion of %d images is %.4f' % (p, count, result))

