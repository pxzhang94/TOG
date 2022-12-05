import sys
sys.path.append("../")

import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim

# def ssim(adv, ori):
#     #change to gray pic
#     # if 3 == len(adv.shape):
#     #     adv = adv * np.asarray([0.3 , 0.59, 0.11])
#     #     adv = adv.sum(axis=2)
#     #     ori = ori * np.asarray([0.3 , 0.59, 0.11])
#     #     ori = ori.sum(axis=2)
#     adv = adv.reshape(-1)
#     ori = ori.reshape(-1)
#
#     c1 = (0.01 * 255) ** 2
#     c2 = (0.03 * 255) ** 2
#     c3 = c2 / 2
#     alpha = 1
#     beta = 1
#     gama = 1
#
#     miu_x = adv.mean()
#     miu_y = ori.mean()
#
#     theta_x = adv.std(ddof=1)
#     theta_y = ori.std(ddof=1)
#     theta_xy = sum((adv - miu_x) * (ori - miu_y)) / (len(adv) - 1)
#
#     l = (2 * miu_x * miu_y + c1) / (miu_x ** 2 + miu_y ** 2 + c1)
#     c = (2 * theta_x * theta_y + c2) / (theta_x ** 2 + theta_y ** 2 + c2)
#     s = (theta_xy + c3) / (theta_x * theta_y + c3)
#
#     return (l ** alpha) * (c ** beta) * (s ** gama)

testpath = '../example/test_frame_W/'
advpath = "../example/adversarial_frame_W/"

files = [f for f in os.listdir(testpath) if os.path.isfile(os.path.join(testpath, f))]
files.sort(key=lambda x: int(x[:-4]))

ssim_value = 0
count = 0
#TODO: define based on wrong testcase
for i in range(len(files)):
    if os.path.exists(advpath  + files[i]):
        testimg = cv2.imread(testpath + files[i])
        testimg = cv2.cvtColor(testimg, cv2.COLOR_BGR2GRAY)
        advimg = cv2.imread(advpath  + files[i])
        advimg = cv2.cvtColor(advimg, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim_value + ssim(advimg, testimg)
        count += 1

result = ssim_value / count
print('average structural similarity of %d images is %.4f' % (count, result))