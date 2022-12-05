'''
percentage of clean frames, based on which all the corrupted images are correctly predicted, defined as:
1. same number of boxes (no extra boxes)
2. the labels of all boxes are correctly predicted
'''

import os
import pickle
import numpy as np

labels = [0,2]
imgpath = "../example/test_frame_W/"
files = [f for f in os.listdir(imgpath) if os.path.isfile(os.path.join(imgpath, f))]
files.sort(key=lambda x: int(x[:-4]))

gtpath = '../example/bbox_test_W/'
testpath = '../example/bbox_adv_W/'

ovthresh = 0.5
cleannum = 0
for i in range(len(files)):
    gtinfo = np.load(gtpath + files[i][:-4] + ".npy")
    testinfo = np.load(testpath + files[i][:-4] + ".npy")

    gtindex = np.array([])
    testindex = np.array([])
    for label in labels:
        if gtinfo.size > 0:
            gtindex = np.concatenate((gtindex, np.where(gtinfo[:,5] == label)[0]))
        if testinfo.size > 0:
            testindex = np.concatenate((testindex, np.where(testinfo[:, 5] == label)[0]))

    gtinfo = gtinfo[gtindex.astype(int)]
    testinfo = testinfo[testindex.astype(int)]

    if len(gtinfo) != len(testinfo):
        continue
    # else:
    if len(gtinfo) == 0 and len(testinfo) == 0:
        cleannum += 1
        continue

    testboxnum = set(range(len(testinfo)))
    count = 0
    for gtb in gtinfo:
        gtlabel = int(gtb[5])
        gtbox = gtb[:4].astype(int)
        if testinfo.size > 0:
            testbox = testinfo[:,:4].astype(int)

            ovmax = -np.inf  # 负数最大值

            ixmin = np.maximum(testbox[:, 0], gtbox[0])
            iymin = np.maximum(testbox[:, 1], gtbox[1])
            ixmax = np.minimum(testbox[:, 2], gtbox[2])
            iymax = np.minimum(testbox[:, 3], gtbox[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((gtbox[2] - gtbox[0] + 1.) * (gtbox[3] - gtbox[1] + 1.) +
                   (testbox[:, 2] - testbox[:, 0] + 1.) *
                   (testbox[:, 3] - testbox[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)  # 最大重叠
            jmax = np.argmax(overlaps)  # 最大重合率对应的gt

            # 计算tp 和 fp个数
            if ovmax > ovthresh:
                testlabel = int(testinfo[jmax,5])
                if testlabel == gtlabel:
                    count += 1

    if count == len(gtinfo):
        cleannum += 1

print(cleannum / len(files))