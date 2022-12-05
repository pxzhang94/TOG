import os
import pickle
import numpy as np


def parse_rec(filename):
    objs = np.load(filename, allow_pickle=True)
    objects = []
    for obj in objs:
        obj_struct = {}
        obj_struct["label"] = int(obj[5])
        obj_struct["bbox"] = [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])]
        objects.append(obj_struct)

    return objects


# 计算AP，参考前面介绍
def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# 主函数，读取预测和真实数据，计算Recall, Precision, AP
def voc_eval(testpath,
             gtpath,
             label,
             cachedir,
             ovthresh=0.5):
    """
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) 需要计算的类别的txt文件路径.
    classname: 需要计算的类别
    cachedir: 缓存标注的目录
    [ovthresh]: IOU重叠度 (default = 0.5)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # # first load gt 加载ground truth。
    # if not os.path.isdir(cachedir):
    #     os.mkdir(cachedir)
    # cachefile = os.path.join(cachedir, "{}_annots.pkl".format(gtpath.split("_")[-2]))

    files = [f for f in os.listdir(gtpath) if os.path.isfile(os.path.join(gtpath, f))]
    # 所有文件名字。
    imagenames = [x.split(".npy")[0] for x in files]

    # #如果cachefile文件不存在，则写入
    # if not os.path.isfile(cachefile):
    #     # load annots
    #     recs = {}
    #     for i, imagename in enumerate(imagenames):
    #         recs[imagename] = parse_rec(gtpath + imagenames[i] + '.npy')
    #     with open(cachefile,  "wb") as f:
    #         #写入cPickle文件里面。写入的是一个字典，左侧为xml文件名，右侧为文件里面个各个参数。
    #         pickle.dump(recs, f)
    # else:
    #     # load
    #     with open(cachefile,  "rb") as f:
    #         recs = pickle.load(f)
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(gtpath + imagenames[i] + '.npy')

    # 对每张图片的xml获取函数指定类的bbox等
    class_recs = {}  # 保存的是 Ground Truth的数据
    npos = 0
    for imagename in imagenames:
        # 获取Ground Truth每个文件中某种类别的物体
        R = [obj for obj in recs[imagename] if obj["label"] == label]

        bbox = np.array([x["bbox"] for x in R])
        det = [False] * len(R)  # list中形参len(R)个False。
        npos = npos + len(R)  # 自增，~difficult取反,统计样本个数

        # 记录Ground Truth的内容
        class_recs[imagename] = {"bbox": bbox,
                                 "det": det}

    # read dets 读取某类别预测输出
    files = [f for f in os.listdir(testpath) if os.path.isfile(os.path.join(testpath, f))]
    files.sort(key=lambda x: int(x[:-4]))

    predictions = []
    for j in range(len(files)):
        objs = np.load(testpath + files[j], allow_pickle=True)
        for obj in objs:
            if int(obj[5]) == label:
                predictions.append([files[j][:-4], obj[4], obj[0], obj[1], obj[2], obj[3]])

    image_ids = [x[0].split(".")[0] for x in predictions]
    confidence = np.array([float(x[1]) for x in predictions])
    BB = np.array([[float(z) for z in x[2:]] for x in predictions])  # bounding box数值

    # 对confidence的index根据值大小进行降序排列。
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]  # 重排bbox，由大概率到小概率。
    image_ids = [image_ids[x] for x in sorted_ind]  # 图片重排，由大概率到小概率。

    # go down dets and mark TPs and FPs
    nd = len(image_ids)

    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]  # ann

        bb = BB[d, :].astype(float)

        ovmax = -np.inf  # 负数最大值
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)  # 最大重叠
            jmax = np.argmax(overlaps)  # 最大重合率对应的gt
        # 计算tp 和 fp个数
        if ovmax > ovthresh:
            # 该gt被置为已检测到，下一次若还有另一个检测结果与之重合率满足阈值，则不能认为多检测到一个目标
            if not R["det"][jmax]:
                tp[d] = 1.
                R["det"][jmax] = 1  # 标记为已检测
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)  # np.cumsum() 按位累加
    tp = np.cumsum(tp)
    rec = tp / float(npos)

    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    # np.finfo(np.float64).eps 为大于0的无穷小
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return rec, prec, ap

# gtpath = '../example/bbox_test_W/'
# testpath = "../example/bbox_adv_W/"
# mAP = []
# # 计算每个类别的AP
# for i in [0, 2]:
#     rec, prec, ap = voc_eval(testpath, gtpath, i,  "./" )
#     print("{} :	 {} ".format(i, ap))
#     mAP.append(ap)
#
# mAP = tuple(mAP)
#
# # 输出总的mAP
# print("mAP :	 {}".format( float( sum(mAP)/len(mAP)) ))