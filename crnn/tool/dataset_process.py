#coding:utf-8
# for icdar2017
import os
import lmdb # install lmdb by "pip install lmdb"
import numpy as np
import cv2
from math import *
import matplotlib.image as image
from PIL import Image

from PIL import Image

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut = imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
    height, width = imgOut.shape[:2]
    return imgOut

# dataset_path = '/export/home/frankzhan/datasets/icdar2017rctw_train_v1.2/train/'
# save_path = '/export/home/frankzhan/datasets/text_reco_train/'

im_dir = '/export/home/frankzhan/datasets/test/'
save_dir = '/export/home/frankzhan/datasets/test_icdar2017/'

name_list = os.listdir(im_dir)
index = 0
for name in name_list:
    if name.endswith('.jpg'):
        try:
            im_path = im_dir + name
            tmp = name.split('.')[0]
            txt_path = im_dir + tmp + '.txt'
            im = image.imread(im_path)
            txt = open(txt_path).read()
            labels = txt.splitlines()

            for each_label in labels:
                spt = each_label.split(',')
                if (int(spt[2])-int(spt[0]))>(int(spt[7])-int(spt[1])) and int(spt[8])!= 1 and len(spt)==10:
                    pt1 = (float(spt[0]), float(spt[1]))
                    pt2 = (float(spt[2]), float(spt[3]))
                    pt3 = (float(spt[4]), float(spt[5]))
                    pt4 = (float(spt[6]), float(spt[7]))
                    partImg = dumpRotateImage(im, degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])), pt1, pt2, pt3, pt4)
                    r = Image.fromarray(partImg[:, :, 0]).convert('L')
                    g = Image.fromarray(partImg[:, :, 1]).convert('L')
                    b = Image.fromarray(partImg[:, :, 2]).convert('L')
                    rgb = Image.merge("RGB", (r, g, b))
                    imsave_path = save_dir + 'image_{}.jpg'.format(index)
                    rgb.save(imsave_path)
                    txtsave_path = save_dir + 'image_{}.txt'.format(index)
                    File = open(txtsave_path, 'w')
                    spt_txt = eval(spt[9])
                    File.write(spt_txt)
                    index += 1

        except:
            continue

# plt.imshow(crop_img)
# plt.show()
