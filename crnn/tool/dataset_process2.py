# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
import numpy as np
import cv2
from math import *
import matplotlib.image as image
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

im_dir = '/export/home/frankzhan/datasets/stv2k_test/'
save_dir = '/export/home/frankzhan/datasets/test/'

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
            len_labels = len(labels)
            for j in range(0, len_labels, 3):
                coord = labels[j]
                annot = labels[j + 1]
                if annot != '':
                    annot = annot.decode('GB18030')
                    # print (annot)
                    spt = coord.split(',')
                    x_sum = sum([float(spt[0]), float(spt[2]), float(spt[4]), float(spt[6])])
                    x_mean = x_sum/float(4)
                    if float(spt[0])<x_mean:
                        if (float(spt[2]) - float(spt[0])) > (float(spt[7]) - float(spt[1])):
                            pt1 = (float(spt[0]), float(spt[1]))
                            pt2 = (float(spt[2]), float(spt[3]))
                            pt3 = (float(spt[4]), float(spt[5]))
                            pt4 = (float(spt[6]), float(spt[7]))
                            partImg = dumpRotateImage(im, degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])), pt1, pt2, pt3, pt4)
                           # print (partImg.shape)
                            r = Image.fromarray(partImg[:, :, 0]).convert('L')
                            g = Image.fromarray(partImg[:, :, 1]).convert('L')
                            b = Image.fromarray(partImg[:, :, 2]).convert('L')
                            rgb = Image.merge("RGB", (r, g, b))
                            imsave_path = save_dir + 'image_{}.jpg'.format(index)
                            rgb.save(imsave_path)
                            txtsave_path = save_dir + 'image_{}.txt'.format(index)
                            File = open(txtsave_path, 'w')
                            File.write(annot)
                            index += 1
                    else:
                        x1 = float(spt[6])
                        y1 = float(spt[7])
                        x2 = float(spt[0])
                        y2 = float(spt[1])
                        x3 = float(spt[2])
                        y3 = float(spt[3])
                        x4 = float(spt[4])
                        y4 = float(spt[5])
                        if (x2 - x1) > (y4 - y1):
                            pt1 = (x1, y1)
                            pt2 = (x2, y2)
                            pt3 = (x3, y3)
                            pt4 = (x4, y4)
                            partImg = dumpRotateImage(im, degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])), pt1, pt2, pt3,
                                                      pt4)
                            r = Image.fromarray(partImg[:, :, 0]).convert('L')
                            g = Image.fromarray(partImg[:, :, 1]).convert('L')
                            b = Image.fromarray(partImg[:, :, 2]).convert('L')
                            rgb = Image.merge("RGB", (r, g, b))
                            imsave_path = save_dir + 'image_{}.jpg'.format(index)
                            rgb.save(imsave_path)
                            txtsave_path = save_dir + 'image_{}.txt'.format(index)
                            File = open(txtsave_path, 'w')
                            File.write(annot)
                            index += 1
        except:
            continue

