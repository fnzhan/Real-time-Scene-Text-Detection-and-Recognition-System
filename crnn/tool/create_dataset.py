# -*- coding:utf-8 -*-
import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import matplotlib.image as image
from PIL import Image

train_lmdb = '/export/home/frankzhan/datasets/lmdb/train/gaussian'
# validation_lmdb = '/export/home/frankzhan/Projects/sceneReco-master/crnn/dataset/validation.lmdb'
dataset_path = '/export/home/frankzhan/datasets/gaussian/'

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        try:
            imagePath = imagePathList[i]
            label = labelList[i]
            if not os.path.exists(imagePath):
                print('%s does not exist' % imagePath)
                continue
            with open(imagePath, 'r') as f:
                imageBin = f.read()
            if checkValid:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue

            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label
            if lexiconList:
                lexiconKey = 'lexicon-%09d' % cnt
                cache[lexiconKey] = ' '.join(lexiconList[i])
            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
        except:
            continue
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


imagePathList = []
labelList = []

name_list = os.listdir(dataset_path)
dict = {}
for name in name_list:
    if name.endswith('.jpg'):
        im_path = dataset_path + name
        im = image.imread(im_path)
        h, w, _ = im.shape
        ratio = float(w) / float(h)
        dict[name] = ratio
lst = list(dict.items())
lst.sort(key=lambda k:k[1])
im_names = [x[0] for x in lst]
# im_names = im_names[:36890]
# im_names.reverse()

for im_name in im_names:
    if im_name.endswith('jpg'):
        im_path = dataset_path + im_name
        imagePathList.append(im_path)

        name_tmp = im_name.split('.')[0]
        txt_name = name_tmp + '.txt'
        txt_path = dataset_path + txt_name
        label = open(txt_path).read().splitlines()
        label = label[0].encode('utf8')
        labelList.append(label)

createDataset(train_lmdb, imagePathList, labelList, lexiconList=None, checkValid=True)


if __name__ == '__main__':
    pass
