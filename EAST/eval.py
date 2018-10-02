import cv2
import time
import math
from math import *
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#east
import tensorflow as tf
import lanms
tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_icdar2015_resnet_v1_50_rbox/', '')
import model as east_model
from icdar import restore_rectangle
FLAGS = tf.app.flags.FLAGS
print ('east')



#crnn
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import sys
sys.path.insert(1, '../crnn')
import util
import dataset
import models.crnn as crnn
from warpctc_pytorch import CTCLoss
import editdistance
device_id = 0
torch.cuda.set_device(device_id)
model_path = '/home/fnzhan/projects/end_to_end/crnn/samples/crnn.pth'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
model = crnn.CRNN(32, 1, 37, 256,1)
model = model.cuda()
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc:storage))
converter = util.strLabelConverter(alphabet)
transformer = dataset.resizeNormalize((100, 32))
criterion = CTCLoss()
model.eval()
show_size = (1920, 1080)  #(1024, 768)
print ('crnn')

font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 0.5
fontColor = (0,69,255)
lineType = 1

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

def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer=0, score_map_thresh=0.9, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    #print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    if len(boxes)!=0:
        boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry = east_model.model(input_images, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        #gpu_options = tf.GPUOptions(allow_growth = True)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            cap = cv2.VideoCapture(0)
            cap.set(3, 960)
            cap.set(4, 540)
            while(True):
                # try:
                    ret, frame = cap.read()
                    im = frame[:, :, ::-1]
                    im_resized, (ratio_h, ratio_w) = resize_image(im)
                    score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                    boxes, _ = detect(score_map=score, geo_map=geometry)

                    # save to file
                    if len(boxes) != 0:
                        try:
                            boxes = boxes[:, :8].reshape((-1, 4, 2))
                            boxes[:, :, 0] /= ratio_w
                            boxes[:, :, 1] /= ratio_h
                            for box in boxes:
                                # to avoid submitting errors
                                box = sort_poly(box.astype(np.int32))
                                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                    continue

                                # recognition
                                pt1 = (box[0, 0], box[0, 1])
                                pt2 = (box[1, 0], box[1, 1])
                                pt3 = (box[2, 0], box[2, 1])
                                pt4 = (box[3, 0], box[3, 1])
                                partImg = dumpRotateImage(im, degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])), pt1, pt2, pt3, pt4)
                                txt_im = Image.fromarray(partImg).convert('L')
                                txt_im = transformer(txt_im).cuda()
                                txt_im = Variable(txt_im.view(1, *txt_im.size()))
                                preds = model(txt_im)
                                _, preds = preds.max(2)
                                preds = preds.transpose(1, 0).contiguous().view(-1)
                                preds_size = Variable(torch.IntTensor([preds.size(0)]))
                                sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

                                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                              color=(0, 255, 0), thickness=1)

                                bottomLeftCornerOfText = pt1
                                cv2.putText(im[:, :, ::-1], sim_pred,
                                            bottomLeftCornerOfText,
                                            font,
                                            fontScale,
                                            fontColor,
                                            lineType)
                            im = cv2.resize(im, show_size)
                            cv2.imshow('frame', im[:, :, ::-1])
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        except:
                            continue
                    else:
                        im = cv2.resize(im, show_size)
                        cv2.imshow('frame', im[:, :, ::-1])
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                # except:
                #     continue
            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    tf.app.run()
