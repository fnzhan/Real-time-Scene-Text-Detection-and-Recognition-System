# -*- coding:utf-8 -*-
from __future__ import print_function
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import util
import dataset
import models.crnn as crnn
import keys
import editdistance
import chardet

device_id = 5
torch.cuda.set_device(device_id)
imgH = 32
nc = 1
nclass = 37
nh = 256
batchSize = 4
workers = 2
max_iter=100
n_test_disp = 10
valroot_path = '/export/home/frankzhan/datasets/lmdb/test/icdar2013'
crnn_path = '/export/home/frankzhan/projects/crnn/samples/crnn.pth'
# crnn_path = '/export/home/frankzhan/files/weights/CRNN_19.pth'

cudnn.benchmark = True
alphabet = keys.alphabet
converter = util.strLabelConverter(alphabet)
image = torch.FloatTensor(batchSize, 3, imgH, imgH)
text = torch.IntTensor(batchSize * 5)
length = torch.IntTensor(batchSize)
image = Variable(image)
text = Variable(text)
length = Variable(length)
criterion = CTCLoss()
crnn = crnn.CRNN(imgH, nc, nclass, nh, 1)
# crnn.apply(weights_init)
print('loading pretrained model')
crnn.load_state_dict(torch.load(crnn_path))

image = image.cuda()
crnn = crnn.cuda()
criterion = criterion.cuda()

test_dataset = dataset.lmdbDataset(root=valroot_path)
# sampler_mode = dataset.randomSequentialSampler(test_dataset, batchSize)
# loss averager
loss_avg = util.averager()

def val(net, test_dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False,
	batch_size=batchSize, num_workers=int(workers),
	collate_fn=dataset.alignCollate(imgH=32, imgW=100, keep_ratio=True)
	)
    val_iter = iter(data_loader)

    i = 0
    n = 0
    n_correct = 0
    n_text = 0
    loss_avg = util.averager()

    max_iter = len(data_loader)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        util.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        util.loadData(text, t)
        util.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
	    if isinstance(target, unicode) is False:
	    	target = target.decode('utf-8')
            pred_encode, _ = converter.encode(pred)
            target_encode,_ = converter.encode(target)
            t = editdistance.eval(pred_encode,target_encode)
	    l = len(target_encode)
	    # chardit1 = chardet.detect(target)
	    # print (chardit1)
	    print (pred+'>>>>'+target)
            n_correct += t
	    n_text += l
            n += 1
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:n_test_disp]
    for raw_pred, sim_pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, sim_pred, gt))

    len_edit = n_correct / float(n)
    len_text = n_text/float(n)
    norm = 1-len_edit/len_text
    print('average editdistance: %f, normalized accuracy: %f' % (len_edit, norm))

val(crnn, test_dataset, criterion)
