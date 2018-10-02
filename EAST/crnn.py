import sys
sys.path.insert(1, '../crnn')
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import util
import dataset
from PIL import Image
import os
import models.crnn as crnn
from warpctc_pytorch import CTCLoss
import numpy as np
import editdistance

use_gpu = True
device_id = 0
if use_gpu:
    torch.cuda.set_device(device_id)

model_path = '/home/zhanfangneng/projects/crnn/samples/crnn.pth'
im_dir = '/home/zhanfangneng/datasets/svt/textline/test/'
#lex_dir = '/export/home/frankzhan/datasets/IIIT5K/lower/test_lexicon_1k/'
sv_path = '/home/zhanfangneng/projects/crnn/submit.txt'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'


model = crnn.CRNN(32, 1, 37, 256,1)
if use_gpu:
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc:storage))
print ('end loading')
converter = util.strLabelConverter(alphabet)
#transformer = dataset.keep_ratio_normalize(True)
transformer = dataset.resizeNormalize((100, 32))
criterion = CTCLoss()

File = open(sv_path,'w')
im_ls = os.listdir(im_dir)
#im_ls = im_ls[:100]
for nm in im_ls:
    if nm.endswith('.png'):
        im_path = im_dir + nm
        image = Image.open(im_path).convert('L')
        image = transformer(image)
        if use_gpu:
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        model.eval()
        preds = model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
	print (nm)

