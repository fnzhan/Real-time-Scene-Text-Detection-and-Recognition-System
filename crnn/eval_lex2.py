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

model_path = '/export/home/frankzhan/files/mjsynth_train.pth'
im_dir = '/export/home/frankzhan/datasets/IIIT5K/test/'
lex_dir = '/export/home/frankzhan/datasets/IIIT5K/lower/test_lexicon_50/'
sv_path = '/export/home/frankzhan/projects/crnn/submit.txt'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

def ascii2label(ascii):
    label = 0
    if ascii >= 48 and ascii <= 57:
        label = ascii - 47
    elif ascii >= 65 and ascii <= 90:
        label = ascii - 64 + 10
    elif ascii >= 97 and ascii <= 122:
        label = ascii - 96 + 10
    return label

def str2label(strs, maxlenth):
    nstrs =  len(strs)
    labels = torch.IntTensor(nstrs, maxlenth)
    for i, str in enumerate(strs):
        for j in range(len(str)):
            ascii = ord(str[j])
            labels[i][j] = ascii2label(ascii)
    return labels

def decodingWithLexicon(input, lexicon):
    # assert(input:dim() == 3 and input:size(1) == 1)
    # assert(type(lexicon) == 'table')
    lexSize = len(lexicon)

    target = str2label(lexicon, 30)
    inputN = input.repeat(lexSize, 1, 1)
    logProb = -torch.nn.CTC_forwardBackward(inputN, target, true, inputN.new())
    _, idx = torch.max(logProb, 1)
    idx = idx[1]
    return lexicon[idx]


model = crnn.CRNN(32, 1, 37, 256,1)
if use_gpu:
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc:storage))
print ('end loading')
converter = util.strLabelConverter(alphabet)
transformer = dataset.keep_ratio_normalize(True)
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

        tmp = nm.split('.')[0]
        tmp2 = tmp + '.txt'
        lex_path = lex_dir + tmp2
        txt = open(lex_path).read()
        wds = txt.splitlines()
        len_lexicon = len(wds)
        edit_dist = []
        lexicon = []
        for wd in wds:
            lexicon.append(wd)
            d = editdistance.eval(wd, sim_pred)
            edit_dist.append(d)
        idx = edit_dist.index(min(edit_dist))
        finnal_pred = lexicon[idx]
	finnal_pred = sim_pred

	nm_tmp = nm.split('_')
	word_nm = 'word'+'_'+nm_tmp[0]+'00'+nm_tmp[1]
	line = word_nm+', '+'"'+finnal_pred+'"'
   	File.write(line+'\n')
        print('%-20s => %-20s' % (finnal_pred, lexicon[0]))






