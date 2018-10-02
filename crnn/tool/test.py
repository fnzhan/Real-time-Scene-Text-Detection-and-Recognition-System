# -*- coding:utf-8 -*-
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf8')
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


from crnn import keys

alphabet = keys.alphabet
print (type(alphabet))
print (alphabet)

dict = {}
alphabet = '-' + alphabet
for i, char in enumerate(alphabet):
    print (char, i)
    dict[char] = i + 1

# alphabet = alphabet.decode()
# print (len(alphabet))
#
# for i, char in enumerate(alphabet):
#     print (char)

# dataset_path = '/home/frankzhan/Dataset/icdar_chinese/'
# txt_path = dataset_path + 'image_1.txt'
# label = open(txt_path).read().splitlines()
# label = label[0].encode('utf8')
# print (label)
# label = unicode(label, "utf8")
# for str in label:
#     index = dict[str]
#     print (index)

d={'site':'http://www.jb51.net','name':'jb51','is_good':'yes'}
#方法1：通过has_key
print (d.has_key('site'))
#方法2：通过in
print ('body' in d.keys())