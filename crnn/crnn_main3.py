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
from collections import OrderedDict

id_gpu = 3

torch.cuda.set_device(id_gpu)

alphabet = keys.alphabet
nclass = len(alphabet)+1
trainroot_path = '/export/home/frankzhan/datasets/lmdb/train/synthtext_3w'
#trainroot_path = '/export/home/frankzhan/projects/text_reco/crnn/dataset'
valroot_path = '/export/home/frankzhan/datasets/lmdb/test/stv2k_sorted'
crnn_path = '/export/home/frankzhan/projects/text_reco_2500/crnn/samples/CRNN_synthtext.pth'
experiment_path = '/export/home/frankzhan/projects/text_reco_2500/crnn/samples'

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', type=str, help='path to dataset', default=trainroot_path)
parser.add_argument('--valroot', type=str, help='path to dataset', default=valroot_path)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--crnn', default=crnn_path, help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default=alphabet)
parser.add_argument('--experiment', default=experiment_path, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=5, help='Number of samples to display when tool')
parser.add_argument('--valInterval', type=int, default=2000, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=2000, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
# print(opt)

# opt.trainroot = trainroot_path

if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
assert train_dataset
if not opt.random_sample:
    sampler_mode = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler_mode = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler_mode,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(root=opt.valroot)

# nclass = len(opt.alphabet) + 1
# nclass = 2529
nc = 1

converter = util.strLabelConverter(opt.alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh, 1)
crnn.apply(weights_init)
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    #crnn.load_state_dict(torch.load(opt.crnn))
   # pretrained_dict = torch.load(opt.crnn)
    #new_state_dict = OrderedDict()
   # for k, v in pretrained_dict.items():
    #    name = k[7:]  # remove module.
     #   new_state_dict[name] = v

   # for name, module in crnn.named_children():
    #    if name == 'cnn':
     #       module_dict = module.state_dict()
      #      new_state_dict = {k: v for k, v in new_state_dict.items() if k in module_dict}

       #     module_dict.update(new_state_dict)
        #    module.load_state_dict(module_dict)


image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)	

if opt.cuda:
    crnn = crnn.cuda(id_gpu)
#    crnn = torch.nn.DataParallel(crnn, device_ids=[1,2,3,4])
    image = image.cuda(id_gpu)
    criterion = criterion.cuda(id_gpu)

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = util.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop([{'params':crnn.cnn.parameters(), 'lr':0.0},
			{'params':crnn.rnn.parameters(), 'lr':opt.lr}])

#for name,module in crnn.named_children():
 #   if name == 'cnn':
#	for name_layer,layer in module.named_children():
 #           for param in layer.parameters():
  #              param.requires_grad = False

#    if name == 'rnn':
#	for name_layer1,layer1 in module.named_children():
#	    for param in layer1.parameters():
#		param.requires_grad = False


def val(net, test_dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

#    layer_dict = net.state_dict()
#    print(layer_dict['cnn.conv1.weight'])

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers),
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
#	print (preds)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
	    if isinstance(target, unicode) is False:
                target = target.decode('utf-8')
            pred_encode, _ = converter.encode(pred)
            target_encode,_ = converter.encode(target)
            t = editdistance.eval(pred_encode,target_encode)
	    l = len(target_encode)
            n_correct += t
	    n_text += l
	    n += 1
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, sim_pred, gt in zip(raw_preds, sim_preds, cpu_texts):
    	print('%-20s => %-20s, gt: %-20s' % (raw_pred, sim_pred, gt))
    len_edit = n_correct/float(n)
    len_text = n_text/float(n)
    norm = 1-len_edit/len_text
    print('aver editd: %f, norm acc: %f' % (len_edit, norm))

def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    # print (type(cpu_texts), cpu_texts)
    batch_size = cpu_images.size(0)
    util.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    
    util.loadData(text, t)
    util.loadData(length, l)

    preds = crnn(image)
#    print (preds.size())
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
   # optimizer.zero_grad()
    cost.backward()
   # optimizer.step()
    torch.nn.utils.clip_grad_norm(crnn.parameters(), 5)
   # for p in crnn.parameters():
   #	p.data.add(-opt.lr, p.grad.data)
    for w in crnn.parameters():
	w.grad.data.clamp_(-5,5)
    optimizer.step()
    return cost

for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        
	crnn.train()
        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            val(crnn, test_dataset, criterion)

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{0}/CRNN_synthtex_3w.pth'.format(opt.experiment))
