import torch
from torch.autograd import Variable
import util
import dataset
from PIL import Image
import os
import models.crnn as crnn

device_id = 5
torch.cuda.set_device(device_id)

model_path = '/export/home/frankzhan/projects/crnn/samples/test.pth'
im_dir = '/export/home/frankzhan/datasets/icdar2013_png/'
sv_path = '/export/home/frankzhan/projects/crnn/submit.txt'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256,1)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = util.strLabelConverter(alphabet)
transformer = dataset.keep_ratio_normalize(True)

txt = open(sv_path,'w')
im_ls = os.listdir(im_dir)
for n in range(1,1096):
    nm = 'word_'+str(n)+'.png'
    im_path = im_dir + nm
    image = Image.open(im_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
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
        # print('%-20s => %-20s' % (finnal_pred, lexicon[0]))

