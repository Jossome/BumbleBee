import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *

# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# np.random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

# target = opt.target
# conf_target = opt.conf_target
# max_count = opt.max_count
# patch_type = opt.patch_type
# patch_size = opt.patch_size
# image_size = opt.image_size
# train_size = opt.train_size
# test_size = opt.test_size
# plot_all = opt.plot_all

print("Loading trained model")
net = torch.load("trained.model")
net.cuda()

'''
CURRENT STEP
'''


print('==> Preparing data..')
normalize = transforms.Normalize(mean=netClassifier.mean,
                                 std=netClassifier.std)
idx = np.arange(50000)
np.random.shuffle(idx)
training_idx = idx[:train_size]
test_idx = idx[train_size:test_size]

train_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('./imagenetdata/val', transforms.Compose([
        transforms.Scale(round(max(netClassifier.input_size)*1.050)),
        transforms.CenterCrop(max(netClassifier.input_size)),
        transforms.ToTensor(),
        ToSpaceBGR(netClassifier.input_space=='BGR'),
        ToRange255(max(netClassifier.input_range)==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(training_idx),
    num_workers=opt.workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('./imagenetdata/val', transforms.Compose([
        transforms.Scale(round(max(netClassifier.input_size)*1.050)),
        transforms.CenterCrop(max(netClassifier.input_size)),
        transforms.ToTensor(),
        ToSpaceBGR(netClassifier.input_space=='BGR'),
        ToRange255(max(netClassifier.input_range)==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(test_idx),
    num_workers=opt.workers, pin_memory=True)

min_in, max_in = netClassifier.input_range[0], netClassifier.input_range[1]
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
mean, std = np.array(netClassifier.mean), np.array(netClassifier.std)
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)


def train(epoch, patch, patch_shape):
    netClassifier.eval()
    success = 0
    total = 0
    recover_time = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)

        prediction = netClassifier(data)

        # only computer adversarial examples on examples that are originally classified correctly
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue

        total += 1

        # transform path
        data_shape = data.data.cpu().numpy().shape
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == 'square':
            patch, mask  = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        patch, mask = Variable(patch), Variable(mask)

        adv_x, mask, patch = attack(data, patch, mask)

        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]

        if adv_label == target:
            success += 1

            if plot_all == 1:
                # plot source image
                vutils.save_image(data.data, "./%s/%d_%d_original.png" %(opt.outf, batch_idx, ori_label), normalize=True)

                # plot adversarial image
                vutils.save_image(adv_x.data, "./%s/%d_%d_adversarial.png" %(opt.outf, batch_idx, adv_label), normalize=True)

        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]):
            for j in range(new_patch.shape[1]):
                new_patch[i][j] = submatrix(patch[i][j])

        patch = new_patch

        # log to file
        progress_bar(batch_idx, len(train_loader), "Train Patch Success: {:.3f}".format(success/total))

    return patch

def test(epoch, patch, patch_shape):
    netClassifier.eval()
    success = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)

        prediction = netClassifier(data)

        # only computer adversarial examples on examples that are originally classified correctly
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue

        total += 1

        # transform path
        data_shape = data.data.cpu().numpy().shape
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == 'square':
            patch, mask = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        patch, mask = Variable(patch), Variable(mask)

        adv_x = torch.mul((1-mask),data) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)

        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]

        if adv_label == target:
            success += 1

        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]):
            for j in range(new_patch.shape[1]):
                new_patch[i][j] = submatrix(patch[i][j])

        patch = new_patch

        # log to file
        progress_bar(batch_idx, len(test_loader), "Test Success: {:.3f}".format(success/total))

def attack(x, patch, mask):
    netClassifier.eval()

    x_out = F.softmax(netClassifier(x))
    target_prob = x_out.data[0][target]

    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)

    count = 0

    while conf_target > target_prob:
        count += 1
        adv_x = Variable(adv_x.data, requires_grad=True)
        adv_out = F.log_softmax(netClassifier(adv_x))

        adv_out_probs, adv_out_labels = adv_out.max(1)

        Loss = -adv_out[0][target]
        Loss.backward()

        adv_grad = adv_x.grad.clone()

        adv_x.grad.data.zero_()

        patch -= adv_grad

        adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)

        out = F.softmax(netClassifier(adv_x))
        target_prob = out.data[0][target]
        #y_argmax_prob = out.data.max(1)[0][0]

        #print(count, conf_target, target_prob, y_argmax_prob)

        if count >= opt.max_count:
            break


    return adv_x, mask, patch


if __name__ == '__main__':
    if patch_type == 'circle':
        patch, patch_shape = init_patch_circle(image_size, patch_size)
    elif patch_type == 'square':
        patch, patch_shape = init_patch_square(image_size, patch_size)
    else:
        sys.exit("Please choose a square or circle patch")

    for epoch in range(1, opt.epochs + 1):
        patch = train(epoch, patch, patch_shape)
        test(epoch, patch, patch_shape)
