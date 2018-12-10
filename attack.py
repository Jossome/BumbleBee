import re
import argparse
import cv2
import os
import random
import numpy as np
import pickle
import imageio
from PIL import Image
from scipy.misc.pilutil import imresize

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

CATEGORIES = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking"
]

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
# Always cuda

parser.add_argument('--target', type=int, default=5, help='The target class: 859 == toaster')
parser.add_argument('--conf_target', type=float, default=0.9,
                    help='Stop attack on image when target classifier reaches this value for target class')

parser.add_argument('--max_count', type=int, default=300, help='max number of iterations to find adversarial example')
parser.add_argument('--patch_type', type=str, default='circle', help='patch type: circle or square')
parser.add_argument('--patch_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image ')

parser.add_argument('--train_size', type=int, default=2000, help='Number of training images')
parser.add_argument('--test_size', type=int, default=2000, help='Number of test images')

parser.add_argument('--frame_height', type=int, default=120, help='the height of the input to network')
parser.add_argument('--frame_width', type=int, default=160, help='the width of the input to network')

parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images')

parser.add_argument('--netClassifier', default='inceptionv3', help="The target classifier")

parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')

opt = parser.parse_args()

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

target = opt.target
conf_target = opt.conf_target
max_count = opt.max_count
patch_type = opt.patch_type
patch_size = opt.patch_size
frame_height = opt.frame_height
frame_width = opt.frame_width
train_size = opt.train_size
test_size = opt.test_size
plot_all = opt.plot_all

dev_id = [19, 20, 21, 23, 24, 25, 1, 4]
test_id = [22, 2, 3, 5, 6, 7, 8, 9, 10]

print("Loading trained model")
net = torch.load("trained.model")
net.cuda()

# opencv optical flow parameters
farneback_params = dict(
    winsize=20, iterations=1,
    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
    pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)


def generate_data_loader(split='train'):
    set_id = dev_id if split == 'train' else test_id

    for i, category in enumerate(CATEGORIES):
        # Get all files in current category's folder.
        folder_path = os.path.join(".", "dataset", category)
        filenames = sorted(os.listdir(folder_path))

        for filename in filenames:
            filepath = os.path.join(".", "dataset", category, filename)

            # Get id of person in this video.
            person_id = int(filename.split("_")[0][6:])
            if person_id not in set_id:
                continue

            # Video, Category, Filename
            yield (imageio.get_reader(filepath, "ffmpeg"), i)


print('==> Preparing data..')
train_loader = generate_data_loader(split='train')
test_loader = generate_data_loader(split='test')


# normalize = transforms.Normalize(mean=net.mean, std=net.std)
# idx = np.arange(50000)
# np.random.shuffle(idx)

# len == 192
# train_frames = pickle.load(open("data/dev.p", "rb"))
# train_flows = pickle.load(open("data/dev_flow.p", "rb"))

# len == 216
# test_frames = pickle.load(open("data/test.p", "rb"))
# test_flows = pickle.load(open("data/test_flow.p", "rb"))

# train_loader = [([0] + train_frames[i]['frames'], [0] + train_flows[i]['flow_x'], [0] + train_flows[i]['flow_y'],
#                  train_frames[i]['category']) for i in range(len(train_frames))]
# test_loader = [([0] + test_frames[i]['frames'], [0] + test_flows[i]['flow_x'], [0] + test_flows[i]['flow_y'],
#                 test_frames[i]['category']) for i in range(len(test_frames))]

# min_in, max_in = net.input_range[0], net.input_range[1]
# min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
# mean, std = np.array(net.mean), np.array(net.std)
# min_out, max_out = np.min((min_in - mean) / std), np.max((max_in - mean) / std)


def forward(x, grad=False, rescue=(False, False, False)):
    frames, flow_x, flow_y = x
    if rescue[0]:
        frames = frames.cpu().numpy()
    if rescue[1]:
        flow_x = flow_x.cpu().numpy()
    if rescue[2]:
        flow_y = flow_y.cpu().numpy()
    flow_x = [0] + flow_x
    flow_y = [0] + flow_y

    # Class probabilities.
    prob = np.zeros(6, dtype=np.float32)
    prob = torch.Tensor(prob).cuda()

    current_block_frame = []
    current_block_flow_x = []
    current_block_flow_y = []
    cnt = 0
    grad_frames = []
    grad_flow_x = []
    grad_flow_y = []

    for i_frame in range(len(frames)):
        current_block_frame.append(frames[i_frame])

        if i_frame % 15 > 0:
            current_block_flow_x.append(flow_x[i_frame])
            current_block_flow_y.append(flow_y[i_frame])

        if (i_frame + 1) % 15 == 0:
            cnt += 1

            current_block_frame = np.array(
                current_block_frame,
                dtype=np.float32).reshape((1, 15, 60, 80))

            current_block_flow_x = np.array(
                current_block_flow_x,
                dtype=np.float32).reshape((1, 14, 30, 40))

            current_block_flow_y = np.array(
                current_block_flow_y,
                dtype=np.float32).reshape((1, 14, 30, 40))

            # current_block_frame -= train_dataset.mean["frames"]
            # current_block_flow_x -= train_dataset.mean["flow_x"]
            # current_block_flow_y -= train_dataset.mean["flow_y"]

            tensor_frames = torch.from_numpy(current_block_frame)
            tensor_flow_x = torch.from_numpy(current_block_flow_x)
            tensor_flow_y = torch.from_numpy(current_block_flow_y)

            instance_frames = torch.tensor(tensor_frames.unsqueeze(0).data, requires_grad=grad)
            instance_flow_x = torch.tensor(tensor_flow_x.unsqueeze(0).data, requires_grad=grad)
            instance_flow_y = torch.tensor(tensor_flow_y.unsqueeze(0).data, requires_grad=grad)

            # score = net(instance_frames, instance_flow_x, instance_flow_y).data[0].cpu().numpy()

            # score -= np.max(score)
            # softmax = np.e ** score / np.sum(np.e ** score)
            # prob += softmax

            score = F.softmax(net(instance_frames.cuda(), instance_flow_x.cuda(), instance_flow_y.cuda()), dim=1)
            prob += score[0]

            if grad:
                loss = -score[0][target]
                loss.backward()

                grad_frames.append(instance_frames.grad.clone())
                grad_flow_x.append(instance_flow_x.grad.clone())
                grad_flow_y.append(instance_flow_y.grad.clone())

                instance_frames.grad.data.zero_()
                instance_flow_x.grad.data.zero_()
                instance_flow_y.grad.data.zero_()


            current_block_frame = []
            current_block_flow_x = []
            current_block_flow_y = []

    prob /= cnt

    if grad:
        grad_frames = torch.cat(grad_frames, dim=2).cuda()
        grad_flow_x = torch.cat(grad_flow_x, dim=2).cuda()
        grad_flow_y = torch.cat(grad_flow_y, dim=2).cuda()
        return prob, grad_frames, grad_flow_x, grad_flow_y
    else:
        return prob


def parse_sequence_file():
    print("Parsing ./dataset/00sequences.txt")

    # Read 00sequences.txt file.
    with open('./dataset/00sequences.txt', 'r') as content_file:
        content = content_file.read()

    # Replace tab and newline character with space, then split file's content
    # into strings.
    content = re.sub("[\t\n]", " ", content).split()

    # Dictionary to keep ranges of frames with humans.
    # Example:
    # video "person01_boxing_d1": [(1, 95), (96, 185), (186, 245), (246, 360)].
    frames_idx = {}

    # Current video that we are parsing.
    current_filename = ""

    for s in content:
        if s == "frames":
            # Ignore this token.
            continue
        elif s.find("-") >= 0:
            # This is the token we are looking for. e.g. 1-95.
            if s[len(s) - 1] == ',':
                # Remove comma.
                s = s[:-1]

            # Split into 2 numbers => [1, 95]
            idx = s.split("-")

            # Add to dictionary.
            if current_filename not in frames_idx:
                frames_idx[current_filename] = []
            frames_idx[current_filename].append((int(idx[0]), int(idx[1])))
        else:
            # Parse next file.
            current_filename = s + "_uncomp.avi"

    return frames_idx


frames_idx = parse_sequence_file()


def process_input(video, patch=None, mask=None):
    frames = []
    flow_x = []
    flow_y = []

    if patch is not None:
        video = torch.mul((1 - mask), video) + torch.mul(mask, patch)
        frames = video

    prev_frame = None
    # Add each frame to correct list.
    for i, frame in enumerate(video):
        # Boolean flag to check if current frame contains human.
        # ok = False
        # for seg in frames_idx[filename]:
        #     if i >= seg[0] and i <= seg[1]:
        #         ok = True
        #         break
        # if not ok:
        #     continue

        # Add patch to the frame according to mask
        if patch is None:
            # Convert to grayscale.
            frame = Image.fromarray(np.array(frame))
            frame = frame.convert("L")
            frame = np.array(frame.getdata(),
                             dtype=np.uint8).reshape((120, 160))
            frame = imresize(frame, (60, 80))

            frames.append(frame)

        else:
            frame = frame.cpu().numpy()

        if prev_frame is not None:
            # Calculate optical flow.
            flows = cv2.calcOpticalFlowFarneback(prev_frame, frame,
                                                 **farneback_params)
            subsampled_x = np.zeros((30, 40), dtype=np.float32)
            subsampled_y = np.zeros((30, 40), dtype=np.float32)

            for r in range(30):
                for c in range(40):
                    subsampled_x[r, c] = flows[r * 2, c * 2, 0]
                    subsampled_y[r, c] = flows[r * 2, c * 2, 1]

            flow_x.append(subsampled_x)
            flow_y.append(subsampled_y)

        prev_frame = frame

    return frames, flow_x, flow_y


def train(patch):
    patch_shape = patch.shape
    net.eval()
    success = 0
    total = 0
    recover_time = 0

    # labels: int index of the list, not str!!!
    for batch_idx, (video, labels) in enumerate(train_loader):
        # if opt.cuda:
        #     data = data.cuda()
        #     labels = labels.cuda()
        # data, labels = Variable(data), Variable(labels)

        x = process_input(video)
        frames, flow_x, flow_y = x

        prob = forward(x)
        pred = np.argmax(prob.cpu().detach().numpy())

        # only computer adversarial examples on examples that are originally classified correctly
        if pred != labels:
            continue

        total += 1

        # TODO: currently fixed patch. Dynamic patch needs one more dimension.
        # Need to change to a higher dimensional matrix instead of using list.
        patch, mask = circle_transform(patch, np.shape(frames))
        patch = Variable(torch.FloatTensor(patch).cuda())
        mask = Variable(torch.FloatTensor(mask).cuda())
        # Note that after the transform, patch size changes to the entire video shape

        adv_x, mask, patch = attack(x, patch, mask)

        adv_prob = forward(adv_x, rescue=(True, False, False))
        adv_label = np.argmax(adv_prob.detach().cpu().numpy())
        # ori_label = labels

        if adv_label == target:
            success += 1

            # if plot_all == 1:
            #     # plot source image
            #     vutils.save_image(data.data, "./%s/%d_%d_original.png" % (opt.outf, batch_idx, ori_label),
            #                       normalize=True)
            #
            #     # plot adversarial image
            #     vutils.save_image(adv_x.data, "./%s/%d_%d_adversarial.png" % (opt.outf, batch_idx, adv_label),
            #                       normalize=True)

        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]):
            # for j in range(new_patch.shape[1]):
                # new_patch[i][j] = submatrix(patch[i][j])
            new_patch[i] = submatrix(patch[i])

        patch = new_patch
        print(patch.shape)

        # log to file
        progress_bar(batch_idx, 192, "Train Patch Success: {:.3f}".format(success / total))

    return patch


def test(patch, patch_shape):
    net.eval()
    success = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)

        prediction = net(data)

        # only computer adversarial examples on examples that are originally classified correctly
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue

        total += 1

        # transform path
        data_shape = data.data.cpu().numpy().shape
        patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, frame_height, frame_width, time_step)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        patch, mask = Variable(patch), Variable(mask)

        adv_x = torch.mul((1 - mask), data) + torch.mul(mask, patch)
        # adv_x = torch.clamp(adv_x, min_out, max_out)

        adv_label = net(adv_x).data.max(1)[1][0]
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
        progress_bar(batch_idx, len(test_loader), "Test Success: {:.3f}".format(success / total))


def attack(x, patch, mask):
    net.eval()

    frames, _, _ = x
    x_out = forward(x)
    target_prob = x_out[target]
    frames = Variable(torch.FloatTensor(frames).cuda())

    adv_frames, adv_flow_x, adv_flow_y = process_input(frames, patch=patch, mask=mask)

    count = 0
    lr = 10000

    while conf_target > target_prob:
        count += 1
        # adv_frames = Variable(adv_frames.data, requires_grad=True)
        # adv_flow_x = Variable(torch.FloatTensor(adv_flow_x).cuda(), requires_grad=True)
        # adv_flow_y = Variable(torch.FloatTensor(adv_flow_y).cuda(), requires_grad=True)
        adv_x = (adv_frames, adv_flow_x, adv_flow_y)
        adv_out, grad_frames, grad_flow_x, grad_flow_y = forward(adv_x, grad=True, rescue=(True, False, False))

        # adv_out_probs, adv_out_labels = adv_out.max(1)
        # if count > 150:
        #     lr = 100000

        # TODO is optical flow differentiable and backpropagatable?
        # patch -= ((grad_frames + grad_flow_x + grad_flow_y) / 3)
        try:
            patch -= grad_frames[0][0] * lr
        except Exception as e:
            patch[:grad_frames[0][0].shape[0], :, :] -= grad_frames[0][0] * lr

        adv_frames, adv_flow_x, adv_flow_y = process_input(frames, patch=patch, mask=mask)
        adv_x = (adv_frames, adv_flow_x, adv_flow_y)
        # adv_x = torch.clamp(adv_x, min_out, max_out)
        # TODO do we need clamp???

        x_out = forward(adv_x, rescue=(True, False, False))
        target_prob = x_out[target]
        y_argmax_prob = x_out.max()

        print(count, conf_target, target_prob, y_argmax_prob)

        if count >= opt.max_count:
            break

    return adv_x, mask, patch


if __name__ == '__main__':
    patch = init_patch_circle(frame_height, frame_width, patch_size)

    for epoch in range(1, opt.epochs + 1):
        patch = train(patch)
        # test(patch)
