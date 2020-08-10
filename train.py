import pandas as pd
import argparse
import torch
import torch.nn.init
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
from model import HardNet, HardNetSTN
from loss import *
from dataset import create_loaders
import random
import numpy as np
from tqdm import tqdm
from utils import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet')

# Recording settings
parser.add_argument('--name', type=str, default='')

# training hyperparameter
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--fliprot', type=str2bool, default=False)
parser.add_argument('--augmentation', type=str2bool, default=False)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
parser.add_argument('--gor', type=str2bool, default=False, help='use gor')
parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA', help='gor parameter')
parser.add_argument('--batch-size', type=int, default=1024, metavar='BS')
parser.add_argument('--n-triplets', type=int, default=5000000, metavar='N')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')

# environment settings
parser.add_argument('--training-set', default='liberty', help='Other options: notredame, yosemite')
parser.add_argument('--dataroot', type=str, default='data/sets/')
parser.add_argument('--enable-logging', type=str2bool, default=False, help='output to tensorlogger')
parser.add_argument('--log-dir', default='data/logs/')
parser.add_argument('--model-dir', default='data/models/', help='folder to output model checkpoints')
parser.add_argument('--experiment-name', default='liberty_train/', help='experiment path')

# Model options
parser.add_argument('--loss', default='triplet_margin', help='Other options: softmax, contrastive')
parser.add_argument('--batch-reduce', default='multi', type=str, help='Other options: adapt, multi, min, L2Net')
parser.add_argument('--ratio', default=0.2, type=float)
parser.add_argument('--neighbor', default=8, type=int)
parser.add_argument('--mag', default=8, type=int)
parser.add_argument('--clamp', default=8, type=int)
parser.add_argument('--type', default='min', type=str)
parser.add_argument('--square', default=True, type=bool)
parser.add_argument('--stn', default=False, type=bool)
parser.add_argument('--second', default=True, type=str2bool)
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers to be created')
parser.add_argument('--pin-memory', type=bool, default=True)
parser.add_argument('--decor', type=str2bool, default=False,
                    help='L2Net decorrelation penalty')
parser.add_argument('--imageSize', type=int, default=32)
parser.add_argument('--anchorswap', type=str2bool, default=True, help='turns on anchor swap')

# Device options
parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

suffix = args.experiment_name
triplet_flag = (args.batch_reduce == 'random_global') or args.gor

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

# create loggin directory
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# set random seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def train(train_loader, model, optimizer, epoch, logger, load_triplets=False):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:
        if load_triplets:
            data_a, data_p, data_n = data
        else:
            data_a, data_p = data

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
            out_a = model(data_a)
            out_p = model(data_p)
        if load_triplets:
            data_n = data_n.cuda()
            out_n = model(data_n)

        if args.batch_reduce == 'min':
            loss1 = loss_HardNet(out_a, out_p)
        elif args.batch_reduce == 'multi':
            loss1 = loss_myloss(out_a, out_p, neighbor=args.neighbor, square=args.square)
        else:
            loss1 = loss_adaption(out_a, out_p, mag=args.mag, square=args.square, clamp=args.clamp)

        if args.second:
            if args.batch_reduce == 'multi':
                loss2 = args.ratio * loss_myloss_second(out_a, out_p, neighbor=args.neighbor, square=args.square)
            elif args.batch_reduce == 'adapt':
                loss2 = args.ratio * loss_adaption_second(out_a, out_p, mag=args.mag, square=args.square, clamp=args.clamp)

        if args.second:
            loss = loss1 + loss2
        else:
            loss = loss1

        if args.gor:
            loss += args.alpha * global_orthogonal_regularization(out_a, out_n)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(args, optimizer)

        if batch_idx % args.log_interval == 0:
            if args.second:
                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]    Loss1: {:.6f}    Loss2: {:.6f}'.format(
                        epoch, batch_idx * len(data_a), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                        loss1.item(), loss2.item()))
            else:
                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}'.format(
                        epoch, batch_idx * len(data_a), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                        loss1.item()))

    if (args.enable_logging):
        logger.log_value('loss', loss.item()).step()

    try:
        os.stat('{}{}'.format(args.model_dir, suffix))
    except:
        os.makedirs('{}{}'.format(args.model_dir, suffix))

    if os.path.isdir('data/models/%s' % args.name) is False:
        os.makedirs('data/models/%s' % args.name)

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(args.model_dir, args.name, epoch))
    return loss.item()


def test(test_loader, model, epoch, logger, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for batch_idx, (data_a, data_p, label) in pbar:
            if args.cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()

            out_a = model(data_a)
            out_p = model(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy().reshape(-1, 1))
            ll = label.data.cpu().numpy().reshape(-1, 1)
            labels.append(ll)

            if batch_idx % args.log_interval == 0:
                pbar.set_description(logger_test_name + ' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data_a), len(test_loader.dataset),
                           100. * batch_idx / len(test_loader)))

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    print('In Epoch: {}, Test set: Accuracy(FPR95): {:.8f}'.format(epoch, fpr95))
    return fpr95


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer


def main(train_loader, test_loaders, model, logger):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    # continue training
    if args.resume != '':
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict['state_dict'])

    if args.cuda:
        model = model.cuda()
    # modify args.name by using hyperparameters
    set_arg_name(args)
    args.experiment_name = args.name
    print(args.name)

    optimizer1 = create_optimizer(model.features, args.lr)

    end = args.epochs
    log = pd.DataFrame(index=[], columns=['epoch', 'loss', 'fpr_notredame', 'fpr_yosemite'])

    for epoch in range(0, end):

        # iterate over test loaders and test results
        record_loss = train(train_loader, model, optimizer1, epoch, logger, triplet_flag)
        fpr = [0, 0]
        for i, test_loader in enumerate(test_loaders):
            fpr[i] = test(test_loader['dataloader'], model, epoch, logger, test_loader['name'])

        tmp = pd.Series([epoch, record_loss, fpr[0], fpr[1]],
                        index=['epoch', 'loss', 'fpr_notredame', 'fpr_yosemite'])

        log = log.append(tmp, ignore_index=True)
        # randomize train loader batches
        train_loader, test_loaders2 = create_loaders(args=args, load_random_triplets=triplet_flag)

    if os.path.isdir('models/%s' % args.name) is False:
        os.makedirs('models/%s' % args.name)
    log.to_csv('models/%s/log.csv' % args.name, index=False)

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

if __name__ == '__main__':
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = os.path.join(args.log_dir, suffix)
    DESCS_DIR = os.path.join(LOG_DIR, 'temp_descs')

    logger, file_logger = None, None
    if args.stn:
        model = HardNetSTN()
    else:
        model = HardNet()

    train_loader, test_loaders = create_loaders(args=args, load_random_triplets=triplet_flag)
    main(train_loader, test_loaders, model, logger)
